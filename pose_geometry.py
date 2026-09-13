"""Batched Instrument-Splatting-compatible kinematics and semantic mesh rendering.

Coordinates: metres, radians, OpenCV camera (+x right, +y down, +z forward).
R/t describe WRIST to CAMERA, not camera to world. No Gaussian dependencies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PARTS = ("shaft", "wrist", "gripper_left", "gripper_right")
LABELS = (0, 1, 2, 2)


def rotation_6d_to_matrix(d):
    """First two COLUMNS; Gram-Schmidt with a finite orthogonal fallback."""
    a, b = d[..., :3], d[..., 3:6]
    unit_x = torch.zeros_like(a)
    unit_x[..., 0] = 1
    x = F.normalize(torch.where(a.norm(dim=-1, keepdim=True) > 1e-6, a, unit_x), dim=-1)
    b = b - (x * b).sum(-1, keepdim=True) * x
    fallback = F.one_hot(x.abs().argmin(-1), 3).to(x)
    fallback = fallback - (fallback * x).sum(-1, keepdim=True) * x
    y = F.normalize(torch.where(b.norm(dim=-1, keepdim=True) > 1e-6, b, fallback), dim=-1)
    return torch.stack((x, y, torch.linalg.cross(x, y)), dim=-1)


def matrix_to_rotation_6d(r):
    return torch.cat((r[..., :, 0], r[..., :, 1]), dim=-1)


def axis_rotation(angle, axis):
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(c), torch.ones_like(c)
    if axis == "y":
        values = (c, z, s, z, o, z, -s, z, c)
    elif axis == "z":
        values = (c, -s, z, s, c, z, z, z, o)
    else:
        values = (o, z, z, z, c, -s, z, s, c)
    return torch.stack(values, -1).reshape(*angle.shape, 3, 3)


def pose_vector(pose):
    return torch.cat((matrix_to_rotation_6d(pose["R"]), pose["t"], pose["joints"]), -1)


def vector_pose(vector):
    return {"R": rotation_6d_to_matrix(vector[..., :6]), "t": vector[..., 6:9], "joints": vector[..., 9:12]}


def detach_pose(pose):
    return {k: v.detach() for k, v in pose.items()}


def part_transforms(pose):
    """Canonical part-local to OpenCV camera transforms, matching InstrumentMesh.

    Three semantic classes contain four rigid bodies: the two jaws rotate
    independently. Canonical coordinates include the OBJ alignment in __init__.
    """
    r,t,a = pose["R"],pose["t"],pose["joints"]
    shaft_r = r @ axis_rotation(-a[:,0],"y")
    pivot = t+(r @ t.new_tensor([.009,0,0])).reshape(-1,3)
    rotations = (shaft_r,r,r @ axis_rotation(a[:,1],"z"),r @ axis_rotation(-a[:,2],"z"))
    translations = (t-(shaft_r @ t.new_tensor([.2159,0,0])).reshape(-1,3),t,pivot,pivot)
    result = {}
    for name,rotation,translation in zip(PARTS,rotations,translations):
        transform = torch.eye(4,device=r.device,dtype=r.dtype)[None].repeat(len(r),1,1)
        transform[:,:3,:3],transform[:,:3,3] = rotation,translation
        result[name] = transform
    return result


def project(points, k):
    p = points @ k.transpose(-1, -2)
    return p[..., :2] / p[..., 2:].clamp_min(1e-5)


@torch.no_grad()
def visible_tips(vertices, faces, tips, near=.001, tolerance=.00075):
    """Ray/triangle visibility, including occluders whose vertices miss the ray."""
    tri = vertices[:, faces]
    a, b, c = tri.unbind(-2)
    e1, e2 = b-a, c-a
    rays = tips / tips[..., 2:].clamp_min(near)
    h = torch.linalg.cross(rays[:,:,None],e2[:,None],dim=-1)
    determinant = (e1[:,None]*h).sum(-1)
    valid_det = determinant.abs() > 1e-12
    inv = torch.where(valid_det,1/determinant.masked_fill(~valid_det,1),0)
    u = (-a[:,None]*h).sum(-1)*inv
    q = torch.linalg.cross(-a,e1,dim=-1)
    v = (rays[:,:,None]*q[:,None]).sum(-1)*inv
    depth = (e2*q).sum(-1)[:,None]*inv
    hit = valid_det & (u >= 0) & (v >= 0) & (u+v <= 1) & (depth >= near)
    nearest = depth.masked_fill(~hit,float("inf")).amin(-1)
    return (tips[...,2] > near) & (nearest >= tips[...,2]-tolerance)


def load_calibration(path=ROOT / "data/surgpose_sample/transforms.json", image_size=(256, 320)):
    from PIL import Image
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    frame = meta["frames"][0]
    first = path.parent / frame["file_path"]
    if not first.suffix:
        first = first.with_suffix(".png")
    with Image.open(first) as im:
        w, h = im.size
    k = np.asarray(meta["intrinsic_matrix"], dtype=np.float32)
    # Pixel-centre resize convention, consistent with OpenCV/PIL resizing.
    sx, sy = image_size[1] / w, image_size[0] / h
    k[0] *= sx
    k[1] *= sy
    k[0, 2] += (sx - 1) / 2
    k[1, 2] += (sy - 1) / 2
    transform = np.asarray(frame["transform_matrix"], dtype=np.float32)
    # Rounded reference rotations are projected back to SO(3).
    u, _, vt = np.linalg.svd(transform[:3, :3])
    u[:, -1] *= np.linalg.det(u @ vt)
    r = torch.from_numpy(u @ vt)
    base = torch.cat((matrix_to_rotation_6d(r), torch.from_numpy(transform[:3, 3]),
                      torch.tensor([0., np.deg2rad(5), np.deg2rad(5)], dtype=torch.float32)))
    return torch.from_numpy(k), base


class InstrumentMesh(nn.Module):
    def __init__(self, mesh_dir=ROOT / "data/instrument_mesh", faces_per_part=0,load_appearance=False):
        super().__init__()
        import trimesh
        self.mesh_dir = str(Path(mesh_dir).resolve())
        self.faces_per_part = faces_per_part
        self.load_appearance = load_appearance
        self.appearance_hashes = {}
        if load_appearance and faces_per_part:
            raise ValueError("Material RGB rendering currently requires --faces-per-part 0")
        all_normals,all_albedo = [],[]
        self.counts = []
        all_v, all_f, all_labels, tips = [], [], [], []
        offset = 0
        hashes = {}
        bias = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
        for i, part in enumerate(PARTS):
            path = Path(mesh_dir) / f"transformed_{part}.obj"
            hashes[part] = hashlib.sha256(path.read_bytes()).hexdigest()
            if load_appearance:
                from pose_appearance import load_material_mesh
                mesh,material_hashes = load_material_mesh(path)
                self.appearance_hashes.update(material_hashes)
            else:
                mesh = trimesh.load(path, force="mesh", process=True)
            # OBJ normal/UV seams duplicate positions. Weld them before geometry
            # decimation, otherwise disconnected triangles collapse independently.
            mesh.merge_vertices(merge_tex=True, merge_norm=True)
            mesh.update_faces(mesh.unique_faces())
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            v = np.asarray(mesh.vertices) @ bias.T
            if part == "shaft":
                v += [0.2159, 0, 0]
            elif part == "wrist":
                v *= [1, -1, -1]
            else:
                v -= [0.009, 0, 0]
                # Distal cap centre on ORIGINAL mesh, independent of simplification.
                tips.append(v[v[:, 0] >= v[:, 0].max() - 0.0003].mean(0))
            mesh.vertices = v
            if faces_per_part and len(mesh.faces) > faces_per_part:
                try:
                    mesh = mesh.simplify_quadric_decimation(face_count=faces_per_part)
                except ImportError as exc:
                    raise RuntimeError("Mesh simplification requires: pip install fast-simplification") from exc
            v = np.asarray(mesh.vertices, dtype=np.float32)
            if load_appearance:
                all_normals.append(np.asarray(mesh.vertex_normals,dtype=np.float32))
                all_albedo.append(np.asarray(mesh.visual.face_colors[:,:3],dtype=np.float32)/255)
            all_v.append(v)
            all_f.append(np.asarray(mesh.faces, dtype=np.int64) + offset)
            all_labels.extend([LABELS[i]] * len(v))
            self.counts.append(len(v))
            offset += len(v)
        self.hashes = hashes
        self.register_buffer("vertices", torch.from_numpy(np.concatenate(all_v)))
        self.register_buffer("faces", torch.from_numpy(np.concatenate(all_f)))
        self.register_buffer("attributes", F.one_hot(torch.tensor(all_labels), 3).float())
        self.register_buffer("local_tips", torch.tensor(np.asarray(tips), dtype=torch.float32))
        if load_appearance:
            self.register_buffer("normals",torch.from_numpy(np.concatenate(all_normals)))
            self.register_buffer("face_albedo",torch.from_numpy(np.concatenate(all_albedo)))

    def camera_normals(self,pose):
        r,a = pose["R"],pose["joints"]
        n = self.normals.split(self.counts)
        rotations = [r @ axis_rotation(-a[:,0],"y"),r,
                     r @ axis_rotation(a[:,1],"z"),r @ axis_rotation(-a[:,2],"z")]
        return torch.cat([v @ rotation.transpose(-1,-2) for v,rotation in zip(n,rotations)],1)

    def forward(self, pose):
        r, t, angles = pose["R"], pose["t"], pose["joints"]
        canonical = self.vertices.split(self.counts)
        transformed = []
        # T_shaft_camera = T_wrist_camera @ inverse(T_wrist_shaft).
        shaft = (canonical[0] - canonical[0].new_tensor([.2159, 0, 0])) @ axis_rotation(-angles[:, 0], "y").transpose(-1, -2)
        transformed.append(shaft @ r.transpose(-1, -2) + t[:, None])
        transformed.append(canonical[1] @ r.transpose(-1, -2) + t[:, None])
        tip_positions = []
        for idx, sign in ((2, 1), (3, -1)):
            gr = axis_rotation(sign * angles[:, idx - 1], "z")
            shift = t.new_tensor([.009, 0, 0])
            v = canonical[idx] @ gr.transpose(-1, -2) + shift
            transformed.append(v @ r.transpose(-1, -2) + t[:, None])
            tip = self.local_tips[idx - 2][None, None] @ gr.transpose(-1, -2) + shift
            tip_positions.append((tip @ r.transpose(-1, -2) + t[:, None])[:, 0])
        return torch.cat(transformed, 1), torch.stack(tip_positions, 1)


def clip_triangles_near(tri, attrs, near):
    """Differentiable near-plane clipping; topology selection is discrete."""
    inside = tri[..., 2] >= near
    count = inside.sum(-1)
    out, labels = [tri[count == 3]], [attrs[count == 3]]
    for n in (1, 2):
        group = tri[count == n]
        if not len(group):
            continue
        flags = inside[count == n]
        # Put the exceptional vertex first, preserving cyclic winding.
        first = (flags if n == 1 else ~flags).long().argmax(-1)
        order = (first[:, None] + torch.arange(3, device=tri.device)) % 3
        g = group.gather(1, order[..., None].expand(-1, -1, 3))
        a, b, c = g.unbind(1)
        def intersect(v):
            return a + ((near - a[:, 2]) / (v[:, 2] - a[:, 2]))[:, None] * (v - a)
        ab, ac = intersect(b), intersect(c)
        label = attrs[count == n]
        if n == 1:
            out.append(torch.stack((a, ab, ac), 1))
            labels.append(label)
        else:
            out.extend((torch.stack((ab, b, c), 1), torch.stack((ab, c, ac), 1)))
            labels.extend((label, label))
    return torch.cat(out), torch.cat(labels)


class SemanticRenderer(nn.Module):
    """Opaque semantic z-buffer with differentiable silhouette antialiasing.

    Visibility/topology choices are discrete. Coverage at selected visible edges
    is an explicit differentiable function of the projected mesh, NOT a detached
    blurred mask or a straight-through gradient. Internal edges of the same label
    do not alter coverage. Supersampling resolves subpixel jaws and silhouettes.
    """
    VERSION = 3
    RGB_VERSION = 4

    def __init__(self, mesh, image_size=(256, 320), backend="nvdiffrast", tile_size=32,
                 supersample=2, edge_width=1.0,render_rgb=False):
        super().__init__()
        self.mesh = mesh
        self.render_rgb = render_rgb
        if render_rgb and not getattr(mesh,"load_appearance",False):
            raise ValueError("RGB rendering requires InstrumentMesh(load_appearance=True)")
        self.image_size = tuple(image_size)
        if not isinstance(supersample, int) or supersample < 1:
            raise ValueError("supersample must be a positive integer")
        if edge_width <= 0 or tile_size < 1:
            raise ValueError("edge_width and tile_size must be positive")
        self.tile_size, self.supersample, self.edge_width = tile_size, supersample, edge_width
        self.raster_size = tuple(supersample*s for s in self.image_size)
        self.near, self.far = .001, 2.
        if backend == "auto":
            # Keep the old CLI spelling, but never silently train on a different
            # renderer when the requested production dependency is missing.
            backend = "nvdiffrast"
        self.backend = backend
        if render_rgb and backend != "nvdiffrast":
            raise ValueError("Differentiable material RGB rendering requires --renderer nvdiffrast")
        self.ctx = None
        if backend == "nvdiffrast":
            if not mesh.vertices.is_cuda:
                raise ValueError("nvdiffrast requires --device cuda")
            try:
                import nvdiffrast.torch as dr
            except ImportError as exc:
                raise RuntimeError("nvdiffrast is required. See README.md for installation; "
                                   "on this Windows machine use .venv/Scripts/python.exe. "
                                   "Select --renderer torch explicitly only for reference/CPU checks.") from exc
            self.dr = dr
            self.ctx = dr.RasterizeCudaContext(device=mesh.vertices.device)
            self.register_buffer("raster_faces",mesh.faces.to(torch.int32).contiguous())
            self.topology_hash = dr.antialias_construct_topology_hash(self.raster_faces)
            # No per-frame face conversions or topology reconstruction.
            return
        elif backend != "torch":
            raise ValueError(f"Unknown renderer: {backend}")
        else:
            warnings.warn("Using portable torch triangle renderer; use nvdiffrast for throughput.", stacklevel=2)
        # Cache topology once. Silhouette edges separate front/back facing faces,
        # or lie on an open mesh boundary. Nonmanifold edges are retained safely;
        # the semantic visibility check below rejects occluded/internal edges.
        faces = mesh.faces.detach().cpu().numpy()
        face_edges = faces[:, [[0,1],[1,2],[2,0]]].reshape(-1,2)
        edges, inverse, counts = np.unique(np.sort(face_edges,axis=1),axis=0,return_inverse=True,return_counts=True)
        device = mesh.vertices.device
        self.register_buffer("edges",torch.as_tensor(edges,device=device))
        self.register_buffer("edge_inverse",torch.as_tensor(inverse,device=device))
        self.register_buffer("edge_counts",torch.as_tensor(counts,device=device))

    def configuration(self):
        config = {"version":self.VERSION,"backend":self.backend,"supersample":self.supersample,
                  "edge_width":self.edge_width,"near":self.near,"far":self.far}
        if self.render_rgb:
            config.update(version=self.RGB_VERSION,appearance="mtl_kd_smooth_headlight_v1",
                          material_sha256=self.mesh.appearance_hashes)
        return config

    def forward(self, pose, k):
        vertices, tips = self.mesh(pose)
        if k.ndim == 2:
            k = k[None].expand(len(vertices), -1, -1)
        raster_k = k.clone()
        raster_k[:, :2] *= self.supersample
        raster_k[:, :2, 2] += (self.supersample-1)/2
        if self.backend == "nvdiffrast":
            mask = self._nvdiffrast(vertices, raster_k,pose)
        else:
            mask = torch.stack([self._torch(v, camera) for v, camera in zip(vertices, raster_k)])
        if self.supersample > 1:
            mask = F.avg_pool2d(mask,self.supersample)
        output = {"mask":mask[:,:3],"tips":project(tips,k),"tips_camera":tips}
        if self.render_rgb:
            # Already premultiplied by pixel coverage, including soft edges.
            output["rgb"] = mask[:,3:6]
        return output

    def _nvdiffrast(self, vertices, k,pose=None):
        h, w = self.raster_size
        p = vertices @ k.transpose(-1, -2)
        z = vertices[..., 2]
        # nvdiffrast rows run bottom-to-top: flip image on output.
        clip = torch.stack((2 * (p[..., 0] + .5 * z) / w - z,
                            z - 2 * (p[..., 1] + .5 * z) / h,
                            (self.far + self.near) / (self.far - self.near) * z -
                            2 * self.far * self.near / (self.far - self.near), z), -1).contiguous()
        faces = self.raster_faces
        rast, _ = self.dr.rasterize(self.ctx, clip, faces, resolution=[h, w],grad_db=False)
        # Attributes are constant semantic one-hots, broadcast across poses.
        attrs = self.mesh.attributes[None].contiguous()
        color, _ = self.dr.interpolate(attrs, rast, faces)
        if self.render_rgb:
            normals,_ = self.dr.interpolate(self.mesh.camera_normals(pose).contiguous(),rast,faces)
            positions,_ = self.dr.interpolate(vertices.contiguous(),rast,faces)
            normals = F.normalize(normals,dim=-1,eps=1e-6)
            view = F.normalize(-positions,dim=-1,eps=1e-6)
            diffuse = (normals*view).sum(-1,keepdim=True).abs().clamp(0,1)
            face_id = (rast[...,3].long()-1).clamp_min(0)
            albedo = self.mesh.face_albedo[face_id]
            # A camera-aligned, two-sided headlight gives smooth shape cues.
            # These are CAD material colors, not recovered endoscopic textures.
            rgb = (albedo*(.35+.65*diffuse)+.12*diffuse.pow(32)).clamp(0,1)
            rgb = rgb*(rast[...,3:] > 0)
            color = torch.cat((color,rgb),-1)
        if torch.is_grad_enabled() and clip.requires_grad and len(vertices) > 1:
            # nvdiffrast 0.4.0 AA backward coalesces by (triangle, edge), without
            # the instance ID. A warp processing two instances can mix their
            # vertex gradients. Keep rasterization/interpolation batched, but
            # isolate differentiable AA per sample. The native library remains
            # unmodified; inference/data generation can use batched AA safely.
            color = torch.cat([self.dr.antialias(color[i:i+1].contiguous(),rast[i:i+1],
                               clip[i:i+1],faces,topology_hash=self.topology_hash)
                               for i in range(len(vertices))],dim=0)
        else:
            color = self.dr.antialias(color.contiguous(),rast,clip,faces,topology_hash=self.topology_hash)
        return color.flip(1).permute(0, 3, 1, 2).clamp(0, 1)

    @torch.no_grad()
    def _hard_visibility(self, vertices, k):
        """Perspective-correct opaque depth/label rasterization; no soft faces."""
        h, w = self.raster_size
        tri = vertices[self.mesh.faces]
        attrs = self.mesh.attributes[self.mesh.faces[:, 0]]
        tri, attrs = clip_triangles_near(tri, attrs, self.near)
        if not len(tri):
            return torch.full((h,w),3,device=vertices.device,dtype=torch.long), vertices.new_full((h,w),float("inf"))
        uv = project(tri, k)
        bounds_min, bounds_max = uv.amin(1), uv.amax(1)
        labels = torch.full((h,w),3,device=vertices.device,dtype=torch.long)
        zbuffer = vertices.new_full((h,w),float("inf"))
        part = attrs.argmax(-1)
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                th, tw = min(self.tile_size, h-y), min(self.tile_size, w-x)
                keep = ((bounds_max[:, 0] >= x) & (bounds_min[:, 0] < x+tw) &
                        (bounds_max[:, 1] >= y) & (bounds_min[:, 1] < y+th))
                if not keep.any():
                    continue
                a, b, c = uv[keep].unbind(1)
                yy, xx = torch.meshgrid(torch.arange(y,y+th,device=vertices.device),
                                        torch.arange(x,x+tw,device=vertices.device), indexing="ij")
                pixels = torch.stack((xx, yy), -1).reshape(-1, 2).to(vertices)
                def cross(u, v):
                    return u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]
                area = cross(b-a, c-a)
                valid_area = area.abs() > 1e-8
                safe_area = area.masked_fill(~valid_area,1)
                wb = cross(pixels[None]-a[:, None], (c-a)[:, None]) / safe_area[:, None]
                wc = cross((b-a)[:, None], pixels[None]-a[:, None]) / safe_area[:, None]
                bary = torch.stack((1-wb-wc, wb, wc), -1)
                inside = (bary >= -1e-6).all(-1) & valid_area[:,None]
                depth = 1 / (bary / tri[keep, :, 2][:, None]).sum(-1).clamp_min(1e-6)
                depth = depth.masked_fill(~inside | (depth < self.near) | (depth > self.far),float("inf"))
                front, index = depth.min(0)
                selected = torch.where(torch.isfinite(front),part[keep][index],3)
                labels[y:y+th,x:x+tw] = selected.reshape(th,tw)
                zbuffer[y:y+th,x:x+tw] = front.reshape(th,tw)
        return labels, zbuffer

    def _silhouette_edges(self, vertices):
        with torch.no_grad():
            tri = vertices[self.mesh.faces]
            normal = torch.linalg.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1)
            facing = ((normal*tri[:,0]).sum(-1) >= 0).long().repeat_interleave(3)
            lo = torch.ones(len(self.edges),device=vertices.device,dtype=torch.long)
            hi = torch.zeros_like(lo)
            lo.scatter_reduce_(0,self.edge_inverse,facing,reduce="amin")
            hi.scatter_reduce_(0,self.edge_inverse,facing,reduce="amax")
            keep = (lo != hi) | (self.edge_counts != 2)
        indices = self.edges[keep]
        edges = vertices[indices]
        valid = (edges[...,2] >= self.near).any(-1)
        edges, indices = edges[valid], indices[valid]
        a,b = edges.unbind(1)
        # Clip edges intersecting the near plane while retaining geometric grads.
        crossing = (a[:,2] < self.near) ^ (b[:,2] < self.near)
        denominator = (b[:,2]-a[:,2]).masked_fill(~crossing,1)
        hit = a + ((self.near-a[:,2])/denominator)[:,None]*(b-a)
        a = torch.where((a[:,2] < self.near)[:,None],hit,a)
        b = torch.where((b[:,2] < self.near)[:,None],hit,b)
        return torch.stack((a,b),1), self.mesh.attributes[indices[:,0]].argmax(-1)

    def _torch(self, vertices, k):
        labels,zbuffer = self._hard_visibility(vertices,k)
        edges,edge_labels = self._silhouette_edges(vertices)
        h,w = self.raster_size
        hard = F.one_hot(labels,4)[...,:3].permute(2,0,1).to(vertices)
        if not len(edges):
            return hard + vertices.sum()*0
        uv = project(edges,k)
        radius = self.edge_width/2  # in HIGH-resolution pixels
        bounds_min = uv.detach().amin(1)-radius
        bounds_max = uv.detach().amax(1)+radius
        palette = torch.cat((torch.eye(3,device=vertices.device),vertices.new_zeros(1,3)),0)
        rows = []
        for y in range(0,h,self.tile_size):
            tiles = []
            for x in range(0,w,self.tile_size):
                th,tw = min(self.tile_size,h-y),min(self.tile_size,w-x)
                keep = ((bounds_max[:,0] >= x)&(bounds_min[:,0] < x+tw)&
                        (bounds_max[:,1] >= y)&(bounds_min[:,1] < y+th))
                original = hard[:,y:y+th,x:x+tw]
                if not keep.any():
                    tiles.append(original+vertices.sum()*0)
                    continue
                a,b = uv[keep].unbind(1)
                direction = b-a
                normal = F.normalize(torch.stack((-direction[:,1],direction[:,0]),-1),dim=-1)
                yy,xx = torch.meshgrid(torch.arange(y,y+th,device=vertices.device),
                                      torch.arange(x,x+tw,device=vertices.device),indexing="ij")
                pixels = torch.stack((xx,yy),-1).reshape(-1,2).to(vertices)
                delta = pixels[None]-a[:,None]
                fraction = ((delta*direction[:,None]).sum(-1)/direction.square().sum(-1)[:,None].clamp_min(1e-12)).clamp(0,1)
                closest = a[:,None]+fraction[...,None]*direction[:,None]
                displacement = pixels[None]-closest
                distance = displacement.square().sum(-1).clamp_min(1e-12).sqrt()
                with torch.no_grad():
                    def sample(offset):
                        location = (closest+offset*normal[:,None]).round().long()
                        sx,sy = location[...,0],location[...,1]
                        inside = (sx >= 0)&(sx < w)&(sy >= 0)&(sy < h)
                        sx,sy = sx.clamp(0,w-1),sy.clamp(0,h-1)
                        return labels[sy,sx].masked_fill(~inside,3),zbuffer[sy,sx].masked_fill(~inside,float("inf"))
                    plus,zplus = sample(max(.75,radius+.25))
                    minus,zminus = sample(-max(.75,radius+.25))
                    own = edge_labels[keep,None]
                    edge_depth = 1/((1-fraction)/edges[keep,0,2,None]+fraction/edges[keep,1,2,None])
                    # Reject rear silhouettes and internal edges with identical
                    # visible semantics on both sides. The tolerance permits the
                    # pixel-centre depth change across a sloped face.
                    own_depth = torch.minimum(zplus.masked_fill(plus != own,float("inf")),
                                              zminus.masked_fill(minus != own,float("inf")))
                    current = labels[y:y+th,x:x+tw].reshape(1,-1)
                    valid = ((plus != minus)&((plus == own)|(minus == own))&
                             (edge_depth <= own_depth+.00075)&
                             ((current == plus)|(current == minus))&(distance <= radius))
                    best_distance,best = distance.masked_fill(~valid,float("inf")).min(0)
                    found = torch.isfinite(best_distance)
                selected_distance = distance.gather(0,best[None])[0]
                signed = (displacement*normal[:,None]).sum(-1).gather(0,best[None])[0]
                signed_distance = torch.where(signed >= 0,selected_distance,-selected_distance)
                weight = (.5+signed_distance/self.edge_width).clamp(0,1)
                plus_label = plus.gather(0,best[None])[0]
                minus_label = minus.gather(0,best[None])[0]
                antialiased = palette[plus_label]*weight[:,None]+palette[minus_label]*(1-weight[:,None])
                color = torch.where(found[:,None],antialiased,original.reshape(3,-1).T)
                tiles.append(color.T.reshape(3,th,tw)+vertices.sum()*0)
            rows.append(torch.cat(tiles,-1))
        return torch.cat(rows,-2).clamp(0,1)
