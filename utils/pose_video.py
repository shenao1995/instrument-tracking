"""Continuous articulated trajectories and OpenCV video I/O helpers."""
from pathlib import Path
import json
from collections import deque

import cv2
import numpy as np
import torch

from utils.pose_geometry import axis_rotation, vector_pose, part_transforms, PARTS
from utils.pose_data import natural_key, read_mask


def continuous_pose(times, base, k, size, min_depth, max_depth, rotation_degrees=5., seed=2026):
    """Evaluate an analytic trajectory at timestamps; independent of render batches.

    Units: seconds/metres/radians. Position, rotation and jaw motion have bounded
    smooth velocities. No per-frame random draws, pose resets or rejection sampling.
    """
    times = torch.as_tensor(times,device=base.device,dtype=base.dtype).reshape(-1)
    phase = float(np.random.default_rng(seed).uniform(-.4,.4))
    wave = lambda period,offset=0.: torch.sin(times*(2*np.pi/period)+offset)
    h,w = size
    z = (min_depth+max_depth)/2 + .3*(max_depth-min_depth)*wave(10.,phase)
    u = w*(.42+.10*wave(9.,phase))
    v = h*(.50+.12*wave(12.,-phase))
    pose = vector_pose(base[None].repeat(len(times),1))
    pose["t"] = torch.stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z),-1)
    amplitude = np.deg2rad(rotation_degrees)
    pose["R"] = (axis_rotation(amplitude*wave(11.,phase),"z") @
                 axis_rotation(amplitude*wave(13.),"y") @
                 axis_rotation(amplitude*wave(14.,-phase),"x") @ pose["R"])
    alpha = np.deg2rad(35.)*wave(9.5)
    half = np.deg2rad(23.)+np.deg2rad(21.)*wave(5.2,-np.pi/2)
    yaw = np.deg2rad(15.)*wave(7.3)
    pose["joints"] = torch.stack((alpha,half+yaw,half-yaw),-1)
    return pose


def open_video_writer(path, fps, size):
    """MP4V encoded MP4; semantic labels are saved separately as lossless PNGs."""
    h,w = size
    if min(h,w) < 2 or h % 2 or w % 2 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video requires positive FPS and even H/W >= 2")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Video already exists: {path}")
    writer = cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),float(fps),(w,h))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot open MP4V video writer: {path}")
    return writer


def resize_intrinsics(k, source_size, target_size):
    k = torch.as_tensor(k,dtype=torch.float32).clone()
    sy,sx = target_size[0]/source_size[0],target_size[1]/source_size[1]
    k[0] *= sx
    k[1] *= sy
    k[0,2] += .5*sx-.5
    k[1,2] += .5*sy-.5
    return k


def part_axis_anchors(mesh):
    """Display anchors in canonical part coordinates; orientations are unchanged.

    The shaft's actual frame origin is ~216 mm from the wrist and often off
    screen, so its display origin is shifted to 25 mm behind the wrist.
    Jaw anchors sit toward their distal ends, separating the two triads visually.
    """
    anchors = []
    for name,vertices in zip(PARTS,mesh.vertices.detach().cpu().split(mesh.counts)):
        low,high = vertices.amin(0),vertices.amax(0)
        anchor = (low+high)/2
        if name == "shaft":
            anchor = torch.tensor([.2159-.025,0.,0.],dtype=vertices.dtype)
        elif name.startswith("gripper"):
            anchor[0] = low[0]+.70*(high[0]-low[0])
        anchors.append(anchor)
    return torch.stack(anchors)


def part_axis_points(pose,anchors,length_m=.004):
    """Return B x 4 parts x (origin,X,Y,Z) x XYZ-camera, in metres."""
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError("Axis length must be positive and finite")
    transforms = part_transforms(pose)
    points = []
    anchors = anchors.to(device=pose["R"].device,dtype=pose["R"].dtype)
    for i,name in enumerate(PARTS):
        tf = transforms[name]
        rotation = tf[:,:3,:3]
        origin = (rotation @ anchors[i]).reshape(-1,3)+tf[:,:3,3]
        endpoints = origin[:,None,:]+length_m*rotation.transpose(1,2)
        points.append(torch.cat((origin[:,None,:],endpoints),1))
    return torch.stack(points,1)


def draw_part_axes(image,points_camera,k,near=.001):
    """Draw camera-projected local axes on one panel, with near/image clipping."""
    points = np.asarray(points_camera,dtype=np.float64)
    k = np.asarray(k,dtype=np.float64)
    h,w = image.shape[:2]
    colors = ((0,0,255),(0,255,0),(255,0,0))  # BGR: X red, Y green, Z blue
    offsets = ((-18,15),(-10,-10),(-25,-12),(8,20))
    def project(point):
        uv = k @ point
        return uv[:2]/uv[2]
    def pixel(point):
        return tuple(np.rint(np.clip(point,-1_000_000,1_000_000)).astype(int))
    def label(text,xy,color=(255,255,255)):
        cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
    for part,(name,points3d) in enumerate(zip(("S","W","L","R"),points)):
        if not np.isfinite(points3d).all():
            continue
        origin = points3d[0]
        for axis,color in enumerate(colors):
            start,end = origin.copy(),points3d[axis+1].copy()
            if start[2] < near and end[2] < near:
                continue
            if start[2] < near:
                start += (end-start)*((near-start[2])/(end[2]-start[2]))
            if end[2] < near:
                end += (start-end)*((near-end[2])/(start[2]-end[2]))
            p0,p1 = project(start),project(end)
            visible,a,b = cv2.clipLine((0,0,w,h),pixel(p0),pixel(p1))
            if not visible:
                continue
            # Draw a dark outline so all three colors remain readable over masks.
            cv2.arrowedLine(image,a,b,(0,0,0),4,cv2.LINE_AA,tipLength=.23)
            cv2.arrowedLine(image,a,b,color,2,cv2.LINE_AA,tipLength=.23)
            if end[2] > near and 0 <= p1[0] < w-12 and 0 <= p1[1] < h-8:
                label("XYZ"[axis],(int(p1[0])+3,int(p1[1])-3),color)
        if origin[2] >= near:
            uv = project(origin)
            if 0 <= uv[0] < w and 0 <= uv[1] < h:
                at = pixel(uv)
                cv2.circle(image,at,2,(255,255,255),-1,cv2.LINE_AA)
                dx,dy = offsets[part]
                label(name,(max(0,min(w-15,at[0]+dx)),max(12,min(h-5,at[1]+dy))))
    # Axis colors have a different meaning from the semantic mask palette.
    label("Axes: X",(8,h-25),colors[0])
    label("Y",(65,h-25),colors[1])
    label("Z",(80,h-25),colors[2])
    label("S shaft  W wrist  L/R jaws",(8,h-9))


class RollingFPS:
    """Throughput of completed frames, including work between frame boundaries."""
    def __init__(self,window=30):
        if window < 1:
            raise ValueError("FPS window must be positive")
        self.durations = deque(maxlen=window)
        self.last = None

    def start(self,timestamp):
        self.last = timestamp
        self.durations.clear()

    def complete(self,timestamp):
        if self.last is None or timestamp <= self.last:
            raise ValueError("Start FPS timer first and supply increasing completion timestamps")
        self.durations.append(timestamp-self.last)
        self.last = timestamp

    @property
    def fps(self):
        return len(self.durations)/sum(self.durations) if self.durations else None


def video_comparison(frame_bgr, predicted_mask, fps=None,axes_camera=None,axes_k=None):
    """Original on left; overlap and red measured end-to-end FPS on right."""
    h,w = frame_bgr.shape[:2]
    right = frame_bgr.copy()
    if predicted_mask is not None:
        mask = np.asarray(predicted_mask,dtype=np.float32).transpose(1,2,0)
        if mask.shape[:2] != (h,w):
            mask = cv2.resize(mask,(w,h),interpolation=cv2.INTER_LINEAR)
        mask = mask.clip(0,1)
        palette_bgr = np.array([[.15,.15,1.],[.15,1.,.15],[1.,.35,.15]],np.float32)
        color = mask @ palette_bgr
        alpha = mask.sum(-1,keepdims=True).clip(0,1)
        right = (frame_bgr.astype(np.float32)*(1-.4*alpha)+255*.4*color).clip(0,255).round().astype(np.uint8)
    if axes_camera is not None:
        if axes_k is None:
            raise ValueError("Camera intrinsics required for drawing axes")
        draw_part_axes(right,axes_camera,axes_k)
    panel = np.concatenate((frame_bgr,right),axis=1)
    text = f"FPS: {fps:.1f}" if fps is not None and np.isfinite(fps) and fps > 0 else "FPS: --"
    scale = max(.4,min(.85,w/640*.85))
    thickness = 2 if w >= 320 else 1
    (tw,th),_ = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thickness)
    cv2.putText(panel,text,(max(w,2*w-tw-10),th+10),cv2.FONT_HERSHEY_SIMPLEX,
                scale,(0,0,255),thickness,cv2.LINE_AA)
    return panel


class VideoSource:
    """Sequential video + one lossless semantic mask per decoded frame."""
    def __init__(self,path,mask_dir=None,metadata_path=None):
        self.path = Path(path)
        metadata_path = Path(metadata_path) if metadata_path else self.path.with_name(self.path.stem+"_metadata.json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
        if self.metadata and not self.metadata.get("complete"):
            raise ValueError("Video generation is incomplete")
        if mask_dir is None:
            if not self.metadata or "mask_dir" not in self.metadata:
                raise ValueError("This pose network needs per-frame semantic masks: provide --mask-dir or video metadata")
            mask_dir = metadata_path.parent/self.metadata["mask_dir"]
        self.masks = sorted([p for p in Path(mask_dir).iterdir() if p.suffix.lower() in (".png",".tif",".tiff",".bmp")],key=natural_key)
        if not self.masks:
            raise ValueError(f"No video masks in {mask_dir}")
        if self.metadata and self.metadata.get("kind") == "synthetic_pose_video":
            expected = [self.metadata["mask_pattern"].format(index=i) for i in range(self.metadata["frames"])]
            if [p.name for p in self.masks] != expected:
                raise ValueError("Generated video mask names/count do not match metadata")
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot open video: {self.path}")
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        if not np.isfinite(self.fps) or self.fps <= 0 or min(self.size) < 2:
            self.close()
            raise ValueError("Video has invalid dimensions or FPS")
        if self.frame_count > 0 and self.frame_count != len(self.masks):
            self.close()
            raise ValueError(f"Video has {self.frame_count} frames but {len(self.masks)} masks; align every frame before inference")
        if self.metadata and (self.metadata["image_size"] != list(self.size) or
                              self.metadata["frames"] != len(self.masks) or
                              not np.isclose(self.metadata["fps"],self.fps,rtol=1e-3)):
            self.close()
            raise ValueError("Video dimensions/FPS/frame count differ from metadata")

    def restart(self):
        self.capture.release()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise ValueError(f"Cannot reopen video: {self.path}")

    def read(self,index,network_size):
        ok,frame = self.capture.read()
        if not ok:
            return None
        if index >= len(self.masks):
            raise ValueError("Video contains more frames than semantic masks")
        mask = read_mask(self.masks[index],network_size)
        rgb = cv2.cvtColor(cv2.resize(frame,network_size[::-1],interpolation=cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)
        rgb = torch.from_numpy(rgb.copy()).permute(2,0,1).float()/255
        rgb *= mask.sum(0,keepdim=True).clamp(0,1)
        return frame,{"rgb":rgb[None],"mask":mask[None]}

    def close(self):
        self.capture.release()
