"""Generate consistent articulated mesh RGB, semantic masks, poses and tip labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from utils.pose_geometry import (ROOT, InstrumentMesh, SemanticRenderer, axis_rotation,
                           load_calibration, pose_vector, vector_pose, visible_tips)
from utils.pose_data import perturb_pose, save_mask, seed_everything


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"data/synthetic_rgb_pose")
    p.add_argument("--mesh-dir", type=Path, default=ROOT/"data/instrument_mesh")
    p.add_argument("--calibration", type=Path, default=ROOT/"data/surgpose_sample/transforms.json")
    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--val-fraction", type=float, default=.1)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--faces-per-part", type=int, default=0, help="0 (default) keeps original meshes")
    p.add_argument("--renderer", choices=("auto","torch","nvdiffrast"), default="nvdiffrast")
    p.add_argument("--supersample",type=int,choices=(1,2,4),default=2,
                   help="Rasterize at this multiple of each dimension, then area-downsample")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--rotation-range", type=float, default=60, help="Degrees around initial orientation; use 180 for broad rotations")
    p.add_argument("--min-depth", type=float, default=.045)
    p.add_argument("--max-depth", type=float, default=.15)
    p.add_argument("--max-attempt-factor", type=int, default=30)
    p.add_argument("--video",action="store_true",help="Generate a continuous video plus synchronized masks instead of independent samples")
    p.add_argument("--duration",type=float,default=15.,help="Video duration in seconds")
    p.add_argument("--fps",type=float,default=30.,help="Video playback frame rate")
    p.add_argument("--motion-rotation",type=float,default=5.,help="Video smooth orientation amplitude in degrees; independent of --rotation-range")
    return p


@torch.no_grad()
def generate(args):
    if args.video:
        return generate_video(args)
    if args.num_samples < 2 or not 0 < args.val_fraction < 1:
        raise ValueError("Need >=2 samples and 0 < val-fraction < 1")
    if min(args.height,args.width) < 32 or args.batch_size < 1:
        raise ValueError("Image dimensions must be >=32 and batch-size positive")
    if not .025 < args.min_depth < args.max_depth < .22:
        raise ValueError("Depth range must lie within (0.025, 0.22) metres")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.output}; choose a new --output")
    seed_everything(args.seed)
    size = (args.height,args.width)
    k, base = load_calibration(args.calibration, size)
    mesh = InstrumentMesh(args.mesh_dir,args.faces_per_part,load_appearance=True).to(args.device)
    renderer = SemanticRenderer(mesh,size,args.renderer,supersample=args.supersample,render_rgb=True)
    k, base = k.to(args.device), base.to(args.device)
    metadata = {"schema_version": 1, "image_size": list(size), "K": k.tolist(), "base_pose": base.tolist(),
                "mesh_dir": str(args.mesh_dir.resolve()), "mesh_sha256": mesh.hashes,
                "faces_per_part": args.faces_per_part, "renderer": renderer.backend,
                "renderer_config":renderer.configuration(),
                "rgb_source":"rendered_mtl_v1","rgb_premultiplied":True,
                "effective_triangle_count": len(mesh.faces), "geometry_convention": "instrument_splatting_wrist_opencv_v1",
                "mask_channels": ["shaft","wrist","grippers"], "units": "metres/radians",
                "seed": args.seed, "rotation_range_degrees": args.rotation_range,
                "depth_range_m": [args.min_depth,args.max_depth],
                "calibration_source": str(args.calibration.resolve()),
                "real_data_usage": "Only first-frame initialization and camera calibration; no real pixels used"}
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (args.output/"samples").mkdir()
    (args.output/"preview").mkdir()
    val_count = min(args.num_samples-1,max(1,round(args.num_samples*args.val_fraction)))
    val_ids = set(np.random.default_rng(args.seed).choice(args.num_samples,val_count,replace=False).tolist())
    written, attempted = 0, 0
    start = time.perf_counter()
    with (args.output/"manifest.jsonl").open("w",encoding="utf-8") as manifest:
        while written < args.num_samples:
            if attempted >= args.num_samples*args.max_attempt_factor:
                raise RuntimeError(f"Only {written} visible samples generated after {attempted} attempts; check camera/pose range. Partial data retained.")
            b = min(args.batch_size,args.num_samples-written)
            p = vector_pose(base[None].repeat(b,1))
            a = (torch.rand(b,3,device=args.device)*2-1)*np.deg2rad(args.rotation_range)
            p["R"] = axis_rotation(a[:,0],"x") @ axis_rotation(a[:,1],"y") @ axis_rotation(a[:,2],"z") @ p["R"]
            z = torch.rand(b,device=args.device)*(args.max_depth-args.min_depth)+args.min_depth
            u = (torch.rand(b,device=args.device)*.7+.15)*(args.width-1)
            v = (torch.rand(b,device=args.device)*.7+.15)*(args.height-1)
            p["t"] = torch.stack(((u-k[0,2])*z/k[0,0],(v-k[1,2])*z/k[1,1],z),-1)
            alpha = (torch.rand(b,device=args.device)*2-1)*np.deg2rad(75)
            half = torch.rand(b,device=args.device)*np.deg2rad(50)
            yaw = (torch.rand(b,device=args.device)*2-1)*torch.minimum(half.new_full((b,),np.deg2rad(35)),np.deg2rad(80)-half)
            p["joints"] = torch.stack((alpha,half+yaw,half-yaw),-1)
            render = renderer(p,k)
            vertices, camera_tips = mesh(p)
            visibility = visible_tips(vertices,mesh.faces,camera_tips)
            rgb = render["rgb"]
            vectors = pose_vector(p)
            initial = pose_vector(perturb_pose(vectors))
            # Include difficult first-frame initialization as well as tracking pairs.
            fixed = torch.rand(b,device=args.device) < .5
            initial[fixed] = base
            for i in range(b):
                attempted += 1
                mask, tips = render["mask"][i], render["tips"][i]
                area = (mask > .5).sum((-2,-1))
                if area[1] < max(3,args.height*args.width*.0001) or area[2] < max(4,args.height*args.width*.0002):
                    continue
                if (p["t"][i,:2].abs() >= .075).any():
                    continue
                tip_valid = ((tips[:,0] >= 2)&(tips[:,0] < args.width-2)&
                             (tips[:,1] >= 2)&(tips[:,1] < args.height-2)&
                             visibility[i])
                rel = f"samples/{written:07d}.npz"
                np.savez_compressed(args.output/rel,
                    mask=(mask.cpu().numpy()*255).round().astype(np.uint8),
                    rgb=(rgb[i].cpu().numpy()*255).round().astype(np.uint8),
                    pose_vector=vectors[i].cpu().numpy(), initial_vector=initial[i].cpu().numpy(),
                    K=k.cpu().numpy(), tips=tips.cpu().numpy(), tip_confidence=tip_valid.float().cpu().numpy())
                manifest.write(json.dumps({"file":rel,"split":"val" if written in val_ids else "train"})+"\n")
                if written < 8:
                    save_mask(mask,args.output/"preview"/f"{written:04d}_mask.png")
                    from PIL import Image
                    Image.fromarray((rgb[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)).save(args.output/"preview"/f"{written:04d}_rgb.png")
                written += 1
            manifest.flush()
            print(f"Generated {written}/{args.num_samples}, attempts={attempted}, elapsed={time.perf_counter()-start:.1f}s",flush=True)
    metadata["num_samples"] = written
    metadata["num_val"] = val_count
    metadata["complete"] = True
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


@torch.no_grad()
def generate_video(args):
    import cv2
    from utils.pose_video import continuous_pose, open_video_writer
    if (not np.isfinite([args.duration,args.fps,args.motion_rotation]).all() or
            args.duration <= 0 or args.fps <= 0 or not 0 <= args.motion_rotation <= 30):
        raise ValueError("Positive duration/FPS required; motion-rotation must be in [0,30] degrees")
    if min(args.height,args.width) < 32 or args.height % 2 or args.width % 2 or args.batch_size < 1:
        raise ValueError("Video H/W must be even and >=32; batch-size must be positive")
    if not .025 < args.min_depth < args.max_depth < .22:
        raise ValueError("Depth range must lie within (0.025,0.22) metres")
    frames = int(round(args.duration*args.fps))
    if frames < 2:
        raise ValueError("Video must contain at least 2 frames")
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty --output directory: {args.output}")
    seed_everything(args.seed)
    size = (args.height,args.width)
    k,base = load_calibration(args.calibration,size)
    mesh = InstrumentMesh(args.mesh_dir,args.faces_per_part,load_appearance=True).to(args.device)
    renderer = SemanticRenderer(mesh,size,args.renderer,supersample=args.supersample,render_rgb=True)
    k,base = k.to(args.device),base.to(args.device)
    # Fixed approximate initial pose, not per-frame ground truth fed to inference.
    initial = base.clone()
    depth = (args.min_depth+args.max_depth)/2
    initial[6:9] = initial.new_tensor([(.42*args.width-float(k[0,2]))*depth/float(k[0,0]),
                                     (.50*args.height-float(k[1,2]))*depth/float(k[1,1]),depth])
    metadata = {"schema_version":1,"kind":"synthetic_pose_video","complete":False,
                "video":"video.mp4","mask_dir":"masks","mask_pattern":"frame_{index:06d}.png",
                "frame_index_start":0,"fps":args.fps,"frames":frames,"duration_seconds":frames/args.fps,
                "image_size":list(size),"K":k.tolist(),"initial_pose":initial.tolist(),
                "mesh_dir":str(args.mesh_dir.resolve()),"mesh_sha256":mesh.hashes,
                "renderer_config":renderer.configuration(),"seed":args.seed,
                "motion_rotation_degrees":args.motion_rotation,"depth_range_m":[args.min_depth,args.max_depth],
                "trajectory":"analytic_sinusoids_v1","gt_poses":"poses_gt.jsonl",
                "mask_codes":[0,10,20,30],"units":"metres/radians"}
    metadata_path = args.output/"video_metadata.json"
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    (args.output/"masks").mkdir()
    writer = open_video_writer(args.output/"video.mp4",args.fps,size)
    start = time.perf_counter()
    try:
        with (args.output/"poses_gt.jsonl").open("w",encoding="utf-8") as labels:
            for offset in range(0,frames,args.batch_size):
                indices = torch.arange(offset,min(frames,offset+args.batch_size),device=args.device)
                pose = continuous_pose(indices/args.fps,base,k,size,args.min_depth,args.max_depth,args.motion_rotation,args.seed)
                result = renderer(pose,k)
                rgb = (result["rgb"].permute(0,2,3,1).cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
                masks = result["mask"].cpu()
                vectors = pose_vector(pose).cpu().tolist()
                tips = result["tips"].cpu().tolist()
                for j,frame in enumerate(rgb):
                    index = offset+j
                    if not (masks[j,1:] > .5).flatten(1).any(1).all():
                        raise ValueError(f"Wrist or grippers left view at frame {index}; reduce motion or increase depth. No frames were skipped.")
                    writer.write(cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    save_mask(masks[j],args.output/"masks"/f"frame_{index:06d}.png")
                    labels.write(json.dumps({"frame":index,"time_seconds":index/args.fps,
                                             "pose_vector":vectors[j],"tips_px":tips[j]},allow_nan=False)+"\n")
                if offset == 0 or (offset+len(rgb)) % 60 < args.batch_size or offset+len(rgb) == frames:
                    print(f"Video frames: {offset+len(rgb)}/{frames}, elapsed={time.perf_counter()-start:.1f}s",flush=True)
    finally:
        writer.release()
    metadata["complete"] = True
    metadata["generation_seconds"] = time.perf_counter()-start
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"Saved {args.output/'video.mp4'}: {frames} frames, {args.fps:g} FPS, {frames/args.fps:g} seconds",flush=True)


if __name__ == "__main__":
    generate(build_parser().parse_args())
