"""Pose inference for real videos or synthetic validation, with GT/prediction panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import warnings

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from utils.pose_geometry import ROOT, InstrumentMesh, SemanticRenderer, detach_pose, load_calibration
from utils.pose_data import RealFrameDataset, SyntheticDataset, save_mask, to_device
from network.pose_losses import part_iou, part_dice, tip_metrics
from utils.pose_tip_labels import prepare_tip_cache
from utils.pose_visualization import validation_images
from network.poseNet import PoseNet, predict_stages


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, help="Real frame root or synthetic dataset root; --split val defaults to checkpoint training data")
    p.add_argument("--split", choices=("auto","real","val"), default="auto",
                   help="auto detects synthetic manifest and selects only val records; otherwise real frames")
    p.add_argument("--mask-dir", type=Path)
    p.add_argument("--color-dir", type=Path)
    p.add_argument("--calibration", type=Path, help="Defaults to data/transforms.json; resized to checkpoint resolution")
    p.add_argument("--mesh-dir", type=Path)
    p.add_argument("--output", type=Path, default=ROOT/"runs/inference")
    p.add_argument("--renderer", choices=("auto","torch","nvdiffrast"), default="nvdiffrast")
    p.add_argument("--supersample",type=int,choices=(1,2,4),default=None,
                   help="Override checkpoint supersampling for speed/quality comparison")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--track", action="store_true", help="Use previous frame pose as next initialization")
    p.add_argument("--reset-every", type=int, default=0, help="Periodic temporal prior reset, 0 disables")
    p.add_argument("--mask-only", action="store_true")
    p.add_argument("--no-render-output", action="store_true", help="Skip final rendering, overlays and IoU for throughput")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--video",type=Path,help="Predict an MP4/video with one semantic mask per frame")
    p.add_argument("--video-metadata",type=Path,help="Default: <video stem>_metadata.json next to the video")
    p.add_argument("--independent-frames",action="store_true",help="Video only: disable the default previous-prediction initialization")
    p.add_argument("--no-part-axes",action="store_true",help="Video only: hide the default per-part XYZ axes")
    p.add_argument("--axis-length-mm",type=float,default=4.,help="Length of displayed part axes in millimetres (video, default 4)")
    p.add_argument("--fps-window",type=int,default=30,help="Completed frames used for the end-to-end FPS overlay (video, default 30)")
    return p


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class NamedValidationDataset(SyntheticDataset):
    def __getitem__(self,index):
        row = super().__getitem__(index)
        row["name"] = Path(self.records[index]["file"]).stem
        return row


def inference_dataset(args,meta,checkpoint_config):
    root = args.data
    if root is None:
        if args.split == "val":
            if not checkpoint_config.get("data"):
                raise ValueError("Checkpoint has no dataset path; provide --data for validation")
            root = Path(checkpoint_config["data"])
            if not root.is_absolute():
                root = ROOT/root
        else:
            root = ROOT/"data/surgpose_sample"
    split = args.split
    if split == "auto":
        split = "val" if (root/"manifest.jsonl").exists() else "real"
    if split == "val":
        if args.track or args.reset_every:
            raise ValueError("Validation samples are independent: remove --track and --reset-every")
        if args.calibration or args.mask_dir or args.color_dir:
            raise ValueError("Synthetic validation uses the saved RGB, masks and per-sample K; remove real-data overrides")
        dataset = NamedValidationDataset(root,"val",augment=False)
        if not dataset.metadata.get("complete"):
            raise ValueError("Validation dataset is incomplete")
        for key in ("image_size","mesh_sha256","geometry_convention","renderer_config"):
            if dataset.metadata.get(key) != meta.get(key):
                raise ValueError(f"Validation {key} differs from checkpoint; use its matching dataset")
        if args.limit:
            dataset.records = dataset.records[:args.limit]
        print(f"Selected validation samples: {len(dataset)}; checking labels only for these samples.",flush=True)
        labels,_ = prepare_tip_cache(root,records=dataset.records)
        dataset.tip_labels = labels
    else:
        k,base = load_calibration(args.calibration or root/"transforms.json",tuple(meta["image_size"]))
        dataset = RealFrameDataset(root,tuple(meta["image_size"]),k,base,args.mask_dir,args.color_dir,args.limit,
                                   include_tip_labels=False)
    return dataset,split,root


def finite_part_means(values):
    stack = np.asarray(values)
    counts = np.isfinite(stack).sum(0)
    sums = np.nansum(stack,axis=0)
    return [float(s/c) if c else None for s,c in zip(sums,counts)]


@torch.inference_mode()
def infer(args):
    if args.video:
        return infer_video(args)
    if args.video_metadata or args.independent_frames:
        raise ValueError("--video-metadata/--independent-frames require --video")
    checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    input_mode = checkpoint.get("input_mode","legacy_mask_pair")
    rgb_pair = input_mode == "rgb_pair"
    if rgb_pair and args.mask_only:
        raise ValueError("This checkpoint uses paired RGB inputs; remove --mask-only")
    meta = checkpoint["metadata"]
    size = tuple(meta["image_size"])
    steps = args.steps if args.steps is not None else checkpoint["config"]["steps"]
    if steps < 1 or min(args.warmup,args.limit,args.reset_every) < 0:
        raise ValueError("steps must be positive; warmup/limit/reset-every must be nonnegative")
    if (args.output/"poses.jsonl").exists():
        raise FileExistsError("Choose a new --output to preserve existing predictions")
    dataset,split,data_root = inference_dataset(args,meta,checkpoint["config"])
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"poses.jsonl").exists():
        raise FileExistsError("Choose a new --output to preserve existing predictions")
    mesh = InstrumentMesh(args.mesh_dir or meta["mesh_dir"],meta["faces_per_part"],load_appearance=rgb_pair).to(args.device)
    if mesh.hashes != meta["mesh_sha256"]:
        raise ValueError("Inference mesh differs from the checkpoint")
    render_config = meta.get("renderer_config",{})
    supersample = args.supersample if args.supersample is not None else render_config.get("supersample",2)
    renderer = SemanticRenderer(mesh,size,args.renderer,supersample=supersample,
                                edge_width=render_config.get("edge_width",1.0),render_rgb=rgb_pair)
    if renderer.configuration() != render_config:
        warnings.warn("Rendering differs from checkpoint training (legacy version, backend, or sampling). Regenerate data and retrain for matched results.",stacklevel=2)
    model = PoseNet(meta["base_pose"],input_mode=input_mode).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0)
    folders = ("masks","overlays","rgb_pairs") if split == "val" and rgb_pair else ("masks","overlays")
    for folder in folders:
        if not args.no_render_output:
            (args.output/folder).mkdir(exist_ok=True)
    if checkpoint.get("smoke_test"):
        print("Checkpoint is from a smoke test; predictions do not establish accuracy.",flush=True)
    latencies, ious, dices, previous = [], [], [], None
    pair_count, gap_sum = 0, 0.
    print(f"Dataset: {data_root}, split={split}, samples={len(dataset)}, HxW={size[0]}x{size[1]}",flush=True)
    warmup_batch = to_device(next(iter(loader)),args.device)
    amp_dtype = torch.bfloat16 if torch.device(args.device).type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    def predict(batch,initial):
        with torch.autocast(device_type=torch.device(args.device).type,dtype=amp_dtype,enabled=args.amp and torch.device(args.device).type == "cuda"):
            return predict_stages(model,renderer,None if args.mask_only else batch["rgb"],batch["mask"],
                                  initial,batch["K"],steps,render_final=not args.no_render_output)[-1]
    for _ in range(args.warmup):
        predict(warmup_batch,warmup_batch["initial_pose"])
    synchronize(args.device)
    overall_start = time.perf_counter()
    with (args.output/"poses.jsonl").open("w",encoding="utf-8") as output:
        for index,batch in enumerate(loader):
            batch = to_device(batch,args.device)
            initial = previous if args.track and previous is not None else batch["initial_pose"]
            if args.reset_every and index % args.reset_every == 0:
                initial = batch["initial_pose"]
            if batch["mask"].sum() == 0:
                previous = None
                output.write(json.dumps({"frame":batch["name"][0],"status":"empty_mask","pose":None})+"\n")
                continue
            synchronize(args.device)
            start = time.perf_counter()
            result = predict(batch,initial)
            synchronize(args.device)
            latency = (time.perf_counter()-start)*1000
            latencies.append(latency)
            pose = result["pose"]
            previous = detach_pose(pose)
            transform = torch.eye(4,device=args.device)
            transform[:3,:3], transform[:3,3] = pose["R"][0], pose["t"][0]
            row = {"frame":batch["name"][0],"status":"predicted","T_wrist_camera":transform.tolist(),
                   "translation_m":pose["t"][0].tolist(),"joints_rad":pose["joints"][0].tolist(),
                   "joint_order":["alpha","theta_left","theta_right"],"pipeline_ms":latency}
            if not args.no_render_output:
                metric = part_iou(result["mask"],batch["mask"])[0].cpu().numpy()
                ious.append(metric)
                row["part_iou"] = [None if not np.isfinite(x) else float(x) for x in metric]
                dice = part_dice(result["mask"],batch["mask"])[0].cpu().numpy()
                dices.append(dice)
                row["part_dice"] = [None if not np.isfinite(x) else float(x) for x in dice]
                name = batch["name"][0]
                save_mask(result["mask"][0],args.output/"masks"/f"{name}.png")
                if split == "val":
                    tip = tip_metrics(result["tips"],batch)
                    pair_count += tip["tip_pair_count"]
                    gap_sum += tip["tip_gap_error_sum_px"]
                    row.update(tips_pred_px=result["tips"][0].tolist(),tips_gt_px=batch["tips"][0].tolist(),
                               tip_confidence=batch["tip_confidence"][0].tolist(),
                               tip_pair_valid=bool(tip["tip_pair_count"]),
                               tip_gap_error_px=tip["tip_gap_error_sum_px"] if tip["tip_pair_count"] else None)
                    # Legacy mask checkpoints have no rendered RGB; emit only overlap.
                    panels = validation_images(batch,result if rgb_pair else {**result,"rgb":torch.zeros_like(batch["rgb"])})
                    for key,folder in (("overlap_GT_left_prediction_right","overlays"),
                                       ("RGB_target_left_render_right","rgb_pairs")):
                        if folder == "rgb_pairs" and not rgb_pair:
                            continue
                        pixels = (panels[key].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                        Image.fromarray(pixels).save(args.output/folder/f"{name}.png")
                else:
                    target = batch["rgb"][0].permute(1,2,0).cpu().numpy()
                    colors = np.array([[.1,.4,1.],[.1,1.,.25],[1.,.3,.1]],dtype=np.float32)
                    pred = np.einsum("chw,cd->hwd",result["mask"][0].cpu().numpy(),colors)
                    panel = np.concatenate((target,.6*target+.4*pred),1)
                    Image.fromarray((panel.clip(0,1)*255).astype(np.uint8)).save(args.output/"overlays"/f"{name}.png")
            output.write(json.dumps(row,allow_nan=False)+"\n")
            output.flush()
            print(f"{index+1}/{len(dataset)}, sample={batch['name'][0]}: {latency:.2f} ms",flush=True)
    elapsed = time.perf_counter()-overall_start
    stats = {"frames":len(dataset),"predicted_frames":len(latencies),"renderer":renderer.backend,
             "dataset":str(data_root.resolve()),"split":split,"part_names":["shaft","wrist","grippers"],
             "input_mode":input_mode,
             "renderer_config":renderer.configuration(),
             "training_renderer":meta["renderer"],"image_size":list(size),"steps":steps,
             "pipeline_mean_ms":float(np.mean(latencies)) if latencies else None,
             "pipeline_p95_ms":float(np.percentile(latencies,95)) if latencies else None,
             "pipeline_fps":1000/float(np.mean(latencies)) if latencies else None,
             "end_to_end_fps":len(dataset)/elapsed,"warmup_frames":args.warmup,
             "timing_scope":"pipeline includes initial render, network passes and intermediate/final renders; excludes mask extraction, disk I/O, preprocessing and output serialization",
             "smoke_test_checkpoint":checkpoint.get("smoke_test",False),
             "pose_accuracy":("Not measured: reports 2D masks and projected endpoint separation only" if split == "val" else
                              "Not measured: real transforms are initialization, not pose ground truth")}
    if ious:
        stats["part_iou"] = finite_part_means(ious)
        stats["part_dice"] = finite_part_means(dices)
    if split == "val" and not args.no_render_output:
        stats.update(tip_pair_count=pair_count,tip_pair_ratio=pair_count/len(latencies) if latencies else None,
                     tip_gap_error_px=gap_sum/pair_count if pair_count else None)
    (args.output/"metrics.json").write_text(json.dumps(stats,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(stats,indent=2),flush=True)
    if not args.no_render_output:
        print(f"Saved {len(ious)} samples; {len(ious)} PNGs in each of: {', '.join(folders)}. "
              "Filenames retain source sample IDs, not sequential output counts.",flush=True)


@torch.inference_mode()
def infer_video(args):
    import cv2
    from utils.pose_geometry import vector_pose
    from utils.pose_video import (VideoSource, open_video_writer, resize_intrinsics, video_comparison,
                            part_axis_anchors,part_axis_points,RollingFPS)
    if args.split == "val" or args.data or args.color_dir or args.no_render_output or args.mask_only:
        raise ValueError("Video mode uses --video and semantic masks; remove --split val/--data/--color-dir/--no-render-output/--mask-only")
    if args.track and args.independent_frames:
        raise ValueError("Choose --track or --independent-frames, not both")
    if min(args.limit,args.warmup,args.reset_every) < 0:
        raise ValueError("limit/warmup/reset-every must be nonnegative")
    if not np.isfinite(args.axis_length_mm) or args.axis_length_mm <= 0:
        raise ValueError("--axis-length-mm must be finite and positive")
    if args.fps_window < 1:
        raise ValueError("--fps-window must be positive")
    if args.video_metadata and not args.video_metadata.exists():
        raise FileNotFoundError(args.video_metadata)
    checkpoint = torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if checkpoint.get("input_mode") != "rgb_pair":
        raise ValueError("Video mode requires a 6-channel RGB-pair checkpoint")
    meta = checkpoint["metadata"]
    size = tuple(meta["image_size"])
    steps = args.steps if args.steps is not None else checkpoint["config"]["steps"]
    if steps < 1:
        raise ValueError("steps must be positive")
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Choose an empty video output directory: {args.output}")
    source = VideoSource(args.video,args.mask_dir,args.video_metadata)
    writer = None
    latencies, end_to_end, dices = [], [], []
    processed, empty_count = 0,0
    try:
        if args.calibration:
            k,base = load_calibration(args.calibration,size)
        elif source.metadata:
            k = resize_intrinsics(source.metadata["K"],source.size,size)
            base = torch.tensor(source.metadata["initial_pose"],dtype=torch.float32)
        else:
            raise ValueError("Provide --calibration for videos without camera/initial-pose metadata")
        if source.metadata and source.metadata.get("mesh_sha256") != meta["mesh_sha256"]:
            raise ValueError("Video mesh differs from checkpoint")
        mesh = InstrumentMesh(args.mesh_dir or meta["mesh_dir"],meta["faces_per_part"],load_appearance=True).to(args.device)
        if mesh.hashes != meta["mesh_sha256"]:
            raise ValueError("Inference mesh differs from checkpoint")
        config = meta["renderer_config"]
        renderer = SemanticRenderer(mesh,size,args.renderer,supersample=args.supersample or config["supersample"],
                                    edge_width=config["edge_width"],render_rgb=True)
        if renderer.configuration() != config:
            warnings.warn("Video inference rendering differs from checkpoint training",stacklevel=2)
        if source.metadata and source.metadata.get("renderer_config") != renderer.configuration():
            warnings.warn("Synthetic video appearance/render settings differ from moving renderer",stacklevel=2)
        model = PoseNet(meta["base_pose"],input_mode="rgb_pair").to(args.device).eval()
        model.load_state_dict(checkpoint["model"])
        anchors = part_axis_anchors(mesh) if not args.no_part_axes else None
        axes_k = resize_intrinsics(k,size,source.size).numpy()
        k = k.to(args.device)
        initial = vector_pose(base[None].to(args.device))
        previous = None
        tracking = not args.independent_frames
        amp_dtype = torch.bfloat16 if torch.device(args.device).type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
        def predict(batch,pose):
            with torch.autocast(device_type=torch.device(args.device).type,dtype=amp_dtype,
                                enabled=args.amp and torch.device(args.device).type == "cuda"):
                return predict_stages(model,renderer,batch["rgb"],batch["mask"],pose,k,steps)[-1]
        warm = source.read(0,size)
        if warm is None:
            raise ValueError("Video has no decodable frames")
        warm_batch = to_device(warm[1],args.device)
        warmups = args.warmup if warm_batch["mask"].any() else 0
        for _ in range(warmups):
            predict(warm_batch,initial)
        synchronize(args.device)
        source.restart()
        count = min(len(source.masks),args.limit) if args.limit else len(source.masks)
        writer = open_video_writer(args.output/"prediction.mp4",source.fps,(source.size[0],2*source.size[1]))
        print(f"Video: {count} frames, source {source.fps:g} FPS, {count/source.fps:g}s; "
              f"network HxW={size}; tracking={tracking}",flush=True)
        start = time.perf_counter()
        fps_meter = RollingFPS(args.fps_window)
        fps_meter.start(start)
        with (args.output/"poses.jsonl").open("w",encoding="utf-8") as output:
            for index in range(count):
                frame_start = time.perf_counter()
                item = source.read(index,size)
                if item is None:
                    raise ValueError(f"Video decoding ended at frame {index}, expected {count}; output is incomplete")
                frame,batch = item
                batch = to_device(batch,args.device)
                row = {"frame":index+1,"source_index":index,"time_seconds":index/source.fps}
                mask = None
                axes = None
                if not batch["mask"].any():
                    previous = None
                    empty_count += 1
                    row.update(status="empty_mask",pose=None,pipeline_ms=None)
                else:
                    pose_in = previous if tracking and previous is not None else initial
                    if args.reset_every and index % args.reset_every == 0:
                        pose_in = initial
                    synchronize(args.device)
                    tick = time.perf_counter()
                    result = predict(batch,pose_in)
                    synchronize(args.device)
                    latency = (time.perf_counter()-tick)*1000
                    latencies.append(latency)
                    pose = result["pose"]
                    previous = detach_pose(pose)
                    transform = torch.eye(4,device=args.device)
                    transform[:3,:3],transform[:3,3] = pose["R"][0],pose["t"][0]
                    dice = part_dice(result["mask"],batch["mask"])[0].cpu().numpy()
                    dices.append(dice)
                    mask = result["mask"][0].cpu().numpy()
                    row.update(status="predicted",T_wrist_camera=transform.tolist(),
                               translation_m=pose["t"][0].tolist(),joints_rad=pose["joints"][0].tolist(),
                               joint_order=["alpha","theta_left","theta_right"],pipeline_ms=latency,
                               part_dice=[float(x) if np.isfinite(x) else None for x in dice])
                    if anchors is not None:
                        # Small FK visualization runs on CPU; no second mesh render.
                        cpu_pose = {key:value.detach().cpu() for key,value in pose.items()}
                        axes = part_axis_points(cpu_pose,anchors,args.axis_length_mm/1000)[0].numpy()
                # Current encoding/I/O is not finished until AFTER drawing this frame.
                # Display only measured completed frames; never use playback FPS.
                row["display_fps"] = fps_meter.fps
                writer.write(video_comparison(frame,mask,fps=row["display_fps"],axes_camera=axes,axes_k=axes_k))
                output.write(json.dumps(row,allow_nan=False)+"\n")
                output.flush()
                processed += 1
                completed = time.perf_counter()
                end_to_end.append((completed-frame_start)*1000)
                fps_meter.complete(completed)
                if processed % 30 == 0 or processed == count:
                    print(f"Video inference {processed}/{count}, processing FPS={processed/(time.perf_counter()-start):.2f}",flush=True)
            if not args.limit:
                extra,_ = source.capture.read()
                if extra:
                    raise ValueError("Video has additional frames without masks; output is incomplete")
        writer.release()
        writer = None
        elapsed = time.perf_counter()-start
        stats = {"complete":True,"video":str(args.video.resolve()),"frames":processed,
                 "predicted_frames":len(latencies),"empty_frames":empty_count,
                 "source_fps":source.fps,"output_fps":source.fps,"duration_seconds":processed/source.fps,
                 "image_size":list(size),"output_size":[source.size[0],2*source.size[1]],
                 "tracking":tracking,"steps":steps,"warmup_frames":warmups,
                 "part_axes":not args.no_part_axes,"axis_length_mm":args.axis_length_mm,
                 "axis_parts":["shaft","wrist","gripper_left","gripper_right"],
                 "axis_colors":{"X":"red","Y":"green","Z":"blue"},
                 "axis_origins":"Display anchors inside canonical parts, not the physical joint pivots",
                 "pipeline_mean_ms":float(np.mean(latencies)) if latencies else None,
                 "pipeline_p95_ms":float(np.percentile(latencies,95)) if latencies else None,
                 "pipeline_fps":1000/float(np.mean(latencies)) if latencies else None,
                 "end_to_end_seconds":elapsed,"end_to_end_fps":processed/elapsed,
                 "end_to_end_p95_ms":float(np.percentile(end_to_end,95)),
                 "frame_budget_ms":1000/source.fps,
                 "fps_overlay":{"metric":"end_to_end","window_completed_frames":args.fps_window,
                                "lag_frames":1,"first_frame":"FPS: --",
                                "scope":"decode, mask loading, preprocessing, transfer, network, renders, Dice, axes/overlap, encoding and JSON I/O; excludes model loading, warmup, final encoder close and segmentation model"},
                 "frames_within_budget_ratio":float((np.array(end_to_end) <= 1000/source.fps).mean()),
                 "timing_scope":"Pipeline: initial/intermediate/final renders + network, CUDA synchronized. End-to-end: video decoding, precomputed-mask loading, preprocessing, transfer, prediction, Dice, overlay/part axes, encoding and JSON I/O; includes encoder close, excludes model loading and warmup. No segmentation model is timed.",
                 "renderer_config":renderer.configuration(),"smoke_test_checkpoint":checkpoint.get("smoke_test",False),
                 "pose_accuracy":"Not measured: Dice compares projected masks; per-frame GT pose file is not used for inference"}
        if dices:
            stats["part_names"] = ["shaft","wrist","grippers"]
            stats["part_dice"] = finite_part_means(dices)
        (args.output/"metrics.json").write_text(json.dumps(stats,indent=2,allow_nan=False),encoding="utf-8")
        print(json.dumps(stats,indent=2),flush=True)
    finally:
        source.close()
        if writer is not None:
            writer.release()


if __name__ == "__main__":
    infer(build_parser().parse_args())
