"""Train paired masked-RGB ResNet-34 with RGB/mask/tip position/separation losses."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pose_geometry import ROOT, InstrumentMesh, SemanticRenderer
from poseNet import PoseNet, predict_stages
from pose_data import SyntheticDataset, seed_everything, to_device
from pose_losses import part_dice, part_iou, total_loss, tip_metrics
from pose_tip_labels import prepare_tip_cache, TIP_LABEL_VERSION
from pose_visualization import validation_images

PART_NAMES = ("shaft", "wrist", "grippers")
LOSS_NAMES = ("rgb", "mask", "tips")
COMPONENT_NAMES = (*LOSS_NAMES, "tips_position", "tips_gap")
LOSS_VERSION = 2


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT/"data/synthetic_rgb_pose")
    p.add_argument("--output", type=Path, default=ROOT/"runs/pose_rgb")
    p.add_argument("--mesh-dir", type=Path)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--renderer", choices=("auto","nvdiffrast"), default="nvdiffrast")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--detach-stages", action="store_true")
    checkpoint = p.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path, help="Exact continuation of a run with the same loss version")
    checkpoint.add_argument("--init-from", type=Path, help="Load model weights only, for a new loss/run; resets optimizer and best score")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-batches", type=int, default=0, help="Smoke test: cap train/val batches")
    p.add_argument("--val-every", type=int, default=2)
    p.add_argument("--save-best-by", choices=("val_loss","mean_dice"), default="val_loss")
    p.add_argument("--tensorboard-dir", type=Path, help="Default: output/tensorboard")
    p.add_argument("--rgb-weight", type=float, default=.1)
    p.add_argument("--mask-weight", type=float, default=1.)
    p.add_argument("--tips-weight", type=float, default=1.)
    p.add_argument("--tips-position-weight", type=float, default=1.)
    p.add_argument("--tips-gap-weight", type=float, default=1.)
    return p


def atomic_checkpoint(payload,path):
    tmp = path.with_suffix(".tmp")
    torch.save(payload,tmp)
    tmp.replace(path)



def finite_means(sums,counts):
    return [float(s/c) if c > 0 else None for s,c in zip(sums.cpu(),counts.cpu())]


def run_epoch(model,renderer,loader,args,optimizer=None,scaler=None,writer=None,epoch=0,global_step=0):
    training = optimizer is not None
    model.train(training)
    if training and args.batch_size < 4:
        for module in model.modules():
            if isinstance(module,torch.nn.BatchNorm2d):
                module.eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    weights = {name:getattr(args,name+"_weight") for name in COMPONENT_NAMES}
    count,totals,updates,skipped_updates = 0,{},0,0
    tip_pair_count, gap_error_sum = 0, 0.
    metric_sums = {name:torch.zeros(3,device=args.device) for name in ("dice","iou")}
    metric_counts = {name:torch.zeros(3,device=args.device) for name in metric_sums}
    batch_count = min(len(loader),args.max_batches) if args.max_batches else len(loader)
    for index,batch in enumerate(loader):
        if index >= batch_count:
            break
        batch = to_device(batch,args.device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type="cuda",dtype=amp_dtype,enabled=args.amp):
                outputs = predict_stages(model,renderer,batch["rgb"],batch["mask"],
                                         batch["initial_pose"],batch["K"],args.steps,args.detach_stages)
            losses = [total_loss(result,batch,weights) for result in outputs]
            normalizer = sum(range(1,len(losses)+1))
            # Logged components use the SAME stage weights as the total loss.
            terms = {name:sum((i+1)*item[1][name] for i,item in enumerate(losses))/normalizer
                     for name in COMPONENT_NAMES}
            loss = sum(weights[name]*terms[name] for name in LOSS_NAMES)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in batch {index+1}")
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=not scaler.is_enabled())
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                updates += int(not skipped)
                skipped_updates += int(skipped)
        metrics = {"loss":float(loss.detach()),**{name:float(value.detach()) for name,value in terms.items()}}
        b = len(batch["mask"])
        tip_stats = tip_metrics(outputs[-1]["tips"],batch)
        tip_pair_count += tip_stats["tip_pair_count"]
        gap_error_sum += tip_stats["tip_gap_error_sum_px"]
        pair_ratio = tip_stats["tip_pair_count"]/b
        gap_error = tip_stats["tip_gap_error_sum_px"]/tip_stats["tip_pair_count"] if tip_stats["tip_pair_count"] else None
        for name,value in metrics.items():
            totals[name] = totals.get(name,0.)+value*b
        count += b
        for name,fn in (("dice",part_dice),("iou",part_iou)):
            value = fn(outputs[-1]["mask"].detach(),batch["mask"])
            metric_sums[name] += torch.nan_to_num(value).sum(0)
            metric_counts[name] += torch.isfinite(value).sum(0)
        prefix = "train" if training else "val"
        gap_text = f"{gap_error:.3f}" if gap_error is not None else "N/A"
        print(f"{index+1}/{batch_count}, {prefix}_loss: {metrics['loss']:.6f}, "
              f"rgb_loss: {metrics['rgb']:.6f}, mask_loss: {metrics['mask']:.6f}, "
              f"tips_loss: {metrics['tips']:.6f}, tips_position: {metrics['tips_position']:.6f}, "
              f"tips_gap: {metrics['tips_gap']:.6f}, tip_pair_ratio: {pair_ratio:.3f}, "
              f"tip_gap_error_px: {gap_text}",flush=True)
        if training:
            global_step += 1
            if writer:
                for name,value in metrics.items():
                    writer.add_scalar("train_batch/"+name,value,global_step)
                writer.add_scalar("train_batch/tip_pair_ratio",pair_ratio,global_step)
                if gap_error is not None:
                    writer.add_scalar("train_batch/tip_gap_error_px",gap_error,global_step)
        elif index == 0:
            images = validation_images(batch,outputs[-1])
            image_dir = args.output/"val_images"
            image_dir.mkdir(exist_ok=True)
            for name,grid in images.items():
                if writer:
                    writer.add_image("val/"+name,grid,epoch+1)
                image = (grid.permute(1,2,0).numpy()*255).round().astype(np.uint8)
                Image.fromarray(image).save(image_dir/f"epoch_{epoch+1:04d}_{name}.png")
    if not count:
        raise ValueError("Empty data loader")
    if training and not updates:
        raise FloatingPointError("All optimizer steps overflowed; rerun without --amp")
    report = {**{name:value/count for name,value in totals.items()},
              "tip_pair_ratio":tip_pair_count/count,"tip_pair_count":tip_pair_count,
              "tip_gap_error_px":gap_error_sum/tip_pair_count if tip_pair_count else None,
              "part_dice":finite_means(metric_sums["dice"],metric_counts["dice"]),
              "part_iou":finite_means(metric_sums["iou"],metric_counts["iou"]),
              "samples":count,"optimizer_updates":updates,"amp_skipped_updates":skipped_updates}
    present = [value for value in report["part_dice"] if value is not None]
    report["mean_dice"] = sum(present)/len(present) if present else None
    if writer:
        prefix = "train_epoch" if training else "val"
        for name in ("loss",*COMPONENT_NAMES,"mean_dice","tip_pair_ratio","tip_gap_error_px"):
            if report[name] is not None:
                writer.add_scalar(prefix+"/"+name,report[name],epoch+1)
        for name,value in zip(PART_NAMES,report["part_dice"]):
            if value is not None:
                writer.add_scalar(prefix+"/dice_"+name,value,epoch+1)
    return report,global_step


def train(args):
    if min(args.epochs,args.batch_size,args.steps,args.val_every) < 1 or args.max_batches < 0:
        raise ValueError("epochs/batch-size/steps/val-every must be positive; max-batches >=0")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Differentiable material RGB training requires CUDA")
    weights = [getattr(args,name+"_weight") for name in COMPONENT_NAMES]
    if not all(np.isfinite(weights)) or min(weights) < 0 or (
        args.rgb_weight+args.mask_weight+args.tips_weight*(args.tips_position_weight+args.tips_gap_weight) <= 0
    ):
        raise ValueError("Loss weights must be nonnegative with at least one positive")
    seed_everything(args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((args.output/name).exists() for name in ("last.pt","best.pt","history.jsonl")):
        raise FileExistsError("Existing run: use --resume or a new --output")
    train_set = SyntheticDataset(args.data,"train",True)
    val_set = SyntheticDataset(args.data,"val",False)
    meta = train_set.metadata
    if not meta.get("complete"):
        raise ValueError("Synthetic generation is incomplete")
    if meta.get("rgb_source") != "rendered_mtl_v1" or not meta.get("rgb_premultiplied"):
        raise ValueError("RGB loss requires renderer-consistent RGB targets. Regenerate with current create_data.py "
                         "into data/synthetic_rgb_pose; old randomized-color data cannot supervise RGB.")
    mesh = InstrumentMesh(args.mesh_dir or meta["mesh_dir"],meta["faces_per_part"],load_appearance=True).to(args.device)
    if mesh.hashes != meta["mesh_sha256"]:
        raise ValueError("Mesh hashes differ from the dataset")
    render_config = meta.get("renderer_config",{})
    if render_config.get("version") != SemanticRenderer.RGB_VERSION:
        raise ValueError("Regenerate data for RGB renderer version 4")
    renderer = SemanticRenderer(mesh,meta["image_size"],args.renderer,supersample=render_config["supersample"],
                                edge_width=render_config["edge_width"],render_rgb=True)
    if renderer.configuration() != render_config:
        raise ValueError("Renderer or MTL settings differ from target generation; regenerate data")
    labels, label_report = prepare_tip_cache(args.data)
    train_set.tip_labels = val_set.tip_labels = labels
    print(f"Network input [B,6,H,W] = [B,6,{meta['image_size'][0]},{meta['image_size'][1]}]",flush=True)
    model = PoseNet(meta["base_pose"],pretrained=args.pretrained and not (args.resume or args.init_from),input_mode="rgb_pair").to(args.device)
    if args.init_from:
        checkpoint = torch.load(args.init_from,map_location="cpu",weights_only=True)
        if checkpoint.get("input_mode") != "rgb_pair":
            raise ValueError("--init-from requires a 6-channel RGB checkpoint")
        for key in ("mesh_sha256","geometry_convention"):
            if checkpoint["metadata"].get(key) != meta.get(key):
                raise ValueError(f"--init-from {key} differs from current data")
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model weights from {args.init_from}; new optimizer, scheduler and best score",flush=True)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda",init_scale=1024.,enabled=args.amp and not torch.cuda.is_bf16_supported())
    start_epoch,global_step = 0,0
    best_val_loss = float("inf")
    best_score = float("inf") if args.save_best_by == "val_loss" else -float("inf")
    generator = torch.Generator().manual_seed(args.seed)
    if args.resume:
        checkpoint = torch.load(args.resume,map_location="cpu",weights_only=True)
        if checkpoint.get("loss_version") != LOSS_VERSION or checkpoint.get("tip_label_version") != TIP_LABEL_VERSION:
            raise ValueError("Loss/label version changed: use --init-from with a new --output, not --resume")
        if checkpoint.get("tip_label_fingerprint") != label_report["fingerprint"]:
            raise ValueError("Resume tip label sources changed; use --init-from with a new --output")
        if checkpoint.get("input_mode") != "rgb_pair":
            raise ValueError("Legacy 9-channel checkpoint cannot resume the new 6-channel RGB model; start a new run")
        if checkpoint["metadata"] != meta:
            raise ValueError("Resume dataset metadata differs")
        for name in ("steps","epochs","val_every","save_best_by",*(key+"_weight" for key in COMPONENT_NAMES)):
            if checkpoint["config"][name] != getattr(args,name):
                raise ValueError(f"Resume must preserve --{name.replace('_','-')}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch,global_step = checkpoint["epoch"]+1,checkpoint["global_step"]
        best_score,best_val_loss = checkpoint["best_score"],checkpoint["best_val_loss"]
        torch.set_rng_state(checkpoint["rng_cpu"])
        generator.set_state(checkpoint["loader_rng"])
        if checkpoint.get("rng_cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
    options = dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    train_loader = DataLoader(train_set,shuffle=True,generator=generator,**options)
    val_loader = DataLoader(val_set,shuffle=False,**options)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.output/"config.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    writer = SummaryWriter(str(args.tensorboard_dir or args.output/"tensorboard"),
                           purge_step=global_step+1 if args.resume else None)
    writer.add_text("configuration",json.dumps(config,indent=2))
    writer.add_text("tip_label_audit",json.dumps(label_report,indent=2))
    writer.add_text("validation_images","Rows: samples in the first validation batch; columns: GT/target left, prediction right. "
                    "Overlap colors: shaft red, wrist green, grippers blue. GT endpoints/line yellow, predicted endpoints/line pink; "
                    "invalid GT landmarks gray crosses, no GT line unless both valid. Incomplete batches use their actual size.")
    try:
        for epoch in range(start_epoch,args.epochs):
            print("-"*10,flush=True)
            print(f"epoch {epoch+1}/{args.epochs}",flush=True)
            start = time.perf_counter()
            train_metrics,global_step = run_epoch(model,renderer,train_loader,args,optimizer,scaler,writer,epoch,global_step)
            val_metrics,improved = None,False
            if (epoch+1) % args.val_every == 0:
                val_metrics,_ = run_epoch(model,renderer,val_loader,args,writer=writer,epoch=epoch,global_step=global_step)
                best_val_loss = min(best_val_loss,val_metrics["loss"])
                score = val_metrics["loss"] if args.save_best_by == "val_loss" else val_metrics["mean_dice"]
                improved = score is not None and (score < best_score if args.save_best_by == "val_loss" else score > best_score)
                if improved:
                    best_score = score
                dice_text = ", ".join(f"{name} Dice: {value:.4f}" if value is not None else f"{name} Dice: N/A"
                                      for name,value in zip(PART_NAMES,val_metrics["part_dice"]))
                print(f"val_loss: {val_metrics['loss']:.6f}, {dice_text}",flush=True)
            scheduler.step()
            writer.add_scalar("train_epoch/lr",optimizer.param_groups[0]["lr"],epoch+1)
            record = {"epoch":epoch+1,"train":train_metrics,"val":val_metrics,"seconds":time.perf_counter()-start}
            with (args.output/"history.jsonl").open("a",encoding="utf-8") as log:
                log.write(json.dumps(record,allow_nan=False)+"\n")
            payload = {"schema_version":3,"loss_version":LOSS_VERSION,"tip_label_version":TIP_LABEL_VERSION,
                       "tip_label_fingerprint":label_report["fingerprint"],
                       "input_mode":"rgb_pair","model":model.state_dict(),"optimizer":optimizer.state_dict(),
                       "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"global_step":global_step,
                       "best_score":best_score,"best_val_loss":best_val_loss,"metadata":meta,"config":config,
                       "rng_cpu":torch.get_rng_state(),"loader_rng":generator.get_state(),"rng_cuda":torch.cuda.get_rng_state_all(),
                       "smoke_test":bool(args.max_batches)}
            atomic_checkpoint(payload,args.output/"last.pt")
            if improved:
                atomic_checkpoint(payload,args.output/"best.pt")
                print(f"Saved best.pt ({args.save_best_by}={best_score:.6f})",flush=True)
            writer.flush()
    finally:
        writer.close()


if __name__ == "__main__":
    train(build_parser().parse_args())
