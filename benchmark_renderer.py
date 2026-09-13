"""Compare real, unretouched semantic masks and synchronized CUDA latencies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch

from pose_geometry import ROOT, InstrumentMesh, SemanticRenderer, load_calibration, vector_pose

CASES = [("torch_simplified_ss2","torch","simplified",2),
         ("nvdiffrast_simplified_ss2","nvdiffrast","simplified",2),
         ("nvdiffrast_original_ss1","nvdiffrast","original",1),
         ("nvdiffrast_original_ss2","nvdiffrast","original",2),
         ("nvdiffrast_original_ss4","nvdiffrast","original",4)]


def measure(operation, warmup, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-start)*1000)
    return {"mean_ms":float(np.mean(samples)),"p95_ms":float(np.percentile(samples,95)),
            "median_ms":float(np.median(samples)),"repeats":repeats,"warmup":warmup}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/renderer_benchmark_v3")
    parser.add_argument("--repeats",type=int,default=100)
    parser.add_argument("--warmup",type=int,default=20)
    parser.add_argument("--case",choices=[case[0] for case in CASES],help="Rerun one case without comparison figure")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats must be positive; warmup must be nonnegative")
    args.output.mkdir(parents=True,exist_ok=True)
    size = (256,320)
    k,base = (x.cuda() for x in load_calibration(image_size=size))
    cases = [case for case in CASES if not args.case or case[0] == args.case]
    meshes = {name:InstrumentMesh(faces_per_part=2000 if name == "simplified" else 0).cuda()
              for name in dict.fromkeys(case[2] for case in cases)}
    records, masks = [], {}
    for name,backend,mesh_name,ss in cases:
        mesh = meshes[mesh_name]
        renderer = SemanticRenderer(mesh,size,backend,supersample=ss)
        with torch.no_grad():
            mask = renderer(vector_pose(base[None]),k)["mask"][0].cpu().numpy()
            opened = base[None].clone()
            opened[:,10:] = np.deg2rad(30)
            open_mask = renderer(vector_pose(opened),k)["mask"][0].cpu().numpy()
            masks[name] = mask
            masks[name+"_opened"] = open_mask
            Image.fromarray((mask.transpose(1,2,0)*255).round().astype(np.uint8)).save(args.output/f"{name}.png")
            repeats = min(args.repeats,10) if backend == "torch" else args.repeats
            warmup = min(args.warmup,3) if backend == "torch" else args.warmup
            forward = measure(lambda:renderer(vector_pose(base[None]),k),warmup,repeats)
        vector = base[None].clone().requires_grad_()
        weights = torch.linspace(0,1,size[1],device="cuda")[None,None,None,:]
        weights = weights*torch.tensor([1.,2.,3.],device="cuda")[None,:,None,None]
        def backward():
            vector.grad = None
            (renderer(vector_pose(vector),k)["mask"]*weights).mean().backward()
        training = measure(backward,warmup,repeats)
        assert torch.isfinite(vector.grad).all() and vector.grad.abs().sum() > 0
        interior = binary_erosion(mask[1] > .5,iterations=3)
        record = {"name":name,"vertices":len(mesh.vertices),"triangles":len(mesh.faces),
                  "configuration":renderer.configuration(),"forward":forward,"forward_backward":training,
                  "pose_vector_gradient":vector.grad[0].tolist(),
                  "wrist_interior_pixels":int(interior.sum()),
                  "wrist_interior_mean":float(mask[1,interior].mean()),
                  "wrist_interior_std":float(mask[1,interior].std())}
        records.append(record)
        print(json.dumps(record),flush=True)
        del renderer
    original = masks.get("nvdiffrast_original_ss2")
    for record in records:
        if original is None:
            continue
        current = masks[record["name"]]
        record["mean_absolute_difference_vs_original_ss2"] = float(np.abs(current-original).mean())
        a,b = current > .5,original > .5
        record["part_iou_vs_original_ss2"] = ((a&b).sum((1,2))/(a|b).sum((1,2)).clip(1)).tolist()
    report = {"device":torch.cuda.get_device_name(),"torch":torch.__version__,"image_size_hw":size,
              "batch_size":1,"scope":"FK, projection, rasterization, interpolation, AA, area pooling and tip projection; per-call CUDA synchronization; excludes mesh loading, context/topology setup, network, disk I/O",
              "backward_scope":"same forward plus a spatially weighted semantic mask mean and its pose backward; excludes network/optimizer",
              "records":records}
    (args.output/"benchmark.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    np.savez_compressed(args.output/"soft_masks.npz",**masks,K=k.cpu().numpy(),base_pose=base.cpu().numpy())
    if args.case:
        return
    old_path = ROOT/"runs/renderer_inspection/soft_masks.npz"
    panels = []
    if old_path.exists():
        with np.load(old_path) as old:
            if np.allclose(old["K"],k.cpu().numpy()) and np.allclose(old["base_pose"],base.cpu().numpy()):
                panels.append((old["initial"].copy(),"最初自写渲染器\n简化网格 · 三角面暗纹"))
    panels.extend([(masks["torch_simplified_ss2"],"上一版自写渲染器\n简化网格 · 2× 采样"),
                   (masks["nvdiffrast_simplified_ss2"],"nvdiffrast\n相同简化网格 · 2× 采样"),
                   (original,"当前默认 nvdiffrast\n原始网格 · 2× 采样")])
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig,axes = plt.subplots(2,len(panels),figsize=(4*len(panels),8),facecolor="#171b23")
    for col,(mask,title) in enumerate(panels):
        rgb = mask.transpose(1,2,0)
        for row,array in enumerate((rgb,rgb[65:165,115:250])):
            axes[row,col].imshow(array.clip(0,1),interpolation="nearest")
            axes[row,col].axis("off")
        axes[0,col].set_title(title,fontproperties=font,color="white",fontsize=13)
    fig.suptitle("相同位姿与相机的实际 soft mask：无修图、无模糊或补洞",fontproperties=font,color="white",fontsize=18)
    fig.text(.5,.035,"红：杆身  ·  绿：腕部  ·  蓝：夹爪  |  上：全图  下：原始像素局部放大  |  输出 320×256",fontproperties=font,color="#c5cbd4",ha="center",fontsize=12)
    fig.subplots_adjust(left=.01,right=.99,top=.88,bottom=.08,wspace=.04,hspace=.1)
    fig.savefig(args.output/"renderer_comparison.png",dpi=130,facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
