"""Render current mesh masks and make an unretouched inspection figure."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
import torch

from pose_data import read_mask, save_mask
from pose_geometry import ROOT, InstrumentMesh, SemanticRenderer, load_calibration, vector_pose


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"runs/renderer_nvdiffrast")
    parser.add_argument("--renderer",choices=("nvdiffrast","torch"),default="nvdiffrast")
    parser.add_argument("--faces-per-part",type=int,default=0)
    parser.add_argument("--height",type=int,default=256)
    parser.add_argument("--width",type=int,default=320)
    parser.add_argument("--supersample",type=int,choices=(1,2,4),default=2)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    size = (args.height,args.width)
    if min(size) < 32:
        raise ValueError("height and width must be >= 32")
    crop_y = slice(round(65*size[0]/256),round(165*size[0]/256))
    crop_x = slice(round(115*size[1]/320),round(250*size[1]/320))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    k, base = load_calibration(image_size=size)
    mesh = InstrumentMesh(faces_per_part=args.faces_per_part).to(device)
    renderer = SemanticRenderer(mesh, size, args.renderer,supersample=args.supersample)
    pose = vector_pose(base[None].to(device))
    initial = renderer(pose, k.to(device))["mask"][0].cpu().numpy()
    pose["joints"] = torch.tensor([[0., np.deg2rad(30), np.deg2rad(30)]], device=device, dtype=torch.float32)
    opened = renderer(pose, k.to(device))["mask"][0].cpu().numpy()
    target = read_mask(ROOT/"data/surgpose_sample/l_mask/frame000.png",size).numpy()
    with Image.open(ROOT/"data/surgpose_sample/color/frame000.png") as im:
        rgb = np.asarray(im.convert("RGB").resize(size[::-1],Image.Resampling.BILINEAR)).copy()
    hard = np.eye(3,dtype=np.float32)[initial.argmax(0)] * (initial.max(0) > .5)[...,None]
    save_mask(initial,output/"initial_labels.png")
    save_mask(opened,output/"opened_labels.png")
    # Loss consumes the float arrays, not the colored images or thresholded labels.
    np.savez_compressed(output/"soft_masks.npz",initial=initial,opened=opened,target=target,K=k.numpy(),base_pose=base.numpy())
    for name, array in (("initial_soft",initial.transpose(1,2,0)),("initial_hard",hard),
                        ("opened_soft",opened.transpose(1,2,0)),("target",target.transpose(1,2,0))):
        Image.fromarray((array.clip(0,1)*255).round().astype(np.uint8)).save(output/f"{name}.png")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else FontProperties()
    panels = [
        (rgb,"目标帧 frame000（仅作位置参照）"),
        (target.transpose(1,2,0),"真实标注 mask"),
        (initial.transpose(1,2,0),"初始位姿 → soft mask（损失实际输入）"),
        (hard,"同一投影 → 标签 mask（阈值 0.5）"),
        (initial.transpose(1,2,0)[crop_y,crop_x],"腕部/夹爪局部放大（保留原始像素）"),
        (opened.transpose(1,2,0),"固定腕部位姿，左右夹爪各开 30°"),
    ]
    fig, axes = plt.subplots(2,3,figsize=(15,8.8),facecolor="#171b23")
    for ax,(array,title) in zip(axes.flat,panels):
        ax.imshow(array,interpolation="nearest")
        ax.set_title(title,fontproperties=font,color="white",fontsize=12,pad=10)
        ax.axis("off")
    fig.suptitle(f"当前 {renderer.backend} 渲染器实测  |  红：杆身   绿：腕部   蓝：左右夹爪",fontproperties=font,color="white",fontsize=17,y=.98)
    fig.text(.5,.025,f"输出宽×高 {size[1]}×{size[0]}  ·  {len(mesh.faces)} 个三角面  ·  {args.supersample}× 超采样  ·  可见轮廓抗锯齿，无高斯模糊/补洞",ha="center",fontproperties=font,color="#c5cbd4",fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.07,wspace=.06,hspace=.15)
    fig.savefig(output/"comparison.png",dpi=130,facecolor=fig.get_facecolor())
    plt.close(fig)
    old_path = ROOT/"runs/renderer_inspection/soft_masks.npz"
    if old_path.exists():
        with np.load(old_path) as old:
            old_mask = old["initial"]
            same_pose = np.allclose(base.numpy(),old["base_pose"]) and np.allclose(k.numpy(),old["K"])
        if same_pose and old_mask.shape == initial.shape:
            fig,axes = plt.subplots(2,2,figsize=(11,8.8),facecolor="#171b23")
            before,after = old_mask.transpose(1,2,0),initial.transpose(1,2,0)
            for ax,array,title in zip(axes.flat,
                    (before,after,before[crop_y,crop_x],after[crop_y,crop_x]),
                    ("修改前：内部三角边产生暗纹",f"修改后：可见轮廓抗锯齿 + {args.supersample}× 超采样",
                     "修改前：腕部/夹爪局部", "修改后：腕部/夹爪局部")):
                ax.imshow(array.clip(0,1),interpolation="nearest")
                ax.set_title(title,fontproperties=font,color="white",fontsize=12,pad=10)
                ax.axis("off")
            fig.suptitle("同一位姿、同一相机：旧渲染器/简化网格与当前设置",fontproperties=font,color="white",fontsize=17,y=.98)
            fig.text(.5,.025,"红：杆身   绿：腕部   蓝：夹爪  |  展示训练所用 soft mask，未阈值化",ha="center",fontproperties=font,color="#c5cbd4",fontsize=11)
            fig.subplots_adjust(left=.02,right=.98,top=.91,bottom=.07,wspace=.06,hspace=.14)
            fig.savefig(output/"before_after.png",dpi=130,facecolor=fig.get_facecolor())
            plt.close(fig)
    metadata = {"renderer":renderer.backend,"height":size[0],"width":size[1],"triangles":len(mesh.faces),
                "faces_per_part":args.faces_per_part,
                "initial_pose_source":"transforms.json frame000, alpha=0, jaws=5 degrees each",
                "alignment":"Initial pose is not fitted to the target frame", "postprocessing":"Differentiable edge antialiasing, then area supersampling; no blur or hole filling",
                "renderer_config":renderer.configuration(),
                "raw_soft_file":"soft_masks.npz"}
    (output/"inspection.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(output/"comparison.png")


if __name__ == "__main__":
    main()
