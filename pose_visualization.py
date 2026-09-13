"""Shared GT/prediction panels for training validation and inference."""
import numpy as np
from PIL import Image, ImageDraw
import torch


def validation_images(batch,result):
    """B rows, GT on the left and prediction on the right; same target RGB base."""
    rgb = batch["rgb"].detach().float().clamp(0,1)
    palette = rgb.new_tensor([[1.,.15,.15],[.15,1.,.15],[.15,.35,1.]])
    def overlap(mask):
        mask = mask.detach().float()
        alpha = mask.sum(1,keepdim=True).clamp(0,1)
        color = torch.einsum("bchw,cd->bdhw",mask,palette)
        return (rgb*(1-.4*alpha)+.4*color).clamp(0,1)
    def rows(left,right):
        paired = torch.cat((left,right),-1)
        b,c,h,w = paired.shape
        return paired.permute(1,0,2,3).reshape(c,b*h,w).cpu()
    left, right = overlap(batch["mask"]), overlap(result["mask"])
    if "tips" in batch and "tips" in result:
        gt = batch["tips"].detach().cpu().numpy()
        pred = result["tips"].detach().cpu().numpy()
        conf = batch["tip_confidence"].detach().cpu().numpy()
        annotated = []
        for panels, prediction in ((left,False),(right,True)):
            images = []
            for i, panel in enumerate(panels):
                image = Image.fromarray((panel.cpu().numpy().transpose(1,2,0)*255).round().astype(np.uint8))
                draw = ImageDraw.Draw(image)
                valid = (conf[i] > 0) & np.isfinite(gt[i]).all(-1)
                def tips(points, good, color, prefix):
                    finite = np.isfinite(points).all(-1)
                    # PIL clips segments to the image; cap extreme off-screen projections.
                    safe = np.clip(points, -4*max(image.size), 4*max(image.size))
                    if (good & finite).all():
                        draw.line([tuple(p) for p in safe],fill=color,width=2)
                    for j, (x,y) in enumerate(safe):
                        if not finite[j] or not (0 <= x < image.width and 0 <= y < image.height):
                            continue
                        if good[j]:
                            draw.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
                            draw.text((x+4,y+2),f"{prefix}{j+1}",fill=color)
                        else:
                            draw.line((x-3,y-3,x+3,y+3),fill=(160,160,160),width=1)
                            draw.line((x-3,y+3,x+3,y-3),fill=(160,160,160),width=1)
                tips(gt[i],valid,(255,230,0),"G")
                gap = np.linalg.norm(gt[i,0]-gt[i,1]) if valid.all() else None
                caption = f"GT yellow; valid {valid.sum()}/2; gap {gap:.1f}px" if gap is not None else f"GT yellow; valid {valid.sum()}/2; gap N/A"
                if prediction:
                    tips(pred[i],np.ones(2,bool),(255,60,220),"P")
                    error = abs(np.linalg.norm(pred[i,0]-pred[i,1])-gap) if gap is not None else None
                    caption = f"GT yellow / Pred pink; gap error {error:.1f}px" if error is not None else "GT yellow / Pred pink; gap error N/A"
                draw.rectangle((0,0,image.width,15),fill=(0,0,0))
                draw.text((3,2),caption,fill=(255,255,255))
                images.append(torch.from_numpy(np.array(image)).permute(2,0,1).float()/255)
            annotated.append(torch.stack(images))
        left, right = annotated
    return {"overlap_GT_left_prediction_right":rows(left,right),
            "RGB_target_left_render_right":rows(rgb,result["rgb"].detach().float().clamp(0,1))}

