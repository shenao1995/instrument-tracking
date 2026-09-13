"""Soft semantic overlap, reliable projected tips, and articulated pose losses."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from pose_tip_labels import extract_gripper_tips


def soft_mask_loss(pred, target):
    """No thresholding of predictions; part-balanced Dice + BCE at 3 scales."""
    losses = []
    for scale in (1, 2, 4):
        p = pred if scale == 1 else F.avg_pool2d(pred, scale)
        t = target if scale == 1 else F.avg_pool2d(target, scale)
        dims = (-2, -1)
        dice = 1 - (2*(p*t).sum(dims)+1) / (p.sum(dims)+t.sum(dims)+1)
        bce = F.binary_cross_entropy(p.clamp(1e-5, 1-1e-5), t, reduction="none")
        foreground = (bce*t).sum(dims) / t.sum(dims).clamp_min(1)
        background = (bce*(1-t)).sum(dims) / (1-t).sum(dims).clamp_min(1)
        losses.append(dice.mean() + .25*(foreground+background).mean())
    return torch.stack(losses).mean()


def tip_scales(mask):
    """GT gripper bbox diagonal in pixels, floored at 10; fully detached."""
    grip = mask[:, 2].detach() > .5
    h, w = grip.shape[-2:]
    x = torch.arange(w, device=mask.device)[None, :]
    y = torch.arange(h, device=mask.device)[None, :]
    cols, rows = grip.any(1), grip.any(2)
    width = (torch.where(cols, x, -1).amax(-1)-torch.where(cols, x, w).amin(-1)+1).clamp_min(0)
    height = (torch.where(rows, y, -1).amax(-1)-torch.where(rows, y, h).amin(-1)+1).clamp_min(0)
    return torch.hypot(width.float(), height.float()).clamp_min(10.)


def _tip_inputs(pred, target, confidence):
    confidence = torch.nan_to_num(confidence.detach(), nan=0., posinf=0., neginf=0.).clamp(0, 1)
    confidence = confidence * torch.isfinite(target).all(-1)
    # Invalid labels may contain NaNs. Masking a NaN loss afterwards is too late.
    return pred.float(), torch.nan_to_num(target.detach().float(), nan=0., posinf=0., neginf=0.), confidence.float()


def endpoint_loss(pred, target, confidence, image_size=None, scale=None):
    """Permutation-invariant positions; local isotropic scale in new training."""
    pred, target, confidence = _tip_inputs(pred, target, confidence)
    if scale is None:
        if image_size is None:
            raise ValueError("Provide GT tip scale or image_size")
        scale = pred.new_full((len(pred),), float(sum(v*v for v in image_size))**.5)
    scale = scale.detach().reshape(-1, 1, 1).clamp_min(10.)

    def cost(order):
        error = F.smooth_l1_loss(pred[:, order]/scale, target/scale, beta=.05, reduction="none").sum(-1)
        return (error*confidence).sum(-1)/confidence.sum(-1).clamp_min(1e-8)

    sample_weight = confidence.amax(-1)
    costs = torch.minimum(cost([0,1]), cost([1,0]))
    return (costs*sample_weight).sum()/sample_weight.sum().clamp_min(1e-8)


def tip_gap_loss(pred, target, confidence, scale):
    """Projected FK separation; no predicted-mask thresholding or GT-gap division."""
    pred, target, confidence = _tip_inputs(pred, target, confidence)
    predicted_gap = torch.linalg.vector_norm(pred[:,0]-pred[:,1], dim=-1)
    target_gap = torch.linalg.vector_norm(target[:,0]-target[:,1], dim=-1)
    error = (predicted_gap-target_gap)/scale.detach().reshape(-1).clamp_min(10.)
    weight = confidence.amin(-1)
    cost = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.05, reduction="none")
    return (cost*weight).sum()/weight.sum().clamp_min(1e-8)


@torch.no_grad()
def tip_metrics(pred, batch):
    """Sufficient statistics: epoch gap MAE is divided by valid PAIRS, not batches."""
    pred, target, confidence = _tip_inputs(pred, batch["tips"], batch["tip_confidence"])
    valid = (confidence > 0).all(-1)
    error = (torch.linalg.vector_norm(pred[:,0]-pred[:,1],dim=-1)-
             torch.linalg.vector_norm(target[:,0]-target[:,1],dim=-1)).abs()
    return {"tip_pair_count":int(valid.sum()), "tip_gap_error_sum_px":float(error[valid].sum())}


def total_loss(result, batch, weights=None):
    """RGB + mask + locally normalized positions/separation; bounds stay in PoseNet."""
    weights = weights or {"rgb":.1, "mask":1., "tips":1.}
    scale = batch.get("tip_scale")
    if scale is None:
        scale = tip_scales(batch["mask"])
    position = endpoint_loss(result["tips"], batch["tips"], batch["tip_confidence"], scale=scale)
    gap = tip_gap_loss(result["tips"], batch["tips"], batch["tip_confidence"], scale)
    terms = {
        "rgb": masked_rgb_loss(result["rgb"],batch["rgb"],result["mask"],batch["mask"]),
        "mask": soft_mask_loss(result["mask"],batch["mask"]),
        "tips_position": position,
        "tips_gap": gap,
        "tips": weights.get("tips_position",1.)*position + weights.get("tips_gap",1.)*gap,
    }
    return sum(weights[name]*terms[name] for name in ("rgb","mask","tips")), terms


def masked_rgb_loss(pred,target,pred_mask,target_mask):
    """Multiscale Charbonnier on premultiplied RGB, normalized by foreground union.

    Both RGB images already have black background. Detach only normalization
    support, never predicted RGB; empty-background pixels cannot dilute loss.
    """
    losses = []
    union = (pred_mask.sum(1,keepdim=True)+target_mask.sum(1,keepdim=True)).clamp(0,1).detach()
    for scale in (1,2,4):
        p,t,w = (F.avg_pool2d(x,scale) if scale > 1 else x for x in (pred,target,union))
        robust = ((p-t).square()+1e-6).sqrt()-1e-3
        losses.append(((robust*w).sum((1,2,3))/(3*w.sum((1,2,3))).clamp_min(1)).mean())
    return torch.stack(losses).mean()


def part_dice(pred,target):
    """Per-sample, per-part hard Dice; absent in both images is undefined."""
    p,t = pred > .5,target > .5
    denominator = p.sum((-2,-1))+t.sum((-2,-1))
    return torch.where(denominator > 0,2*(p&t).sum((-2,-1))/denominator.clamp_min(1),torch.nan)


def part_iou(pred, target):
    p, t = pred > .5, target > .5

    inter = (
        p & t
    ).sum((-2, -1)).float()

    union = (
        p | t
    ).sum((-2, -1))

    return torch.where(
        union > 0,
        inter / union.clamp_min(1),
        torch.nan
    )
