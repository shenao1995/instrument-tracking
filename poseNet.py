"""ResNet-34 paired-image absolute pose regression and fixed-step refinement."""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from pose_geometry import (detach_pose, matrix_to_rotation_6d, pose_vector,
                           rotation_6d_to_matrix)


class PoseNet(nn.Module):
    """6 image channels = target masked RGB + moving masked RGB, at every stage.

    Legacy 9-channel checkpoints are supported explicitly for comparison.
    R is SO(3), z positive, alpha in +/-90 deg, jaws in +/-80 deg, and
    theta_left + theta_right >= 0 by construction. All angles are radians.
    """
    def __init__(self, base_pose, pretrained=False, translation_min=(-.08, -.08, .025),
                 translation_max=(.08, .08, .22),input_mode="rgb_pair"):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights
        if input_mode not in ("rgb_pair","legacy_mask_pair"):
            raise ValueError(f"Unknown input mode: {input_mode}")
        self.input_mode = input_mode
        backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        old = backbone.conv1
        repeats = 2 if input_mode == "rgb_pair" else 3
        backbone.conv1 = nn.Conv2d(3*repeats,64,7,stride=2,padding=3,bias=False)
        with torch.no_grad():
            backbone.conv1.weight.copy_(old.weight.repeat(1,repeats,1,1)/repeats)
        # Preserve coarse spatial information for translation regression.
        backbone.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(2048 + 12 + 4 + 1, 512), nn.ReLU(inplace=True),
                                  nn.Dropout(.1), nn.Linear(512, 12))
        self.register_buffer("base_pose", torch.as_tensor(base_pose).float().clone())
        self.register_buffer("translation_min", torch.tensor(translation_min))
        self.register_buffer("translation_max", torch.tensor(translation_max))
        self.register_buffer("rgb_mean", torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor([.229, .224, .225])[None, :, None, None])
        nn.init.normal_(self.head[-1].weight, std=1e-4)
        with torch.no_grad():
            bias = torch.zeros(12)
            bias[:6] = self.base_pose[:6]
            t = (self.base_pose[6:9]-self.translation_min)/(self.translation_max-self.translation_min)
            bias[6:9] = torch.logit(t.clamp(.001, .999))
            bias[11] = torch.logit(torch.tensor(5/80))  # half opening = 5 deg
            self.head[-1].bias.copy_(bias)

    def decode(self, raw):
        lim = math.radians(80)
        half_open = lim * raw[:, 11].sigmoid()
        yaw = (lim-half_open) * raw[:, 10].tanh()
        joints = torch.stack((math.pi/2 * raw[:, 9].tanh(),
                              half_open+yaw, half_open-yaw), -1)
        return {"R": rotation_6d_to_matrix(raw[:, :6]),
                "t": self.translation_min + raw[:, 6:9].sigmoid() * (self.translation_max-self.translation_min),
                "joints": joints}

    def forward(self, target_rgb, target_mask, moving_mask, current_pose, k, stage=0,moving_rgb=None):
        b, _, h, w = target_mask.shape
        if self.input_mode == "rgb_pair":
            if target_rgb is None or moving_rgb is None:
                raise ValueError("RGB-pair PoseNet requires target RGB and differentiably rendered moving RGB")
            target_alpha = target_mask.sum(1,keepdim=True).clamp(0,1)
            moving_alpha = moving_mask.sum(1,keepdim=True).clamp(0,1)
            # Inputs are already RGB * coverage; do not mask soft edges twice.
            target = (target_rgb-self.rgb_mean*target_alpha)/self.rgb_std
            moving = (moving_rgb-self.rgb_mean*moving_alpha)/self.rgb_std
            images = torch.cat((target,moving),1)
        elif target_rgb is None or stage > 0:
            rgb = torch.zeros_like(target_mask)
        else:
            foreground = target_mask.sum(1, keepdim=True).clamp(0, 1)
            rgb = (target_rgb-self.rgb_mean) / self.rgb_std * foreground
            rgb = rgb * (target_rgb.abs().sum((1,2,3)) > 0)[:,None,None,None]
        if self.input_mode == "legacy_mask_pair":
            images = torch.cat((rgb,target_mask,moving_mask),1)
        features = self.backbone(images)
        context = pose_vector(current_pose).clone()
        context[:, 6:9] = (context[:, 6:9]-self.translation_min) / (self.translation_max-self.translation_min)
        if k.ndim == 2:
            k = k[None].expand(b, -1, -1)
        camera = torch.stack((k[:,0,0]/w, k[:,1,1]/h, k[:,0,2]/w, k[:,1,2]/h), -1)
        stage_flag = features.new_full((b, 1), float(stage > 0))
        raw = self.head(torch.cat((features, context, camera, stage_flag), -1))
        # Rendering/geometry remains float32 under mixed precision.
        return self.decode(raw.float())


def predict_stages(model, renderer, target_rgb, target_mask, initial_pose, k,
                   steps=2, detach_between_stages=False, render_final=True):
    """Fixed-cost feed-forward loop. No test-time optimization or backward pass.

    Every prediction has its own loss in training. Detaching intermediate renders
    is optional and saves memory; the current stage render always has gradients.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    pose = initial_pose
    with torch.no_grad(), torch.autocast(device_type=target_mask.device.type, enabled=False):
        initial_render = renderer(pose,k)
        moving = initial_render["mask"]
        moving_rgb = initial_render.get("rgb")
    outputs = []
    for stage in range(steps):
        pose = model(target_rgb,target_mask,moving,pose,k,stage,moving_rgb=moving_rgb)
        result = {"pose": pose}
        if stage < steps-1 or render_final:
            with torch.autocast(device_type=target_mask.device.type, enabled=False):
                result.update(renderer(pose, k))
        outputs.append(result)
        if stage < steps-1:
            moving = result["mask"]
            moving_rgb = result.get("rgb")
            if detach_between_stages:
                moving = moving.detach()
                if moving_rgb is not None:
                    moving_rgb = moving_rgb.detach()
                pose = detach_pose(pose)
    return outputs
