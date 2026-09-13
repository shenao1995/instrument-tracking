"""Shared data I/O. Real sample frames are test-only; transforms are initialization."""
from __future__ import annotations

import json
from pathlib import Path
import random
import re

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from pose_geometry import axis_rotation, matrix_to_rotation_6d, vector_pose
from pose_losses import extract_gripper_tips
from pose_tip_labels import check_tip_labels, gripper_scale


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_mask(path, image_size=None):
    with Image.open(path) as image:
        a = np.asarray(image)
    if a.ndim == 3:
        # Reference RGB files are stored as BGR semantics: blue shaft, green wrist,
        # red grippers. Identical RGB channels instead encode scalar labels.
        if np.array_equal(a[...,0], a[...,1]) and np.array_equal(a[...,1], a[...,2]):
            a = a[...,0]
        else:
            if a.shape[-1] not in (3,4):
                raise ValueError(f"Unsupported mask shape: {a.shape}")
            a = a[...,:3][...,::-1].astype(np.float32) / 255
            if image_size:
                a = cv2.resize(a, image_size[::-1], interpolation=cv2.INTER_NEAREST)
            return torch.from_numpy(a.copy()).permute(2,0,1)
    values = set(np.unique(a).tolist())
    if values <= {0,1,2,3}:
        codes = (1,2,3)
    elif values <= {0,10,20,30}:
        codes = (10,20,30)
    else:
        raise ValueError(f"{path}: unsupported labels {sorted(values)}. Expected 0/10/20/30 or 0/1/2/3; binary masks cannot recover part labels.")
    if image_size:
        a = cv2.resize(a, image_size[::-1], interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(np.stack([a == code for code in codes]).astype(np.float32))


def save_mask(mask, path):
    a = mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask
    confidence, labels = a.max(0), a.argmax(0)+1
    Image.fromarray(np.where(confidence > .5, labels*10, 0).astype(np.uint8)).save(path)


def synthetic_rgb(mask, generator=None):
    """Appearance randomization from masks, without using real test images."""
    device = mask.device
    b, _, h, w = mask.shape
    colors = torch.rand(b,3,3,device=device,generator=generator)*.6+.25
    # Mix metallic gray and independently colored appearances.
    gray = colors.mean(-1, keepdim=True)
    colors = .65*gray + .35*colors
    rgb = torch.einsum("bchw,bcd->bdhw", mask, colors)
    light = torch.rand(b,1,4,4,device=device,generator=generator)*.6+.7
    light = torch.nn.functional.interpolate(light, size=(h,w), mode="bilinear", align_corners=False)
    noise = torch.randn(rgb.shape,device=device,generator=generator)*.035
    return ((rgb*light+noise).clamp(0,1)*mask.sum(1,keepdim=True).clamp(0,1))


def perturb_pose(vector, rotation_degrees=15, translation_mm=5, joints_degrees=10):
    p = vector_pose(vector)
    random_angles = torch.randn_like(p["joints"])*np.deg2rad(rotation_degrees)
    r = axis_rotation(random_angles[:,0], "x") @ axis_rotation(random_angles[:,1], "y") @ axis_rotation(random_angles[:,2], "z")
    p["R"] = r @ p["R"]
    p["t"] = (p["t"] + torch.randn_like(p["t"])*translation_mm/1000)
    p["t"][:,2].clamp_(.03,.20)
    p["t"][:,:2].clamp_(-.07,.07)
    joints = p["joints"] + torch.randn_like(p["joints"])*np.deg2rad(joints_degrees)
    joints[:,0].clamp_(-np.pi/2,np.pi/2)
    half = (joints[:,1:].sum(-1)/2).clamp(0,np.deg2rad(80))
    yaw = (joints[:,1]-joints[:,2])/2
    yaw = torch.minimum(torch.maximum(yaw, -(np.deg2rad(80)-half)), np.deg2rad(80)-half)
    p["joints"] = torch.stack((joints[:,0],half+yaw,half-yaw), -1)
    return p


class SyntheticDataset(Dataset):
    def __init__(self, root, split="train", augment=False, tip_labels=None):
        self.root = Path(root)
        self.metadata = json.loads((self.root/"metadata.json").read_text(encoding="utf-8"))
        self.records = [json.loads(line) for line in (self.root/"manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        self.records = [row for row in self.records if row["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} samples in {root}; run create_data.py first")
        self.augment = augment
        self.tip_labels = tip_labels if tip_labels is not None else {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with np.load(self.root/row["file"], allow_pickle=False) as sample:
            data = {key: torch.from_numpy(sample[key].copy()).float() for key in
                    ("mask", "rgb", "pose_vector", "initial_vector", "K", "tips", "tip_confidence")}
        data["mask"] /= 255
        data["rgb"] /= 255
        if row["file"] not in self.tip_labels:
            self.tip_labels[row["file"]] = check_tip_labels(data["mask"],data["tips"],data["tip_confidence"])
        for key, value in self.tip_labels[row["file"]].items():
            data[key] = torch.as_tensor(np.array(value,copy=True)).float()
        if self.augment:
            # Preserve rendered appearance: random semantic colors or RGB
            # dropout would create inconsistent photometric supervision.
            if torch.rand(()) < .5:
                from pose_geometry import pose_vector
                data["initial_vector"] = pose_vector(perturb_pose(data["pose_vector"][None]))[0]
        data["pose"] = vector_pose(data.pop("pose_vector"))
        data["initial_pose"] = vector_pose(data.pop("initial_vector"))
        return data


def natural_key(path):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", path.stem)]


class RealFrameDataset(Dataset):
    def __init__(self, root, image_size, k, base_pose, mask_dir=None, color_dir=None, limit=0,
                 include_tip_labels=True):
        root = Path(root)
        self.mask_dir = Path(mask_dir) if mask_dir else root/"l_mask"
        if not self.mask_dir.exists() and (root/"l_masks").exists():
            self.mask_dir = root/"l_masks"
        self.color_dir = Path(color_dir) if color_dir else root/"color"
        self.files = sorted([p for p in self.mask_dir.iterdir() if p.suffix.lower() in (".png", ".tif", ".tiff", ".bmp")], key=natural_key)
        if limit:
            self.files = self.files[:limit]
        if not self.files:
            raise ValueError(f"No masks in {self.mask_dir}")
        self.image_size, self.k, self.base_pose = image_size, k, base_pose
        self.include_tip_labels = include_tip_labels

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        mask = read_mask(path, self.image_size)
        matches = [self.color_dir/(path.stem+ext) for ext in (".png", ".jpg", ".jpeg")]
        image_path = next((p for p in matches if p.exists()), None)
        if image_path is None:
            raise FileNotFoundError(f"Missing RGB frame for {path.stem} in {self.color_dir}")
        with Image.open(image_path) as im:
            im = im.convert("RGB").resize(self.image_size[::-1], Image.Resampling.BILINEAR)
            rgb = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255
        rgb *= mask.sum(0,keepdim=True).clamp(0,1)
        row = {"name":path.stem,"rgb":rgb,"mask":mask,"K":self.k.clone(),
               "initial_pose":vector_pose(self.base_pose.clone())}
        if self.include_tip_labels:
            tips,confidence = extract_gripper_tips(mask)
            row.update(tips=torch.from_numpy(tips),tip_confidence=torch.from_numpy(confidence),
                       tip_scale=torch.tensor(gripper_scale(mask)))
        return row


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k,v in value.items()}
    return value
