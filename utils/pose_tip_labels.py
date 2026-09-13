"""Mask-derived distal tips and conservative checks of synthetic FK landmarks.

Synthetic coordinates remain projected mesh cap centres. Mask pseudo-landmarks
can validate them, but never replace them with a different landmark definition.
The cache is a sidecar: source NPZs, renderer and dataset metadata are unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

TIP_LABEL_VERSION = 2
LABEL_KEYS = ("tip_confidence", "tip_original_confidence", "tip_scale",
              "mask_tips", "mask_tip_confidence", "tip_rescued")


def _numpy_mask(mask):
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.shape[0] != 3:
        raise ValueError("Expected semantic mask [3,H,W]")
    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255
    return mask


def gripper_scale(mask):
    """Isotropic GT bbox diagonal, floored at 10 px; no predicted quantities."""
    ys, xs = np.nonzero(_numpy_mask(mask)[2] > .5)
    if not len(xs):
        return np.float32(10.)
    return np.float32(max(10., np.hypot(xs.max()-xs.min()+1, ys.max()-ys.min()+1)))


def _geodesic(component, wrist_distance):
    y, x = np.nonzero(component)
    ids = np.full(component.shape, -1, np.int32)
    ids[y, x] = np.arange(len(x))
    rows, cols, values = [], [], []
    h, w = component.shape
    for dy, dx in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        ny, nx = y+dy, x+dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        src = np.flatnonzero(inside)
        dst = ids[ny[inside], nx[inside]]
        keep = dst >= 0
        rows.append(src[keep])
        cols.append(dst[keep])
        values.append(np.full(keep.sum(), np.hypot(dy, dx)))
    graph = csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(len(x), len(x)))
    # A band at the wrist junction avoids bias toward one corner of a thick jaw.
    root_distance = wrist_distance[y, x]
    roots = np.flatnonzero(root_distance <= root_distance.min()+1.)
    distances = dijkstra(graph, indices=roots, min_only=True, directed=False)
    image = np.zeros(component.shape, np.float64)
    image[y, x] = distances
    return image


def extract_gripper_tips(mask, min_area=8):
    """Wrist-rooted geodesic branches; return XY pseudo-tips and confidence.

    Crop first, remove small disconnected speckles, then try several proximal
    cutoffs. Require long distal branches; a short spur is not a second jaw.
    Border-clipped tips and unresolved overlapping jaws never become valid pairs.
    """
    mask = _numpy_mask(mask)
    grip, wrist = mask[2] > .5, mask[1] > .5
    points, confidence = np.zeros((2,2), np.float32), np.zeros(2, np.float32)
    if not wrist.any() or grip.sum() < min_area:
        return points, confidence
    h, w = grip.shape
    ys, xs = np.nonzero(grip | wrist)
    y0, y1 = max(0, ys.min()-2), min(h, ys.max()+3)
    x0, x1 = max(0, xs.min()-2), min(w, xs.max()+3)
    grip, wrist = grip[y0:y1,x0:x1], wrist[y0:y1,x0:x1]
    wrist_distance = ndimage.distance_transform_edt(~wrist)
    components, _ = ndimage.label(grip, np.ones((3,3)))
    sizes = np.bincount(components.ravel())
    candidates = [i for i in range(1,len(sizes)) if sizes[i] >= max(min_area, grip.sum()*.025)]
    candidates.sort(key=lambda i: (wrist_distance[components == i].min(), -sizes[i]))
    endpoints = []
    for cid in candidates[:2]:
        component = components == cid
        # Detached distant islands are ambiguous, not additional fingertips.
        if wrist_distance[component].min() > max(4., .12*np.hypot(*grip.shape)):
            continue
        distance = _geodesic(component, wrist_distance)
        maximum = distance[component].max()
        if maximum < 4 or not np.isfinite(maximum):
            continue
        chosen = None
        for fraction in (.35, .45, .55, .65, .75):
            branches, nb = ndimage.label(component & (distance > fraction*maximum), np.ones((3,3)))
            parts = []
            for j in range(1,nb+1):
                coords = np.argwhere(branches == j)
                d = distance[coords[:,0],coords[:,1]]
                if len(coords) >= max(3,min_area//2) and d.max() >= .4*maximum and np.ptp(d) >= max(2., .12*d.max()):
                    parts.append(coords)
            if len(parts) == 2:
                chosen = parts
                break
        if chosen is None:
            chosen = [np.argwhere(component)]
        for coords in chosen:
            d = distance[coords[:,0],coords[:,1]]
            cap = coords[d >= d.max()-1.5]
            cap_mask = np.zeros_like(component)
            cap_mask[cap[:,0],cap[:,1]] = True
            cap_labels, cap_count = ndimage.label(cap_mask,np.ones((3,3)))
            if cap_count > 1:
                # Never average disconnected maxima into an imaginary midpoint.
                cap_id = 1+np.argmax(np.bincount(cap_labels.ravel())[1:])
                cap = np.argwhere(cap_labels == cap_id)
            point = cap.mean(0)[::-1] + np.array([x0,y0])
            clipped = ((cap[:,0]+y0 <= 1) | (cap[:,0]+y0 >= h-2) |
                       (cap[:,1]+x0 <= 1) | (cap[:,1]+x0 >= w-2)).any()
            endpoints.append((point, 0. if clipped else 1., float(d.max())))
    endpoints.sort(key=lambda item: item[2], reverse=True)
    selected = []
    for point, conf, _ in endpoints:
        if all(np.linalg.norm(point-other[0]) > 3 for other in selected):
            selected.append((point,conf))
        if len(selected) == 2:
            break
    for i, (point, conf) in enumerate(selected):
        points[i], confidence[i] = point, conf
    if len(selected) == 1:
        confidence[0] *= .4
    return points, confidence


def check_tip_labels(mask, tips, confidence):
    """Keep FK coordinates; rescue only a complete, distinct mask-to-FK match.

    A 3-pixel cap-centre/silhouette tolerance covers rasterization, not arbitrary
    hidden tips. Rescued tips carry 0.5 confidence. Amodal/occluded coordinates
    are not silently turned into visible supervision.
    """
    mask = _numpy_mask(mask)
    tips = np.asarray(tips, np.float32)
    original = np.asarray(confidence, np.float32)
    if tips.shape != (2,2) or original.shape != (2,):
        raise ValueError("Expected two XY tips and two confidence values")
    mask_tips, mask_conf = extract_gripper_tips(mask)
    h, w = mask.shape[-2:]
    finite = np.isfinite(tips).all(-1)
    safe = np.where(np.isfinite(tips), tips, 0.)
    inside = finite & (safe[:,0] >= 2) & (safe[:,0] < w-2) & (safe[:,1] >= 2) & (safe[:,1] < h-2)
    grip = mask[2] > .5
    support = np.zeros(2, bool)
    if grip.any():
        distance = ndimage.distance_transform_edt(~grip)
        xy = np.rint(safe).astype(np.int64)
        support = distance[xy[:,1].clip(0,h-1),xy[:,0].clip(0,w-1)] <= 2.
    valid = inside & support
    checked = np.where(valid, np.nan_to_num(original, nan=0., posinf=0., neginf=0.).clip(0,1), 0.).astype(np.float32)
    rescued = np.zeros(2, np.float32)
    if (mask_conf > 0).all() and valid.all() and np.linalg.norm(safe[0]-safe[1]) > 3:
        errors = [np.linalg.norm(safe-mask_tips[order], axis=-1) for order in ([0,1],[1,0])]
        error = min(errors, key=lambda e: e.max())
        if (error <= 3.).all():
            rescued = (checked == 0).astype(np.float32)
            checked = np.maximum(checked, .5)
    return {"tip_confidence": checked, "tip_original_confidence": original.copy(),
            "tip_scale": gripper_scale(mask), "mask_tips": mask_tips,
            "mask_tip_confidence": mask_conf, "tip_rescued": rescued}


def prepare_tip_cache(root, records=None):
    """Cache all training data by default, or only explicitly selected records.

    Subsets use separate caches and never read/stat unselected sample NPZs.
    The original full-dataset fingerprint is retained for training resume.
    """
    root = Path(root)
    manifest = (root/"manifest.jsonl").read_bytes()
    all_records = [json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()]
    suffix = ""
    if records is None:
        records = all_records
    else:
        records = list(records)
        known = {row["file"]:row for row in all_records}
        if not records or any(known.get(row["file"]) != row for row in records):
            raise ValueError("Tip cache subset must contain nonempty records from the manifest")
        if len({row["file"] for row in records}) != len(records):
            raise ValueError("Duplicate samples in tip cache subset")
        manifest = json.dumps(records,sort_keys=True,separators=(",",":")).encode()
        suffix = "_subset_"+hashlib.sha256(manifest).hexdigest()[:12]
    files = [row["file"] for row in records]
    signature = hashlib.sha256(manifest+(root/"metadata.json").read_bytes())
    signature.update(str(TIP_LABEL_VERSION).encode())
    for fn in (_numpy_mask,gripper_scale,_geodesic,extract_gripper_tips,check_tip_labels):
        signature.update(inspect.getsource(fn).encode())
    for name in files:
        stat = (root/name).stat()
        signature.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    fingerprint = signature.hexdigest()
    path = root/f"tip_labels_v{TIP_LABEL_VERSION}{suffix}.npz"
    arrays = None
    if path.exists():
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["fingerprint"]) == fingerprint and cache["files"].tolist() == files:
                arrays = {key:cache[key].copy() for key in LABEL_KEYS}
    if arrays is None:
        print(f"Checking distal tip labels for {len(files)} samples (no rendering)...", flush=True)
        rows, start = [], time.perf_counter()
        for i, name in enumerate(files):
            with np.load(root/name, allow_pickle=False) as sample:
                rows.append(check_tip_labels(sample["mask"],sample["tips"],sample["tip_confidence"]))
            if (i+1) % 250 == 0 or i+1 == len(files):
                print(f"Tip labels: {i+1}/{len(files)}, {time.perf_counter()-start:.1f}s", flush=True)
        arrays = {key:np.stack([row[key] for row in rows]) for key in LABEL_KEYS}
        temp = path.with_suffix(".tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, fingerprint=np.array(fingerprint), files=np.array(files), **arrays)
        temp.replace(path)
    report = {"version":TIP_LABEL_VERSION,"fingerprint":fingerprint,"samples":len(files)}
    for split in ("train","val"):
        selected = np.array([row["split"] == split for row in records])
        if not selected.any():
            continue
        conf, original = arrays["tip_confidence"][selected], arrays["tip_original_confidence"][selected]
        report[split] = {"samples":int(selected.sum()),
                         "original_pair_ratio":float((original > 0).all(-1).mean()),
                         "checked_pair_ratio":float((conf > 0).all(-1).mean()),
                         "no_valid_tip_ratio":float((conf == 0).all(-1).mean()),
                         "rescued_tips":int(arrays["tip_rescued"][selected].sum()),
                         "invalidated_tips":int(((original > 0) & (conf == 0)).sum())}
    path.with_suffix(".json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Tip label audit: "+json.dumps(report),flush=True)
    return {name:{key:arrays[key][i] for key in LABEL_KEYS} for i,name in enumerate(files)}, report


def save_tip_audit(root, labels, output, count=24):
    """Deterministic examples from rescued, originally visible and unresolved groups."""
    from PIL import Image, ImageDraw
    root, output = Path(root), Path(output)
    rng = np.random.default_rng(2026)
    names = list(labels)
    groups = [[],[],[]]
    for name in names:
        row = labels[name]
        group = 0 if row["tip_rescued"].any() else (1 if (row["tip_confidence"] > 0).all() else 2)
        groups[group].append(name)
    chosen = []
    for group in groups:
        chosen.extend(rng.choice(group,min(len(group),max(1,count//3)),replace=False).tolist())
    canvas = Image.new("RGB",(4*320,((len(chosen)+3)//4)*240),(20,20,20))
    for i, name in enumerate(chosen):
        row = labels[name]
        with np.load(root/name,allow_pickle=False) as sample:
            mask = sample["mask"].astype(np.float32)/255
            points = sample["tips"]
            opening = np.rad2deg(sample["pose_vector"][-2:].mean())
        palette = np.array([[.65,.2,.2],[.2,.65,.2],[.2,.3,.8]])
        rgb = (np.einsum("chw,cd->hwd",mask,palette).clip(0,1)*255).astype(np.uint8)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        valid = row["tip_confidence"] > 0
        if valid.all():
            draw.line([tuple(p) for p in points],fill=(255,230,0),width=1)
        for p,good in zip(points,valid):
            x,y = p
            if np.isfinite(p).all():
                color = (255,230,0) if good else (180,180,180)
                draw.ellipse((x-2,y-2,x+2,y+2),outline=color,width=1)
        for (x,y),conf in zip(row["mask_tips"],row["mask_tip_confidence"]):
            if conf > 0:
                draw.line((x-2,y-2,x+2,y+2),fill=(0,255,255),width=1)
                draw.line((x-2,y+2,x+2,y-2),fill=(0,255,255),width=1)
        ys,xs = np.nonzero(mask[1:].sum(0) > .5)
        if len(xs):
            image = image.crop((max(0,int(xs.min())-8),max(0,int(ys.min())-8),
                                min(image.width,int(xs.max())+9),min(image.height,int(ys.max())+9)))
        image.thumbnail((310,190),Image.Resampling.NEAREST)
        x,y = (i%4)*320,(i//4)*240
        canvas.paste(image,(x+(320-image.width)//2,y+40+(190-image.height)//2))
        draw = ImageDraw.Draw(canvas)
        old = int((row["tip_original_confidence"] > 0).sum())
        draw.text((x+4,y+3),f"{Path(name).stem}  half-open {opening:.1f} deg",fill="white")
        draw.text((x+4,y+17),f"Valid tips {old} -> {valid.sum()} | FK yellow / mask cyan",fill="white")
    output.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(output)
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--preview",type=Path,help="Optional grid showing FK tips and mask-derived tips")
    args = parser.parse_args()
    labels, _ = prepare_tip_cache(args.data)
    if args.preview:
        print(save_tip_audit(args.data,labels,args.preview))
