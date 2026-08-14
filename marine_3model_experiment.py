#!/usr/bin/env python3
"""
RunPod-ready controlled comparison for the paper:
YOLOv8s vs Faster R-CNN ResNet-50-FPN vs MobileNet SSD/SSDLite320 MobileNetV3-Large
on Trash-ICRA19, plus class-agnostic cross-domain testing on River Floating Trash.

Main design choices:
- One deterministic image-level split for all models.
- YOLO-format dataset preparation with label validation.
- TorchMetrics mAP for in-domain and cross-domain evaluation.
- Cross-domain river labels are remapped to one class to avoid the common bug where
  YOLO ignores label IDs > 0 when nc=1.
- CSV + Markdown outputs ready to paste into an IEEE paper.

Authoring note: results should be reported exactly as produced by this script, not
hardcoded from older notebook runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# TorchVision imports are intentionally inside functions where possible, because
# a mismatched torch/torchvision CUDA image fails early. The README explains this.

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "TorchMetrics detection dependencies are missing. Run: pip install 'torchmetrics[detection]' pycocotools"
    ) from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CLASSES = ["plastic", "bio", "rov"]


# -----------------------------
# Utilities
# -----------------------------


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def link_or_copy(src: Path, dst: Path, copy_files: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_files:
        shutil.copy2(src, dst)
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_pct(x: float) -> float:
    return round(float(x) * 100.0, 3)


def xywhn_to_xyxy(box: Sequence[float], w: int, h: int) -> List[float]:
    cx, cy, bw, bh = map(float, box)
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return [x1, y1, x2, y2]


def clip_xyxy(box: Sequence[float], w: int, h: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = map(float, box)
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    return [x1, y1, x2, y2]


def xyxy_to_xywhn(box: Sequence[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = map(float, box)
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
    xa = np.maximum(x11[:, None], x21[None, :])
    ya = np.maximum(y11[:, None], y21[None, :])
    xb = np.minimum(x12[:, None], x22[None, :])
    yb = np.minimum(y12[:, None], y22[None, :])
    inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    area1 = np.maximum(0, x12 - x11) * np.maximum(0, y12 - y11)
    area2 = np.maximum(0, x22 - x21) * np.maximum(0, y22 - y21)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass
class YoloRecord:
    image: Path
    label: Optional[Path]
    split_hint: Optional[str] = None


@dataclass
class PreparedPaths:
    trash_yaml: Path
    river_yaml: Optional[Path]
    trash_dir: Path
    river_dir: Optional[Path]
    summary_csv: Path


# -----------------------------
# Dataset scanning and preparation
# -----------------------------


def find_dataset_yaml(root: Path) -> Optional[Path]:
    for name in ("data.yaml", "dataset.yaml", "trash.yaml", "obj.yaml"):
        p = root / name
        if p.exists():
            return p
    candidates = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))
    for p in candidates:
        try:
            data = read_yaml(p)
        except Exception:
            continue
        if "names" in data and ("train" in data or "path" in data):
            return p
    return None


def detect_split_from_parts(path: Path) -> Optional[str]:
    parts = [p.lower() for p in path.parts]
    aliases = {
        "train": "train", "training": "train",
        "val": "val", "valid": "val", "validation": "val",
        "test": "test", "testing": "test",
    }
    for p in parts:
        if p in aliases:
            return aliases[p]
    return None


def label_for_image(image_path: Path, root: Path) -> Optional[Path]:
    """Find YOLO txt label for an image in common layouts."""
    stem = image_path.stem + ".txt"
    candidates: List[Path] = []

    # Standard YOLO layout: .../images/.../x.jpg -> .../labels/.../x.txt
    parts = list(image_path.parts)
    lower = [p.lower() for p in parts]
    if "images" in lower:
        idx = lower.index("images")
        replaced = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
        candidates.append(replaced)

    # Same directory or sibling labels folders
    candidates.append(image_path.with_suffix(".txt"))
    candidates.append(root / "labels" / stem)
    candidates.append(root / "Annotations" / stem)
    candidates.append(root / "annotations" / stem)

    for c in candidates:
        if c.exists():
            return c
    return None


def scan_yolo_records(root: Path) -> List[YoloRecord]:
    root = root.resolve()
    image_paths = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])
    records: List[YoloRecord] = []
    for img in image_paths:
        # Avoid accidentally scanning previous runs.
        if any(part.lower() in {"runs", "wandb", "__macosx"} for part in img.parts):
            continue
        label = label_for_image(img, root)
        records.append(YoloRecord(image=img, label=label, split_hint=detect_split_from_parts(img)))
    return records


def read_yolo_label(label_path: Optional[Path], num_classes: int, image_size: Tuple[int, int], remap_all_to_zero: bool = False) -> List[Tuple[int, float, float, float, float]]:
    """Return valid YOLO normalized labels. Invalid boxes/classes are skipped."""
    if label_path is None or not label_path.exists():
        return []
    w, h = image_size
    labels: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                vals = list(map(float, parts[1:5]))
            except ValueError:
                continue
            if any(not np.isfinite(v) for v in vals):
                continue
            if remap_all_to_zero:
                cls = 0
            elif cls < 0 or cls >= num_classes:
                continue
            xyxy = clip_xyxy(xywhn_to_xyxy(vals, w, h), w, h)
            if xyxy is None:
                continue
            cx, cy, bw, bh = xyxy_to_xywhn(xyxy, w, h)
            # Keep numbers safe inside YOLO expected range.
            if bw <= 0 or bh <= 0:
                continue
            labels.append((cls, cx, cy, bw, bh))
    return labels


def image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as im:
        return im.size


def dominant_class(labels: List[Tuple[int, float, float, float, float]], num_classes: int) -> int:
    if not labels:
        return num_classes  # separate stratum for no-object images
    counts = Counter(int(x[0]) for x in labels)
    return int(counts.most_common(1)[0][0])


def stratified_split(records: List[YoloRecord], class_count_by_record: Dict[Path, int], seed: int) -> Dict[str, List[YoloRecord]]:
    y = [class_count_by_record[r.image] for r in records]
    try:
        train, tmp, y_train, y_tmp = train_test_split(
            records, y, test_size=0.30, random_state=seed, stratify=y
        )
        val, test, _, _ = train_test_split(
            tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
        )
    except ValueError as e:
        warnings.warn(f"Stratified split failed ({e}); falling back to random split.")
        train, tmp = train_test_split(records, test_size=0.30, random_state=seed, shuffle=True)
        val, test = train_test_split(tmp, test_size=0.50, random_state=seed, shuffle=True)
    return {"train": list(train), "val": list(val), "test": list(test)}


def official_split(records: List[YoloRecord]) -> Dict[str, List[YoloRecord]]:
    splits = {"train": [], "val": [], "test": []}
    for r in records:
        if r.split_hint in splits:
            splits[r.split_hint].append(r)
    if not all(len(v) > 0 for v in splits.values()):
        raise RuntimeError(
            "split_mode=official was requested, but train/val/test folders were not detected. "
            "Use split_mode=stratified_70_15_15 instead."
        )
    return splits


def materialize_yolo_dataset(
    records_by_split: Dict[str, List[YoloRecord]],
    dst: Path,
    class_names: List[str],
    copy_files: bool,
    remap_all_to_zero: bool = False,
) -> pd.DataFrame:
    clean_dir(dst)
    rows = []
    nc = 1 if remap_all_to_zero else len(class_names)
    out_names = ["trash"] if remap_all_to_zero else class_names

    for split, records in records_by_split.items():
        for idx, r in enumerate(tqdm(records, desc=f"Preparing {dst.name}/{split}")):
            try:
                w, h = image_size(r.image)
            except Exception:
                continue
            labels = read_yolo_label(r.label, nc if not remap_all_to_zero else 10_000, (w, h), remap_all_to_zero=remap_all_to_zero)
            # Keep unlabeled images in test/val? For training, unlabeled images can be kept but not useful.
            # We keep all images for transparent accounting.
            suffix = r.image.suffix.lower()
            safe_name = f"{r.image.stem}_{abs(hash(str(r.image))) % 10**8}{suffix}"
            img_dst = dst / "images" / split / safe_name
            lbl_dst = dst / "labels" / split / (Path(safe_name).stem + ".txt")
            link_or_copy(r.image, img_dst, copy_files)
            lbl_dst.parent.mkdir(parents=True, exist_ok=True)
            with lbl_dst.open("w", encoding="utf-8") as f:
                for cls, cx, cy, bw, bh in labels:
                    f.write(f"{int(cls)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}\n")
            counts = Counter(int(x[0]) for x in labels)
            row = {
                "split": split,
                "image": str(img_dst),
                "label": str(lbl_dst),
                "source_image": str(r.image),
                "width": w,
                "height": h,
                "objects": len(labels),
            }
            for i, name in enumerate(out_names):
                row[name] = counts.get(i, 0)
            rows.append(row)

    write_yaml(dst / "data.yaml", {
        "path": str(dst.resolve()),
        "train": "images/train",
        "val": "images/val" if (dst / "images" / "val").exists() else "images/test",
        "test": "images/test",
        "nc": len(out_names),
        "names": out_names,
    })
    df = pd.DataFrame(rows)
    df.to_csv(dst / "dataset_index.csv", index=False)
    return df


def prepare_datasets(cfg: dict) -> PreparedPaths:
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    data_out = out_dir / "prepared_datasets"
    data_out.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg.get("class_names") or DEFAULT_CLASSES)
    trash_root = Path(cfg["trash_root"]).expanduser().resolve()
    river_root = Path(cfg.get("river_root", "")).expanduser().resolve() if cfg.get("river_root") else None
    seed = int(cfg.get("seed", 42))
    split_mode = cfg.get("split_mode", "stratified_70_15_15")
    copy_files = bool(cfg.get("copy_files", False))

    print(f"[{now()}] Scanning Trash dataset: {trash_root}")
    trash_records_all = scan_yolo_records(trash_root)
    if not trash_records_all:
        raise RuntimeError(f"No images found under {trash_root}")

    # Validate class labels and skip images with no label file only if they are truly empty.
    dom = {}
    kept = []
    skipped_bad = 0
    for r in tqdm(trash_records_all, desc="Validating Trash labels"):
        try:
            size = image_size(r.image)
        except Exception:
            skipped_bad += 1
            continue
        labels = read_yolo_label(r.label, len(class_names), size, remap_all_to_zero=False)
        # Keep images even if labels are empty; they are useful as background validation/test cases.
        dom[r.image] = dominant_class(labels, len(class_names))
        kept.append(r)
    trash_records = kept
    if skipped_bad:
        print(f"Skipped {skipped_bad} unreadable images.")

    if split_mode == "official":
        trash_splits = official_split(trash_records)
    elif split_mode == "stratified_70_15_15":
        trash_splits = stratified_split(trash_records, dom, seed)
    else:
        raise ValueError("split_mode must be 'official' or 'stratified_70_15_15'")

    if bool(cfg.get("run", {}).get("quick_debug", False)):
        for k in trash_splits:
            trash_splits[k] = trash_splits[k][: min(32, len(trash_splits[k]))]

    trash_dst = data_out / "trash_icra19_clean"
    trash_df = materialize_yolo_dataset(trash_splits, trash_dst, class_names, copy_files, remap_all_to_zero=False)

    river_yaml = None
    river_dst = None
    river_df = None
    if river_root and river_root.exists():
        print(f"[{now()}] Scanning River dataset: {river_root}")
        river_records = scan_yolo_records(river_root)
        if river_records:
            # River is used only for cross-domain evaluation; put all records in test.
            if bool(cfg.get("run", {}).get("quick_debug", False)):
                river_records = river_records[:64]
            river_dst = data_out / "river_trash_class_agnostic"
            river_df = materialize_yolo_dataset({"test": river_records}, river_dst, ["trash"], copy_files, remap_all_to_zero=True)
            river_yaml = river_dst / "data.yaml"
        else:
            warnings.warn(f"No river images found under {river_root}; cross-domain evaluation will be skipped.")

    # Summary
    summary_rows = []
    for name, df in [("Trash-ICRA19", trash_df), ("River", river_df)]:
        if df is None:
            continue
        for split, group in df.groupby("split"):
            row = {"dataset": name, "split": split, "images": len(group), "objects": int(group["objects"].sum())}
            for cls_name in (["trash"] if name == "River" else class_names):
                if cls_name in group.columns:
                    row[cls_name] = int(group[cls_name].sum())
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "dataset_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print("\nDataset summary:")
    print(summary.to_string(index=False))

    return PreparedPaths(
        trash_yaml=trash_dst / "data.yaml",
        river_yaml=river_yaml,
        trash_dir=trash_dst,
        river_dir=river_dst,
        summary_csv=summary_csv,
    )


# -----------------------------
# Torch dataset and loaders
# -----------------------------


def letterbox_image_and_boxes(
    img: Image.Image,
    boxes: np.ndarray,
    size: int,
    fill: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[Image.Image, np.ndarray]:
    orig_w, orig_h = img.size
    scale = min(size / orig_w, size / orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), fill)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    if len(boxes):
        boxes = boxes.astype(np.float32).copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, size - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, size - 1)
    return canvas, boxes


class YoloDetectionDataset(Dataset):
    def __init__(self, yolo_root: Path, split: str, input_size: int, num_classes: int, class_agnostic: bool = False):
        self.root = Path(yolo_root)
        self.split = split
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.class_agnostic = class_agnostic
        self.images = sorted([p for p in (self.root / "images" / split).glob("*") if p.suffix.lower() in IMG_EXTS])
        if not self.images:
            raise RuntimeError(f"No images found in {self.root / 'images' / split}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        label_path = self.root / "labels" / self.split / (img_path.stem + ".txt")
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        labels = read_yolo_label(
            label_path,
            num_classes=self.num_classes,
            image_size=(w, h),
            remap_all_to_zero=self.class_agnostic,
        )
        boxes_xyxy = []
        labels_out = []
        for cls, cx, cy, bw, bh in labels:
            xyxy = clip_xyxy(xywhn_to_xyxy([cx, cy, bw, bh], w, h), w, h)
            if xyxy is None:
                continue
            boxes_xyxy.append(xyxy)
            # Torchvision detection reserves label 0 for background during training.
            labels_out.append(1 if self.class_agnostic else int(cls) + 1)
        boxes = np.array(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        img, boxes = letterbox_image_and_boxes(img, boxes, self.input_size)

        img_arr = np.asarray(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_arr).permute(2, 0, 1)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels_out, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.as_tensor((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=torch.float32) if len(boxes) else torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "path": str(img_path),
        }
        return tensor, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def make_loader(yolo_root: Path, split: str, input_size: int, num_classes: int, batch: int, workers: int, shuffle: bool, class_agnostic: bool = False) -> DataLoader:
    ds = YoloDetectionDataset(yolo_root, split, input_size, num_classes, class_agnostic=class_agnostic)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers, pin_memory=True, collate_fn=collate_fn)


# -----------------------------
# Model builders
# -----------------------------


def build_faster_rcnn(num_classes: int):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.0)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Stabilize image transform dimensions; dataset already letterboxes.
    model.transform.min_size = (640,)
    model.transform.max_size = 640
    return model


def build_ssdlite(num_classes: int):
    import functools
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights, score_thresh=0.0)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
    return model


def set_torchvision_trainable(model: nn.Module, architecture: str, stage: str) -> None:
    """stage=head: freeze backbone; stage=all: unfreeze all."""
    for p in model.parameters():
        p.requires_grad = True
    if stage == "all":
        return
    if architecture == "frcnn":
        for name, p in model.named_parameters():
            if name.startswith("backbone"):
                p.requires_grad = False
            else:
                p.requires_grad = True
    elif architecture == "ssd":
        for name, p in model.named_parameters():
            if name.startswith("backbone"):
                p.requires_grad = False
            else:
                p.requires_grad = True
    else:
        raise ValueError(architecture)


def trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


# -----------------------------
# Metrics and evaluation
# -----------------------------


def convert_targets_for_map(targets: List[dict], class_agnostic: bool = False) -> List[dict]:
    out = []
    for t in targets:
        labels = t["labels"].detach().cpu().clone()
        if class_agnostic:
            labels = torch.zeros_like(labels)
        else:
            labels = labels - 1  # shift torch labels 1..K to metric labels 0..K-1
        out.append({"boxes": t["boxes"].detach().cpu(), "labels": labels})
    return out


def convert_preds_for_map(preds: List[dict], conf: float, class_agnostic: bool = False, is_torchvision: bool = True) -> List[dict]:
    out = []
    for p in preds:
        boxes = p.get("boxes", torch.zeros((0, 4))).detach().cpu()
        scores = p.get("scores", torch.zeros((0,))).detach().cpu()
        labels = p.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu().long()
        keep = scores >= conf
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if class_agnostic:
            labels = torch.zeros((len(labels),), dtype=torch.int64)
        elif is_torchvision:
            labels = labels - 1
        out.append({"boxes": boxes, "scores": scores, "labels": labels})
    return out


def greedy_prf(preds: List[dict], targets: List[dict], iou_thr: float = 0.5, conf: float = 0.5, class_agnostic: bool = False, is_torchvision: bool = True) -> Tuple[float, float, float, int, int, int]:
    tp = fp = fn = 0
    for pred_raw, target_raw in zip(preds, targets):
        pred = convert_preds_for_map([pred_raw], conf=conf, class_agnostic=class_agnostic, is_torchvision=is_torchvision)[0]
        targ = convert_targets_for_map([target_raw], class_agnostic=class_agnostic)[0]
        p_boxes = pred["boxes"].numpy()
        p_scores = pred["scores"].numpy()
        p_labels = pred["labels"].numpy()
        t_boxes = targ["boxes"].numpy()
        t_labels = targ["labels"].numpy()

        order = np.argsort(-p_scores)
        matched_t = set()
        for pi in order:
            candidates = np.arange(len(t_boxes))
            if not class_agnostic:
                candidates = candidates[t_labels == p_labels[pi]]
            candidates = [int(c) for c in candidates if int(c) not in matched_t]
            if not candidates:
                fp += 1
                continue
            ious = box_iou_np(p_boxes[pi:pi + 1], t_boxes[candidates]).reshape(-1)
            best_j = int(np.argmax(ious)) if len(ious) else -1
            if best_j >= 0 and ious[best_j] >= iou_thr:
                matched_t.add(candidates[best_j])
                tp += 1
            else:
                fp += 1
        fn += len(t_boxes) - len(matched_t)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1, tp, fp, fn


@torch.no_grad()
def evaluate_torchvision_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf: float,
    iou_thr: float,
    class_agnostic: bool = False,
    desc: str = "eval",
) -> dict:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    all_preds_raw: List[dict] = []
    all_targets_raw: List[dict] = []

    for images, targets in tqdm(loader, desc=desc):
        images = [img.to(device, non_blocking=True) for img in images]
        preds = model(images)
        preds_cpu = [{k: v.detach().cpu() for k, v in p.items() if k in {"boxes", "scores", "labels"}} for p in preds]
        targets_cpu = [{k: v.detach().cpu() for k, v in t.items() if k in {"boxes", "labels"}} for t in targets]
        all_preds_raw.extend(preds_cpu)
        all_targets_raw.extend(targets_cpu)
        metric.update(
            convert_preds_for_map(preds_cpu, conf=0.0, class_agnostic=class_agnostic, is_torchvision=True),
            convert_targets_for_map(targets_cpu, class_agnostic=class_agnostic),
        )

    m = metric.compute()
    precision, recall, f1, tp, fp, fn = greedy_prf(
        all_preds_raw, all_targets_raw, iou_thr=iou_thr, conf=conf, class_agnostic=class_agnostic, is_torchvision=True
    )
    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "map": float(m.get("map", torch.tensor(0.0)).item()),
        "map_50": float(m.get("map_50", torch.tensor(0.0)).item()),
        "map_75": float(m.get("map_75", torch.tensor(0.0)).item()),
    }
    if "classes" in m and "map_per_class" in m:
        classes = m["classes"].detach().cpu().numpy().tolist()
        map_pc = m["map_per_class"].detach().cpu().numpy().tolist()
        results["per_class_map"] = {int(c): float(v) for c, v in zip(classes, map_pc) if float(v) >= 0}
    return results


@torch.no_grad()
def measure_torchvision_fps(model: nn.Module, loader: DataLoader, device: torch.device, warmup: int = 20, samples: int = 120) -> Tuple[float, float]:
    model.eval()
    images_flat: List[torch.Tensor] = []
    for imgs, _ in loader:
        for img in imgs:
            images_flat.append(img.to(device))
            if len(images_flat) >= max(warmup + samples, 1):
                break
        if len(images_flat) >= warmup + samples:
            break
    if not images_flat:
        return 0.0, 0.0

    for img in images_flat[:warmup]:
        _ = model([img])
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    for img in images_flat[warmup:warmup + samples]:
        _ = model([img])
        count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = count / max(dt, 1e-9)
    ms = 1000.0 / max(fps, 1e-9)
    return fps, ms


def metrics_row(model_name: str, split: str, metrics: dict, fps: float = math.nan, ms_frame: float = math.nan) -> dict:
    return {
        "model": model_name,
        "split": split,
        "precision_%": format_pct(metrics["precision"]),
        "recall_%": format_pct(metrics["recall"]),
        "f1_%": format_pct(metrics["f1"]),
        "mAP@0.5_%": format_pct(metrics["map_50"]),
        "mAP@0.5:0.95_%": format_pct(metrics["map"]),
        "FPS": round(float(fps), 3) if np.isfinite(fps) else "",
        "ms/frame": round(float(ms_frame), 3) if np.isfinite(ms_frame) else "",
        "TP": int(metrics.get("tp", 0)),
        "FP": int(metrics.get("fp", 0)),
        "FN": int(metrics.get("fn", 0)),
    }


# -----------------------------
# Training loops
# -----------------------------


def train_torchvision_detector(
    architecture: str,
    data_root: Path,
    out_dir: Path,
    cfg: dict,
    device: torch.device,
) -> Tuple[nn.Module, dict, float, float]:
    train_cfg = cfg.get("training", {})
    thr = cfg.get("thresholds", {})
    class_names = cfg.get("class_names") or DEFAULT_CLASSES
    num_classes_with_bg = len(class_names) + 1
    workers = int(train_cfg.get("workers", 8))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if architecture == "frcnn":
        input_size = int(train_cfg.get("imgsz_frcnn", 640))
        batch = int(train_cfg.get("batch_frcnn", 8))
        model = build_faster_rcnn(num_classes_with_bg)
        model_name = "Faster R-CNN"
    elif architecture == "ssd":
        input_size = int(train_cfg.get("imgsz_ssd", 320))
        batch = int(train_cfg.get("batch_ssd", 16))
        model = build_ssdlite(num_classes_with_bg)
        model_name = "MobileNet SSD"
    else:
        raise ValueError(architecture)

    model.to(device)
    train_loader = make_loader(data_root, "train", input_size, len(class_names), batch, workers, shuffle=True)
    val_loader = make_loader(data_root, "val", input_size, len(class_names), batch, workers, shuffle=False)
    test_loader = make_loader(data_root, "test", input_size, len(class_names), batch, workers, shuffle=False)
    fps_loader = make_loader(data_root, "test", input_size, len(class_names), 1, workers, shuffle=False)

    epochs_head = int(train_cfg.get("epochs_head", 10))
    epochs_ft = int(train_cfg.get("epochs_finetune", 100))
    if bool(cfg.get("run", {}).get("quick_debug", False)):
        epochs_head = min(1, epochs_head)
        epochs_ft = min(1, epochs_ft)
    lr = float(train_cfg.get("lr_torch", 0.001))
    out_model_dir = out_dir / "torchvision" / architecture
    out_model_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_model_dir / "best.pt"
    last_path = out_model_dir / "last.pt"
    history = []
    best_map50 = -1.0
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    stages = [("head", epochs_head), ("all", epochs_ft)]
    global_epoch = 0
    for stage_name, epochs in stages:
        set_torchvision_trainable(model, architecture, stage=stage_name)
        optimizer = torch.optim.SGD(trainable_params(model), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        for epoch in range(1, epochs + 1):
            global_epoch += 1
            model.train()
            loss_sum = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f"{model_name} {stage_name} epoch {epoch}/{epochs}")
            for images, targets in pbar:
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(losses.detach().cpu())
                n_batches += 1
                pbar.set_postfix(loss=loss_sum / max(n_batches, 1))
            scheduler.step()

            val_metrics = evaluate_torchvision_model(
                model, val_loader, device, conf=float(thr.get("conf_torch", 0.5)),
                iou_thr=float(thr.get("iou_match", 0.5)), class_agnostic=False,
                desc=f"{model_name} val"
            )
            row = {"epoch": global_epoch, "stage": stage_name, "loss": loss_sum / max(n_batches, 1), **val_metrics}
            history.append(row)
            pd.DataFrame(history).to_csv(out_model_dir / "history.csv", index=False)
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": global_epoch, "val_metrics": val_metrics}, last_path)
            if val_metrics["map_50"] > best_map50:
                best_map50 = val_metrics["map_50"]
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": global_epoch, "val_metrics": val_metrics}, best_path)
                print(f"Saved new best {model_name}: val mAP@0.5={best_map50:.4f}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_torchvision_model(
        model, test_loader, device, conf=float(thr.get("conf_torch", 0.5)),
        iou_thr=float(thr.get("iou_match", 0.5)), class_agnostic=False,
        desc=f"{model_name} TEST"
    )
    fps, ms = measure_torchvision_fps(model, fps_loader, device)
    return model, test_metrics, fps, ms


def train_yolo_detector(paths: PreparedPaths, out_dir: Path, cfg: dict) -> Tuple[object, dict, float, float]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics is not installed. Run: pip install ultralytics") from e

    train_cfg = cfg.get("training", {})
    seed = int(cfg.get("seed", 42))
    workers = int(train_cfg.get("workers", 8))
    epochs_head = int(train_cfg.get("epochs_head", 10))
    epochs_ft = int(train_cfg.get("epochs_finetune", 100))
    if bool(cfg.get("run", {}).get("quick_debug", False)):
        epochs_head = min(1, epochs_head)
        epochs_ft = min(1, epochs_ft)
    imgsz = int(train_cfg.get("imgsz_yolo", 640))
    batch = int(train_cfg.get("batch_yolo", 16))
    lr0 = float(train_cfg.get("lr_yolo", 0.01))
    project = out_dir / "yolo"
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8s.pt")
    # Stage 1: freeze early layers. Ultralytics freeze accepts an integer count of layers.
    print(f"[{now()}] YOLOv8s Stage 1: frozen adaptation for {epochs_head} epochs")
    res1 = model.train(
        data=str(paths.trash_yaml), epochs=epochs_head, imgsz=imgsz, batch=batch,
        project=str(project), name="yolov8s_stage1", exist_ok=True, seed=seed,
        deterministic=True, workers=workers, optimizer="AdamW", lr0=lr0,
        freeze=10, amp=bool(train_cfg.get("amp", True)), cos_lr=True,
    )
    stage1_best = Path(res1.save_dir) / "weights" / "best.pt"
    if not stage1_best.exists():
        stage1_best = Path(res1.save_dir) / "weights" / "last.pt"

    print(f"[{now()}] YOLOv8s Stage 2: full fine-tuning for {epochs_ft} epochs")
    model = YOLO(str(stage1_best))
    res2 = model.train(
        data=str(paths.trash_yaml), epochs=epochs_ft, imgsz=imgsz, batch=batch,
        project=str(project), name="yolov8s_stage2", exist_ok=True, seed=seed,
        deterministic=True, workers=workers, optimizer="AdamW", lr0=lr0,
        freeze=0, amp=bool(train_cfg.get("amp", True)), cos_lr=True,
        close_mosaic=10, patience=30,
    )
    best = Path(res2.save_dir) / "weights" / "best.pt"
    if not best.exists():
        best = Path(res2.save_dir) / "weights" / "last.pt"
    yolo_model = YOLO(str(best))

    metrics, fps, ms = evaluate_yolo_model(
        yolo_model, paths.trash_dir, "test", img_size=imgsz,
        conf=float(cfg.get("thresholds", {}).get("conf_yolo", 0.25)),
        iou_thr=float(cfg.get("thresholds", {}).get("iou_match", 0.5)),
        class_agnostic=False,
        desc="YOLOv8s TEST",
    )
    return yolo_model, metrics, fps, ms


@torch.no_grad()
def evaluate_yolo_model(
    yolo_model: object,
    yolo_root: Path,
    split: str,
    img_size: int,
    conf: float,
    iou_thr: float,
    class_agnostic: bool,
    desc: str,
) -> Tuple[dict, float, float]:
    # We use the same dataset class for consistent target geometry.
    num_classes = 1 if class_agnostic else 3
    ds = YoloDetectionDataset(yolo_root, split, img_size, num_classes=num_classes, class_agnostic=class_agnostic)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    preds_raw: List[dict] = []
    targets_raw: List[dict] = []

    t0 = None
    n_timed = 0
    warmup = min(20, len(ds))
    for i in tqdm(range(len(ds)), desc=desc):
        img_tensor, target = ds[i]
        # yolo_model.predict accepts numpy images, but target boxes are in letterboxed coords.
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        if i == warmup:
            t0 = time.perf_counter()
            n_timed = 0
        result = yolo_model.predict(img_np, imgsz=img_size, conf=0.001, verbose=False)[0]
        if t0 is not None:
            n_timed += 1
        if result.boxes is None or len(result.boxes) == 0:
            pred = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            pred = {
                "boxes": result.boxes.xyxy.detach().cpu().float(),
                "scores": result.boxes.conf.detach().cpu().float(),
                "labels": result.boxes.cls.detach().cpu().long(),
            }
            if class_agnostic:
                pred["labels"] = torch.zeros_like(pred["labels"])
        targ = {"boxes": target["boxes"].detach().cpu(), "labels": target["labels"].detach().cpu()}
        preds_raw.append(pred)
        targets_raw.append(targ)
        metric.update(
            convert_preds_for_map([pred], conf=0.0, class_agnostic=class_agnostic, is_torchvision=False),
            convert_targets_for_map([targ], class_agnostic=class_agnostic),
        )

    dt = time.perf_counter() - t0 if t0 is not None else 0.0
    fps = n_timed / max(dt, 1e-9) if n_timed else 0.0
    ms = 1000.0 / max(fps, 1e-9) if fps else 0.0
    m = metric.compute()
    precision, recall, f1, tp, fp, fn = greedy_prf(
        preds_raw, targets_raw, iou_thr=iou_thr, conf=conf, class_agnostic=class_agnostic, is_torchvision=False
    )
    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "map": float(m.get("map", torch.tensor(0.0)).item()),
        "map_50": float(m.get("map_50", torch.tensor(0.0)).item()),
        "map_75": float(m.get("map_75", torch.tensor(0.0)).item()),
    }
    if "classes" in m and "map_per_class" in m:
        classes = m["classes"].detach().cpu().numpy().tolist()
        map_pc = m["map_per_class"].detach().cpu().numpy().tolist()
        results["per_class_map"] = {int(c): float(v) for c, v in zip(classes, map_pc) if float(v) >= 0}
    return results, fps, ms


# -----------------------------
# Cross-domain and reporting
# -----------------------------


def evaluate_cross_domain(
    yolo_model: Optional[object],
    frcnn_model: Optional[nn.Module],
    ssd_model: Optional[nn.Module],
    paths: PreparedPaths,
    cfg: dict,
    device: torch.device,
    out_dir: Path,
) -> pd.DataFrame:
    if paths.river_dir is None:
        return pd.DataFrame()
    rows = []
    thr = cfg.get("thresholds", {})
    train_cfg = cfg.get("training", {})
    workers = int(train_cfg.get("workers", 8))
    iou_thr = float(thr.get("iou_match", 0.5))

    if yolo_model is not None:
        m, fps, ms = evaluate_yolo_model(
            yolo_model, paths.river_dir, "test", int(train_cfg.get("imgsz_yolo", 640)),
            conf=float(thr.get("conf_yolo", 0.25)), iou_thr=iou_thr,
            class_agnostic=True, desc="YOLOv8s River class-agnostic"
        )
        rows.append(metrics_row("YOLOv8s", "River cross-domain", m, fps, ms))

    if frcnn_model is not None:
        loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_frcnn", 640)), 1, 1, workers, False, class_agnostic=True)
        m = evaluate_torchvision_model(
            frcnn_model, loader, device, conf=float(thr.get("conf_torch", 0.5)), iou_thr=iou_thr,
            class_agnostic=True, desc="Faster R-CNN River class-agnostic"
        )
        fps_loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_frcnn", 640)), 1, 1, workers, False, class_agnostic=True)
        fps, ms = measure_torchvision_fps(frcnn_model, fps_loader, device)
        rows.append(metrics_row("Faster R-CNN", "River cross-domain", m, fps, ms))

    if ssd_model is not None:
        loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_ssd", 320)), 1, 1, workers, False, class_agnostic=True)
        m = evaluate_torchvision_model(
            ssd_model, loader, device, conf=float(thr.get("conf_torch", 0.5)), iou_thr=iou_thr,
            class_agnostic=True, desc="MobileNet SSD River class-agnostic"
        )
        fps_loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_ssd", 320)), 1, 1, workers, False, class_agnostic=True)
        fps, ms = measure_torchvision_fps(ssd_model, fps_loader, device)
        rows.append(metrics_row("MobileNet SSD", "River cross-domain", m, fps, ms))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_cross_domain.csv", index=False)
    return df


def save_per_class(results_by_model: Dict[str, dict], class_names: List[str], out_dir: Path) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results_by_model.items():
        pc = metrics.get("per_class_map", {}) or {}
        row = {"model": model_name}
        for i, name in enumerate(class_names):
            row[f"{name}_mAP"] = format_pct(pc.get(i, float("nan"))) if i in pc else ""
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "results_per_class_map.csv", index=False)
    return df


def plot_results(overall_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        if overall_df.empty or "FPS" not in overall_df.columns:
            return
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, row in overall_df.iterrows():
            ax.scatter(float(row["FPS"]), float(row["mAP@0.5_%"]), s=90)
            ax.annotate(str(row["model"]), (float(row["FPS"]), float(row["mAP@0.5_%"])), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("FPS")
        ax.set_ylabel("mAP@0.5 (%)")
        ax.set_title("Speed-Accuracy Trade-off on Trash-ICRA19 Test Set")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_speed_accuracy.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Could not plot results: {e}")


def write_markdown_report(out_dir: Path, dataset_summary: Path, overall: pd.DataFrame, per_class: pd.DataFrame, cross: pd.DataFrame, cfg: dict) -> None:
    md = []
    md.append("# Marine Debris 3-Model Comparison — Results for IEEE Paper\n")
    md.append(f"Generated: {now()}\n")
    md.append("## Dataset summary\n")
    try:
        ds = pd.read_csv(dataset_summary)
        md.append(ds.to_markdown(index=False))
    except Exception:
        md.append("Dataset summary unavailable.")
    md.append("\n\n## Overall in-domain test results\n")
    md.append(overall.to_markdown(index=False) if not overall.empty else "No results.")
    md.append("\n\n## Per-class mAP\n")
    md.append(per_class.to_markdown(index=False) if not per_class.empty else "No per-class results.")
    md.append("\n\n## Cross-domain river results\n")
    md.append(cross.to_markdown(index=False) if not cross.empty else "Cross-domain results unavailable or skipped.")
    md.append("\n\n## Important paper wording\n")
    md.append(
        "- Report these results as **test-set results** only if this script completed evaluation on the `test` split.\n"
        "- Cross-domain river evaluation here is **class-agnostic IoU matching**: all river labels are remapped to one `trash` class and model predictions are evaluated as object localization, not class-name compatibility.\n"
        "- If YOLO augmentation remains enabled while Faster R-CNN and SSD use only standard preprocessing, mention augmentation asymmetry as a limitation.\n"
        "- Do not copy old notebook numbers if they differ from these CSV files. Update the paper tables from this folder.\n"
    )
    (out_dir / "results_for_paper.md").write_text("\n".join(md), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def load_config(args) -> dict:
    if args.config:
        cfg = read_yaml(Path(args.config))
    else:
        cfg = {}
    # CLI overrides
    if args.trash_root:
        cfg["trash_root"] = args.trash_root
    if args.river_root:
        cfg["river_root"] = args.river_root
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.quick_debug:
        cfg.setdefault("run", {})["quick_debug"] = True
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled marine debris detection comparison on RunPod.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--trash-root", type=str, default=None, help="Override Trash-ICRA19 root path")
    parser.add_argument("--river-root", type=str, default=None, help="Override River Floating Trash root path")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--quick-debug", action="store_true", help="Run a tiny 1-epoch test for debugging")
    args = parser.parse_args()

    cfg = load_config(args)
    required = ["trash_root", "out_dir"]
    missing = [k for k in required if k not in cfg or not cfg[k]]
    if missing:
        raise RuntimeError(f"Missing config keys: {missing}. Edit config.yaml or pass CLI paths.")

    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(out_dir / "used_config.yaml", cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    print(f"[{now()}] Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    run_cfg = cfg.get("run", {})
    if bool(run_cfg.get("prepare_data", True)):
        paths = prepare_datasets(cfg)
    else:
        data_out = out_dir / "prepared_datasets"
        paths = PreparedPaths(
            trash_yaml=data_out / "trash_icra19_clean" / "data.yaml",
            river_yaml=data_out / "river_trash_class_agnostic" / "data.yaml",
            trash_dir=data_out / "trash_icra19_clean",
            river_dir=(data_out / "river_trash_class_agnostic") if (data_out / "river_trash_class_agnostic").exists() else None,
            summary_csv=out_dir / "dataset_summary.csv",
        )

    yolo_model = frcnn_model = ssd_model = None
    results_by_model: Dict[str, dict] = {}
    overall_rows = []

    if bool(run_cfg.get("train_yolo", True)):
        yolo_model, metrics, fps, ms = train_yolo_detector(paths, out_dir, cfg)
        results_by_model["YOLOv8s"] = metrics
        overall_rows.append(metrics_row("YOLOv8s", "Trash-ICRA19 test", metrics, fps, ms))

    if bool(run_cfg.get("train_frcnn", True)):
        frcnn_model, metrics, fps, ms = train_torchvision_detector("frcnn", paths.trash_dir, out_dir, cfg, device)
        results_by_model["Faster R-CNN"] = metrics
        overall_rows.append(metrics_row("Faster R-CNN", "Trash-ICRA19 test", metrics, fps, ms))

    if bool(run_cfg.get("train_ssd", True)):
        ssd_model, metrics, fps, ms = train_torchvision_detector("ssd", paths.trash_dir, out_dir, cfg, device)
        results_by_model["MobileNet SSD"] = metrics
        overall_rows.append(metrics_row("MobileNet SSD", "Trash-ICRA19 test", metrics, fps, ms))

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_dir / "results_overall_test.csv", index=False)
    per_class_df = save_per_class(results_by_model, list(cfg.get("class_names") or DEFAULT_CLASSES), out_dir)
    cross_df = evaluate_cross_domain(yolo_model, frcnn_model, ssd_model, paths, cfg, device, out_dir)
    plot_results(overall_df, out_dir)
    write_markdown_report(out_dir, paths.summary_csv, overall_df, per_class_df, cross_df, cfg)

    print("\nFinished. Main outputs:")
    for p in [
        out_dir / "dataset_summary.csv",
        out_dir / "results_overall_test.csv",
        out_dir / "results_per_class_map.csv",
        out_dir / "results_cross_domain.csv",
        out_dir / "results_for_paper.md",
        out_dir / "fig_speed_accuracy.png",
    ]:
        if p.exists():
            print(f" - {p}")


if __name__ == "__main__":
    main()
