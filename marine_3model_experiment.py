#!/usr/bin/env python3
"""
RunPod-ready controlled comparison for the paper:
YOLOv8s vs Faster R-CNN ResNet-50-FPN vs MobileNet SSD/SSDLite320 MobileNetV3-Large
on Trash-ICRA19, plus class-agnostic cross-domain testing on River Floating Trash.

Main design choices:
- One deterministic, leakage-controlled sequence-level split for all models.
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
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
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


def checkpoint_identity(cfg: dict, architecture: str) -> dict:
    """Return immutable provenance written into every TorchVision checkpoint."""
    provenance = cfg.get("provenance", {})
    config_bytes = canonical_training_config_bytes(cfg)
    return {
        "architecture": architecture,
        "seed": int(cfg.get("seed", 42)),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_sha256": provenance.get("dataset_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "git_commit": os.environ.get("CANONICAL_GIT_COMMIT", "unknown"),
        "experiment_id": os.environ.get("CANONICAL_EXPERIMENT_ID", "unknown"),
    }


def canonical_training_config_bytes(cfg: dict) -> bytes:
    """Hash scientific settings while excluding per-session operational controls."""
    stable = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    for key in ("trash_root", "river_root", "out_dir"):
        stable.pop(key, None)
    stable.get("training", {}).pop("resume", None)
    stable.get("run", {}).pop("quick_debug", None)
    stable.pop("session_control", None)
    return json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")


def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(canonical_training_config_bytes(cfg)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def session_control_sha256(cfg: dict) -> str:
    control = cfg.get("session_control", {})
    return hashlib.sha256(json.dumps(control, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def capture_rng_state() -> dict:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_states"):
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_states"])


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


def dataframe_markdown(frame: pd.DataFrame) -> str:
    """Render a table without making report generation depend on tabulate."""
    try:
        return frame.to_markdown(index=False)
    except ImportError:
        return "```csv\n" + frame.to_csv(index=False).strip() + "\n```"


def t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value without a SciPy dependency."""
    values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
              6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
              11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
              16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
              25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}
    if df in values:
        return values[df]
    larger = [key for key in values if key >= df]
    return values[min(larger)] if larger else 1.960


def image_difficulty_flags(image: Image.Image, boxes: np.ndarray) -> dict:
    """Return reproducible visual proxies for common marine-debris challenges.

    Illumination and turbidity cannot be diagnosed from RGB pixels alone, so the
    latter is explicitly reported as a low-contrast proxy rather than ground truth.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    h, w = gray.shape
    image_area = max(float(h * w), 1.0)
    areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / image_area if len(boxes) else np.array([])
    small_object = bool(len(areas) and np.any(areas < 0.01))
    overlap = False
    if len(boxes) > 1:
        pairwise = box_iou_np(boxes, boxes)
        np.fill_diagonal(pairwise, 0.0)
        overlap = bool(np.max(pairwise) >= 0.10)
    mean_luma = float(gray.mean())
    contrast = float(gray.std())
    return {
        "small_object": small_object,
        "overlap_occlusion_proxy": overlap,
        "low_light": mean_luma < 0.25,
        "overexposed": mean_luma > 0.80,
        "low_contrast_turbidity_proxy": contrast < 0.10,
        "mean_luminance": round(mean_luma, 5),
        "contrast_std": round(contrast, 5),
        "min_object_area_ratio": round(float(areas.min()), 6) if len(areas) else math.nan,
    }


def confidence_threshold(cfg: dict) -> float:
    """One operating point for Precision/Recall/F1 across every architecture."""
    thresholds = cfg.get("thresholds", {})
    if "confidence" in thresholds:
        return float(thresholds["confidence"])
    legacy = [thresholds[k] for k in ("conf_yolo", "conf_torch") if k in thresholds]
    if legacy:
        warnings.warn("conf_yolo/conf_torch are deprecated; use thresholds.confidence.")
        if len(legacy) == 2 and float(legacy[0]) != float(legacy[1]):
            warnings.warn("Legacy confidence thresholds differ; using conf_yolo for all models.")
        return float(legacy[0])
    return 0.25


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weights_size_mb(model: nn.Module) -> float:
    """In-memory tensor size of a model state_dict, excluding optimizer metadata."""
    total_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
    return total_bytes / (1024 ** 2)


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


def sequence_id(record: YoloRecord, cfg: dict) -> str:
    """Infer a capture sequence without requiring dataset-specific annotations."""
    leakage = cfg.get("leakage", {})
    pattern = str(leakage.get("sequence_regex", "")).strip()
    relative = safe_rel(record.image, Path(cfg["trash_root"]).expanduser().resolve())
    if pattern:
        # Trash-ICRA19's sequence tag is encoded in the filename prefix. Matching
        # only the basename avoids accidental captures from parent directories.
        match = re.search(pattern, record.image.name)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        if bool(leakage.get("strict_sequence_regex", True)):
            raise RuntimeError(
                f"sequence_regex={pattern!r} did not match Trash-ICRA19 file {relative!r}. "
                "Refusing an image-level fallback because it could introduce frame leakage."
            )
    if bool(leakage.get("group_by_parent", True)):
        return str(record.image.parent.resolve())
    return str(record.image.resolve())


def deduplicate_records(records: List[YoloRecord], report_path: Path) -> List[YoloRecord]:
    """Remove exact duplicate image bytes before splitting and export an audit trail."""
    seen: Dict[str, YoloRecord] = {}
    rows = []
    kept = []
    for record in tqdm(records, desc="Exact-image deduplication"):
        digest = file_sha256(record.image)
        if digest in seen:
            rows.append({"removed_image": str(record.image), "kept_image": str(seen[digest].image), "sha256": digest})
        else:
            seen[digest] = record
            kept.append(record)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["removed_image", "kept_image", "sha256"]).to_csv(report_path, index=False)
    return kept


def exclude_conflicting_duplicate_groups(records: List[YoloRecord], report_path: Path) -> List[YoloRecord]:
    """Exclude every sample in an exact-image group whose label files disagree."""
    groups: Dict[str, List[YoloRecord]] = defaultdict(list)
    for record in tqdm(records, desc="Auditing River duplicate groups"):
        groups[file_sha256(record.image)].append(record)
    conflicting = {
        digest for digest, members in groups.items()
        if len(members) > 1 and len({file_sha256(member.label) for member in members}) > 1
    }
    rows = [
        {
            "excluded_image": str(record.image),
            "label": str(record.label),
            "image_sha256": digest,
            "reason": "conflicting_labels_for_exact_duplicate_image",
        }
        for digest in sorted(conflicting)
        for record in groups[digest]
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows, columns=["excluded_image", "label", "image_sha256", "reason"]
    ).to_csv(report_path, index=False)
    return [record for digest, members in groups.items() if digest not in conflicting for record in members]


def grouped_split(records: List[YoloRecord], cfg: dict, seed: int) -> Dict[str, List[YoloRecord]]:
    """Frame-balanced 70/15/15 split with every sequence in one partition."""
    groups = [sequence_id(record, cfg) for record in records]
    if len(set(groups)) < 3:
        raise RuntimeError(
            "Fewer than three capture sequences were detected. Refusing an image-level "
            "fallback because it could introduce frame leakage. Check sequence_regex."
        )
    sizes = Counter(groups)
    rng = random.Random(seed)
    ordered = sorted(sizes, key=lambda group: (-sizes[group], rng.random(), group))
    targets = {"train": .70 * len(records), "val": .15 * len(records), "test": .15 * len(records)}
    loads = {name: 0 for name in targets}
    owners = {}
    for group in ordered:
        split = min(targets, key=lambda name: (loads[name] / max(targets[name], 1), loads[name]))
        owners[group] = split
        loads[split] += sizes[group]
    result = {"train": [], "val": [], "test": []}
    for record, group in zip(records, groups):
        result[owners[group]].append(record)
    if any(not records for records in result.values()):
        raise RuntimeError(f"Sequence split produced an empty partition: {loads}")
    return result


def save_split_manifests(splits: Dict[str, List[YoloRecord]], cfg: dict, out_dir: Path) -> None:
    manifest_dir = out_dir / "splits"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        sequences = sorted({sequence_id(record, cfg) for record in records})
        images = sorted(str(record.image.resolve()) for record in records)
        (manifest_dir / f"{split}_sequences.txt").write_text("\n".join(sequences) + "\n", encoding="utf-8")
        (manifest_dir / f"{split}_images.txt").write_text("\n".join(images) + "\n", encoding="utf-8")


def save_sequence_split_audit(splits: Dict[str, List[YoloRecord]], cfg: dict, out_path: Path) -> pd.DataFrame:
    """Export auditable sequence ownership and verify that every sequence has one split."""
    rows = []
    for split, records in splits.items():
        counts = Counter(sequence_id(record, cfg) for record in records)
        for sequence, frames in sorted(counts.items()):
            rows.append({"sequence": sequence, "split": split, "frames": int(frames)})
    audit = pd.DataFrame(rows, columns=["sequence", "split", "frames"])
    duplicated = audit.groupby("sequence")["split"].nunique() if not audit.empty else pd.Series(dtype=int)
    leaking = duplicated[duplicated > 1]
    if not leaking.empty:
        raise RuntimeError(f"Sequence leakage audit failed for: {leaking.index.tolist()}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_path, index=False)
    return audit


def save_sequence_regex_audit(records: List[YoloRecord], cfg: dict, out_path: Path) -> pd.DataFrame:
    """Check every Trash image against sequence_regex and export the result."""
    pattern = str(cfg.get("leakage", {}).get("sequence_regex", "")).strip()
    if not pattern:
        raise RuntimeError("leakage.sequence_regex is empty; there is nothing to validate.")
    root = Path(cfg["trash_root"]).expanduser().resolve()
    rows = []
    for record in records:
        relative = safe_rel(record.image, root)
        match = re.search(pattern, record.image.name)
        rows.append({
            "image": relative,
            "matched": bool(match),
            "sequence": (match.group(1) if match and match.groups() else match.group(0)) if match else "",
        })
    audit = pd.DataFrame(rows, columns=["image", "matched", "sequence"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_path, index=False)
    unmatched = int((~audit["matched"]).sum()) if not audit.empty else 0
    if unmatched:
        raise RuntimeError(
            f"sequence_regex failed for {unmatched}/{len(audit)} images. See {out_path} for exact filenames."
        )
    return audit


def validate_no_split_leakage(splits: Dict[str, List[YoloRecord]], cfg: dict) -> None:
    owners: Dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            group = sequence_id(record, cfg)
            if group in owners and owners[group] != split:
                raise RuntimeError(f"Sequence leakage detected: {group!r} occurs in {owners[group]} and {split}.")
            owners[group] = split


def validate_prepared_sequence_split(data_root: Path, out_dir: Path, cfg: dict) -> None:
    """Refuse an undocumented or leaking prepared dataset in sequence-split mode."""
    required = [out_dir / "sequence_split_audit.csv"]
    required += [out_dir / "splits" / f"{split}_{kind}.txt"
                 for split in ("train", "val", "test") for kind in ("sequences", "images")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "prepare_data=false cannot reuse this dataset as sequence-safe because its audit/manifests "
            f"are missing: {missing}. Run once with prepare_data=true."
        )
    pattern = re.compile(str(cfg.get("leakage", {}).get("sequence_regex", "")))
    owners: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        images = [p for p in (data_root / "images" / split).glob("*") if p.suffix.lower() in IMG_EXTS]
        if not images:
            raise RuntimeError(f"Prepared sequence split has no images in {split!r}.")
        for image in images:
            match = pattern.search(image.name)
            if not match:
                raise RuntimeError(f"Prepared image does not expose a sequence ID: {image}")
            sequence = match.group(1) if match.groups() else match.group(0)
            previous = owners.setdefault(sequence, split)
            if previous != split:
                raise RuntimeError(f"Prepared-data sequence leakage: {sequence!r} is in {previous} and {split}.")


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
            stable_id = hashlib.sha256(str(r.image.resolve()).encode("utf-8")).hexdigest()[:10]
            safe_name = f"{r.image.stem}_{stable_id}{suffix}"
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


def interpretable_image_features(path: Path) -> dict:
    """Low-cost domain descriptors used to quantify dataset dependence/shift."""
    with Image.open(path) as source:
        rgb = np.asarray(source.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    return {
        "luminance": float(gray.mean()), "contrast": float(gray.std()),
        "saturation": float(saturation), "edge_strength": float((dx + dy) / 2.0),
    }


def save_domain_feature_shift(trash_df: pd.DataFrame, river_df: pd.DataFrame, out_dir: Path, seed: int, max_images: int = 500) -> None:
    """Quantify interpretable input-domain shift with standardized mean differences."""
    rng = np.random.default_rng(seed)
    domains = {}
    for name, frame in (("Trash-ICRA19 test", trash_df[trash_df["split"] == "test"]), ("River test", river_df)):
        paths = frame["source_image"].dropna().astype(str).tolist()
        if len(paths) > max_images:
            paths = [paths[i] for i in sorted(rng.choice(len(paths), max_images, replace=False))]
        values = []
        for path in tqdm(paths, desc=f"Domain features: {name}"):
            try:
                values.append(interpretable_image_features(Path(path)))
            except Exception:
                continue
        domains[name] = pd.DataFrame(values)
    source, target = domains["Trash-ICRA19 test"], domains["River test"]
    rows = []
    for feature in ("luminance", "contrast", "saturation", "edge_strength"):
        source_mean, target_mean = float(source[feature].mean()), float(target[feature].mean())
        source_sd, target_sd = float(source[feature].std(ddof=1)), float(target[feature].std(ddof=1))
        pooled = math.sqrt(max((source_sd ** 2 + target_sd ** 2) / 2.0, 1e-12))
        rows.append({
            "feature": feature, "source_n": len(source), "target_n": len(target),
            "source_mean": source_mean, "source_sd": source_sd,
            "target_mean": target_mean, "target_sd": target_sd,
            "standardized_mean_difference": (target_mean - source_mean) / pooled,
        })
    pd.DataFrame(rows).to_csv(out_dir / "domain_feature_shift.csv", index=False)


def prepare_datasets(cfg: dict) -> PreparedPaths:
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    data_out = Path("/kaggle/temp/marine_prepared") if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else out_dir / "prepared_datasets"
    data_out.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg.get("class_names") or DEFAULT_CLASSES)
    trash_root = Path(cfg["trash_root"]).expanduser().resolve()
    river_root = Path(cfg.get("river_root", "")).expanduser().resolve() if cfg.get("river_root") else None
    seed = int(cfg.get("split_seed", cfg.get("seed", 42)))
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
    if bool(cfg.get("leakage", {}).get("deduplicate", True)):
        trash_records = deduplicate_records(trash_records, out_dir / "deduplication_report.csv")
    if skipped_bad:
        print(f"Skipped {skipped_bad} unreadable images.")

    if split_mode == "official":
        trash_splits = official_split(trash_records)
    elif split_mode == "stratified_70_15_15":
        trash_splits = stratified_split(trash_records, dom, seed)
    elif split_mode == "sequence_70_15_15":
        trash_splits = grouped_split(trash_records, cfg, seed)
        validate_no_split_leakage(trash_splits, cfg)
        save_sequence_split_audit(trash_splits, cfg, out_dir / "sequence_split_audit.csv")
        save_split_manifests(trash_splits, cfg, out_dir)
    else:
        raise ValueError("split_mode must be 'official', 'stratified_70_15_15', or 'sequence_70_15_15'")

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
            duplicate_policy = cfg.get("approval", {}).get("river_duplicate_policy")
            if duplicate_policy != "exclude_all_conflicting_groups":
                raise RuntimeError(f"Unsupported or unapproved River duplicate policy: {duplicate_policy!r}")
            river_records = exclude_conflicting_duplicate_groups(
                river_records, out_dir / "river_excluded_conflicting_duplicates.csv"
            )
            # River is used only for cross-domain evaluation; put all records in test.
            if bool(cfg.get("run", {}).get("quick_debug", False)):
                river_records = river_records[:64]
            river_dst = data_out / "river_trash_class_agnostic"
            river_df = materialize_yolo_dataset({"test": river_records}, river_dst, ["trash"], copy_files, remap_all_to_zero=True)
            validation = cfg.get("data_validation", {})
            min_images = int(validation.get("min_river_images", 1))
            min_objects = int(validation.get("min_river_objects", 1))
            if bool(cfg.get("run", {}).get("quick_debug", False)):
                min_images = min(64, min_images)
                min_objects = 1
            actual_images = len(river_df)
            actual_objects = int(river_df["objects"].sum()) if not river_df.empty else 0
            if actual_images != min_images or actual_objects != min_objects:
                raise RuntimeError(
                    "River dataset validation failed: "
                    f"found {actual_images} images/{actual_objects} objects, expected exactly "
                    f"{min_images} images/{min_objects} objects. Check river_root and YOLO label paths."
                )
            river_yaml = river_dst / "data.yaml"
            save_domain_feature_shift(trash_df, river_df, out_dir, seed)
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
    def __init__(self, yolo_root: Path, split: str, input_size: int, num_classes: int,
                 class_agnostic: bool = False, augment: bool = False, augmentation: Optional[dict] = None):
        self.root = Path(yolo_root)
        self.split = split
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)
        self.class_agnostic = class_agnostic
        self.augment = bool(augment)
        self.augmentation = augmentation or {}
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

        if self.augment:
            flip_prob = float(self.augmentation.get("horizontal_flip", 0.5))
            if random.random() < flip_prob:
                img = ImageOps.mirror(img)
                if len(boxes):
                    old_x1 = boxes[:, 0].copy()
                    boxes[:, 0] = self.input_size - boxes[:, 2]
                    boxes[:, 2] = self.input_size - old_x1
            brightness = float(self.augmentation.get("brightness", 0.0))
            saturation = float(self.augmentation.get("saturation", 0.0))
            if brightness > 0:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(1.0 - brightness, 1.0 + brightness))
            if saturation > 0:
                img = ImageEnhance.Color(img).enhance(random.uniform(1.0 - saturation, 1.0 + saturation))

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


def make_loader(yolo_root: Path, split: str, input_size: int, num_classes: int, batch: int, workers: int,
                shuffle: bool, class_agnostic: bool = False, augment: bool = False,
                augmentation: Optional[dict] = None, drop_last: bool = False,
                generator: Optional[torch.Generator] = None) -> DataLoader:
    ds = YoloDetectionDataset(yolo_root, split, input_size, num_classes, class_agnostic=class_agnostic,
                              augment=augment, augmentation=augmentation)
    return DataLoader(
        ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=True, collate_fn=collate_fn, drop_last=drop_last, generator=generator,
    )


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


def validate_resume_checkpoint(checkpoint: dict, cfg: dict, architecture: str) -> None:
    """Reject an inexact or untraceable training resume before loading model state."""
    required_state = {
        "model", "optimizer", "scheduler", "scaler", "epoch", "completed_epoch", "next_epoch",
        "stage", "stage_epoch", "cfg", "best_map", "best_epoch", "training_history",
        "checkpoint_identity", "training_config_sha256", "dataset_sha256", "split_sha256",
        "git_commit", "seed", "experiment_id", "python_random_state", "numpy_random_state",
        "torch_cpu_rng_state", "torch_cuda_rng_states", "dataloader_generator_state", "sampler_state",
    }
    missing = sorted(required_state - set(checkpoint))
    if missing:
        raise RuntimeError(
            "Checkpoint is not safely resumable; missing state: " + ", ".join(missing)
        )
    recorded_architecture = checkpoint.get("architecture")
    if recorded_architecture != architecture:
        raise RuntimeError(
            f"Checkpoint architecture mismatch: {recorded_architecture!r} != {architecture!r}"
        )
    expected = cfg.get("provenance", {})
    recorded = checkpoint.get("cfg", {}).get("provenance", {})
    for key in ("dataset_sha256", "split_manifest_sha256"):
        if not expected.get(key):
            raise RuntimeError(f"Current config is missing required provenance.{key}")
        if recorded.get(key) != expected[key]:
            raise RuntimeError(
                f"Checkpoint provenance mismatch for {key}: "
                f"{recorded.get(key)!r} != {expected[key]!r}"
            )
    identity = checkpoint["checkpoint_identity"]
    expected_identity = checkpoint_identity(cfg, architecture)
    for key in ("architecture", "seed", "config_sha256", "dataset_sha256", "split_manifest_sha256", "git_commit", "experiment_id"):
        if identity.get(key) != expected_identity.get(key):
            raise RuntimeError(f"Checkpoint identity mismatch for {key}: {identity.get(key)!r} != {expected_identity.get(key)!r}")
    if checkpoint["training_config_sha256"] != config_sha256(cfg):
        raise RuntimeError("Checkpoint training_config_sha256 does not match the current scientific config")


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


def common_class_agnostic_nms(pred: dict, iou_thr: float = 0.5, max_detections: int = 100) -> dict:
    """Apply one architecture-independent NMS after collapsing predicted classes."""
    boxes = pred.get("boxes", torch.zeros((0, 4))).detach().cpu()
    scores = pred.get("scores", torch.zeros((0,))).detach().cpu()
    if len(boxes) == 0:
        return {"boxes": boxes, "scores": scores, "labels": torch.zeros((0,), dtype=torch.int64)}
    order = torch.argsort(scores, descending=True)
    kept: List[int] = []
    while len(order) and len(kept) < int(max_detections):
        current = int(order[0].item())
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        ious = box_iou_np(boxes[current:current + 1].numpy(), boxes[remaining].numpy()).reshape(-1)
        order = remaining[torch.from_numpy(ious <= float(iou_thr))]
    keep = torch.as_tensor(kept, dtype=torch.long)
    return {
        "boxes": boxes[keep], "scores": scores[keep],
        "labels": torch.zeros((len(keep),), dtype=torch.int64),
    }


@contextmanager
def class_agnostic_detection_nms_once(model: Optional[nn.Module] = None, iou_threshold: float = 0.5):
    """Replace TorchVision's final class-aware detection NMS for one forward call."""
    import torchvision.models.detection.roi_heads as roi_heads_module
    import torchvision.models.detection.ssd as ssd_module
    from torchvision.ops import nms

    original_roi = roi_heads_module.box_ops.batched_nms
    original_ssd = ssd_module.box_ops.batched_nms
    threshold_owners = []
    for owner in (getattr(model, "roi_heads", None), model):
        if owner is not None and hasattr(owner, "nms_thresh"):
            threshold_owners.append((owner, owner.nms_thresh))
            owner.nms_thresh = float(iou_threshold)

    def agnostic_nms(boxes, scores, labels, iou_threshold):
        del labels
        return nms(boxes, scores, iou_threshold)

    roi_heads_module.box_ops.batched_nms = agnostic_nms
    ssd_module.box_ops.batched_nms = agnostic_nms
    try:
        yield
    finally:
        roi_heads_module.box_ops.batched_nms = original_roi
        ssd_module.box_ops.batched_nms = original_ssd
        for owner, original_threshold in threshold_owners:
            owner.nms_thresh = original_threshold


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


def image_error_counts(pred_raw: dict, target_raw: dict, conf: float, iou_thr: float,
                       class_agnostic: bool, is_torchvision: bool) -> Tuple[int, int]:
    _, _, _, _, fp, fn = greedy_prf([pred_raw], [target_raw], iou_thr, conf, class_agnostic, is_torchvision)
    return fp, fn


def save_qualitative_errors(images: Sequence[Path], preds: List[dict], targets: List[dict], out_dir: Path,
                            conf: float, iou_thr: float, class_agnostic: bool,
                            is_torchvision: bool, limit: int = 20) -> None:
    """Save the hardest FP/FN examples. Green=ground truth, red=prediction."""
    ranked = []
    for path, pred, target in zip(images, preds, targets):
        fp, fn = image_error_counts(pred, target, conf, iou_thr, class_agnostic, is_torchvision)
        if fp or fn:
            ranked.append((fp + fn, fp, fn, Path(path), pred, target))
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, (_, fp, fn, path, pred_raw, target_raw) in enumerate(sorted(ranked, key=lambda x: x[0], reverse=True)[:limit], 1):
        with Image.open(path) as source:
            image = source.convert("RGB")
        # Evaluation boxes use letterboxed coordinates. Compute all area-based
        # diagnostics on that same canvas so object-size ratios are valid.
        image, _ = letterbox_image_and_boxes(image, np.zeros((0, 4), dtype=np.float32), int(target_raw.get("input_size", image.width)))
        difficulty = image_difficulty_flags(image, target_raw["boxes"].numpy())
        draw = ImageDraw.Draw(image)
        for box in target_raw["boxes"].numpy():
            draw.rectangle(tuple(map(float, box)), outline="lime", width=3)
        pred = convert_preds_for_map([pred_raw], conf, class_agnostic, is_torchvision)[0]
        for box in pred["boxes"].numpy():
            draw.rectangle(tuple(map(float, box)), outline="red", width=3)
        draw.text((8, 8), f"FP={fp} FN={fn}", fill="yellow", stroke_width=2, stroke_fill="black")
        destination = out_dir / f"{rank:02d}_fp{fp}_fn{fn}_{path.name}"
        image.save(destination)
        rows.append({"rank": rank, "source": str(path), "output": str(destination), "fp": fp, "fn": fn, **difficulty})
    error_df = pd.DataFrame(rows)
    error_df.to_csv(out_dir / "error_index.csv", index=False)
    flag_columns = [
        "small_object", "overlap_occlusion_proxy", "low_light", "overexposed",
        "low_contrast_turbidity_proxy",
    ]
    summary_rows = []
    for column in flag_columns:
        count = int(error_df[column].sum()) if not error_df.empty else 0
        summary_rows.append({
            "challenge": column,
            "images": count,
            "percent_of_saved_errors": round(100.0 * count / max(len(error_df), 1), 3),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "error_category_summary.csv", index=False)


def _filter_detection_by_area(item: dict, category: str, image_area: float) -> dict:
    boxes = item["boxes"]
    ratios = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) / max(image_area, 1.0)
    if category == "small":
        keep = ratios < 0.01
    elif category == "medium":
        keep = (ratios >= 0.01) & (ratios < 0.09)
    else:
        keep = ratios >= 0.09
    result = {}
    for key, value in item.items():
        result[key] = value[keep] if torch.is_tensor(value) and value.ndim > 0 and len(value) == len(boxes) else value
    return result


def _scale_detection_boxes(item: dict, scale: float) -> dict:
    """Copy a detection dictionary and scale its boxes to a canonical canvas."""
    result = dict(item)
    result["boxes"] = item["boxes"].clone() * float(scale)
    return result


def save_stratified_performance(
    image_paths: Sequence[Path], preds: List[dict], targets: List[dict], out_path: Path,
    conf: float, iou_thr: float, class_agnostic: bool, is_torchvision: bool,
) -> None:
    """Save size- and image-condition-specific detection metrics."""
    rows = []

    def add_slice(kind: str, name: str, indices: Sequence[int], slice_preds: List[dict], slice_targets: List[dict]) -> None:
        if not indices or not any(len(target["boxes"]) for target in slice_targets):
            return
        metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
        metric.update(
            convert_preds_for_map(slice_preds, 0.0, class_agnostic, is_torchvision),
            convert_targets_for_map(slice_targets, class_agnostic),
        )
        computed = metric.compute()
        precision, recall, f1, tp, fp, fn = greedy_prf(
            slice_preds, slice_targets, iou_thr, conf, class_agnostic, is_torchvision
        )
        rows.append({
            "slice_type": kind, "slice": name, "images": len(indices),
            "objects": sum(len(target["boxes"]) for target in slice_targets),
            "precision_%": format_pct(precision), "recall_%": format_pct(recall), "f1_%": format_pct(f1),
            "mAP@0.5_%": format_pct(float(computed["map_50"].item())),
            "mAP@0.5:0.95_%": format_pct(float(computed["map"].item())),
            "TP": tp, "FP": fp, "FN": fn,
        })

    # COCO's small/medium/large limits are pixel-area based. Scale every model's
    # boxes to 640x640 first, otherwise SSD's native 320 input would be assigned to
    # different size bins than the 640-input models.
    canonical_preds, canonical_targets = [], []
    size_counts = Counter()
    for pred, target in zip(preds, targets):
        input_size = float(target.get("input_size", 640))
        scale = 640.0 / max(input_size, 1.0)
        canonical_preds.append(_scale_detection_boxes(pred, scale))
        canonical_target = _scale_detection_boxes(target, scale)
        canonical_targets.append(canonical_target)
        boxes = canonical_target["boxes"]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        size_counts["small"] += int((areas < 32.0 ** 2).sum().item())
        size_counts["medium"] += int(((areas >= 32.0 ** 2) & (areas < 96.0 ** 2)).sum().item())
        size_counts["large"] += int((areas >= 96.0 ** 2).sum().item())
    size_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    size_metric.update(
        convert_preds_for_map(canonical_preds, 0.0, class_agnostic, is_torchvision),
        convert_targets_for_map(canonical_targets, class_agnostic),
    )
    size_results = size_metric.compute()
    for category in ("small", "medium", "large"):
        value = float(size_results[f"map_{category}"].item())
        if size_counts[category] and value >= 0:
            rows.append({
                "slice_type": "object_size_coco_at_640", "slice": category,
                "images": len(targets), "objects": size_counts[category],
                "precision_%": "", "recall_%": "", "f1_%": "",
                "mAP@0.5_%": "", "mAP@0.5:0.95_%": format_pct(value),
                "TP": "", "FP": "", "FN": "",
            })

    condition_indices: Dict[str, List[int]] = defaultdict(list)
    for index, path in enumerate(image_paths):
        try:
            with Image.open(path) as image:
                flags = image_difficulty_flags(image, np.zeros((0, 4), dtype=np.float32))
            for condition in ("low_light", "overexposed", "low_contrast_turbidity_proxy"):
                if flags[condition]:
                    condition_indices[condition].append(index)
        except Exception:
            continue
    for condition, indices in condition_indices.items():
        add_slice("image_condition", condition, indices, [preds[i] for i in indices], [targets[i] for i in indices])

    pd.DataFrame(rows).to_csv(out_path, index=False)


@torch.no_grad()
def evaluate_torchvision_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    conf: float,
    iou_thr: float,
    class_agnostic: bool = False,
    desc: str = "eval",
    class_agnostic_nms_iou: float = 0.5,
    error_dir: Optional[Path] = None,
    error_limit: int = 20,
) -> dict:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    all_preds_raw: List[dict] = []
    all_targets_raw: List[dict] = []

    for images, targets in tqdm(loader, desc=desc):
        images = [img.to(device, non_blocking=True) for img in images]
        if class_agnostic:
            with class_agnostic_detection_nms_once(model, class_agnostic_nms_iou):
                preds = model(images)
        else:
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
        classes = np.atleast_1d(m["classes"].detach().cpu().numpy()).tolist()
        map_pc = np.atleast_1d(m["map_per_class"].detach().cpu().numpy()).tolist()
        results["per_class_map"] = {int(c): float(v) for c, v in zip(classes, map_pc) if float(v) >= 0}
    if error_dir is not None and hasattr(loader.dataset, "images"):
        for target in all_targets_raw:
            target["input_size"] = int(getattr(loader.dataset, "input_size", 640))
        save_qualitative_errors(loader.dataset.images, all_preds_raw, all_targets_raw, error_dir,
                                conf, iou_thr, class_agnostic, True, error_limit)
        save_stratified_performance(
            loader.dataset.images, all_preds_raw, all_targets_raw, error_dir / "stratified_performance.csv",
            conf, iou_thr, class_agnostic, True,
        )
    return results


@torch.no_grad()
def measure_inference_fps(
    cpu_images: Sequence[torch.Tensor], inference_call, device: torch.device,
    warmup: int = 20, samples: int = 120,
) -> Tuple[float, float]:
    """Shared batch-1 latency protocol used by all architectures.

    Images are decoded/resized before timing. The timed region includes CPU-to-GPU
    transfer, model forward, and the detector's standard post-processing/NMS.
    """
    usable = list(cpu_images[: warmup + samples])
    if not usable:
        return 0.0, 0.0
    actual_warmup = min(warmup, max(len(usable) - 1, 0))
    for image in usable[:actual_warmup]:
        inference_call(image)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    count = 0
    for image in usable[actual_warmup:actual_warmup + samples]:
        inference_call(image)
        count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    fps = count / max(dt, 1e-9)
    ms = 1000.0 / max(fps, 1e-9)
    return fps, ms


@torch.no_grad()
def measure_torchvision_fps(model: nn.Module, loader: DataLoader, device: torch.device, warmup: int = 20, samples: int = 120) -> Tuple[float, float]:
    model.eval()
    cpu_images: List[torch.Tensor] = []
    for images, _ in loader:
        cpu_images.extend(image.detach().cpu() for image in images)
        if len(cpu_images) >= warmup + samples:
            break
    return measure_inference_fps(
        cpu_images, lambda image: model([image.to(device, non_blocking=False)]),
        device, warmup, samples,
    )


@torch.no_grad()
def measure_yolo_fps(yolo_model: object, dataset: Dataset, device: torch.device,
                     img_size: int, warmup: int = 20, samples: int = 120) -> Tuple[float, float]:
    cpu_images = [dataset[index][0].detach().cpu() for index in range(min(len(dataset), warmup + samples))]
    device_arg = 0 if device.type == "cuda" else "cpu"
    return measure_inference_fps(
        cpu_images,
        lambda image: yolo_model.predict(
            image.unsqueeze(0), imgsz=img_size, conf=0.001, device=device_arg, verbose=False
        ),
        device, warmup, samples,
    )


def estimate_forward_gflops(forward_call) -> float:
    """Best-effort FLOP estimate for one forward pass using PyTorch's profiler."""
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
            forward_call()
        total_flops = sum(float(event.flops or 0) for event in prof.key_averages())
        return round(total_flops / 1e9, 4) if total_flops > 0 else math.nan
    except Exception as exc:
        warnings.warn(f"FLOP profiling was unavailable: {exc}")
        return math.nan


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
        "training_time_s": round(float(metrics.get("training_time_s", math.nan)), 3),
        "model_size_mb": round(float(metrics.get("model_size_mb", math.nan)), 3),
        "checkpoint_size_mb": round(float(metrics.get("checkpoint_size_mb", math.nan)), 3),
        "parameters": int(metrics.get("parameters", 0)),
        "GFLOPs/frame": round(float(metrics.get("gflops_per_frame", math.nan)), 4),
        "peak_gpu_memory_mb": round(float(metrics.get("peak_gpu_memory_mb", 0.0)), 3),
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
    train_loader = make_loader(
        data_root, "train", input_size, len(class_names), batch, workers, shuffle=True,
        augment=bool(cfg.get("augmentation", {}).get("enabled", True)),
        augmentation=cfg.get("augmentation", {}),
        # SSDLite contains BatchNorm layers whose 1x1 feature maps cannot train
        # on a singleton final batch (e.g. 5,377 images with batch size 16).
        drop_last=True,
    )
    val_loader = make_loader(data_root, "val", input_size, len(class_names), batch, workers, shuffle=False)
    test_loader = make_loader(data_root, "test", input_size, len(class_names), batch, workers, shuffle=False)
    fps_loader = make_loader(data_root, "test", input_size, len(class_names), 1, workers, shuffle=False)

    epochs_head = int(train_cfg.get("epochs_head", 10))
    epochs_ft = int(train_cfg.get("epochs_finetune", 100))
    session_control = cfg.get("session_control", {})
    stop_after_stage2_epoch = int(session_control.get("stop_after_stage2_epoch", epochs_ft))
    if not 1 <= stop_after_stage2_epoch <= epochs_ft:
        raise RuntimeError("session_control.stop_after_stage2_epoch must be within the full Stage 2 plan")
    session_id = str(session_control.get("session_id", "single_session"))
    if bool(cfg.get("run", {}).get("quick_debug", False)):
        epochs_head = min(1, epochs_head)
        epochs_ft = min(1, epochs_ft)
    lr = float(train_cfg.get("lr_torch", 0.001))
    out_model_dir = out_dir / "torchvision" / architecture
    out_model_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_model_dir / "best.pt"
    last_path = out_model_dir / "last.pt"
    history = []
    best_map = -1.0
    best_epoch = 0
    identity = checkpoint_identity(cfg, architecture)
    training_config_digest = config_sha256(cfg)
    session_control_digest = session_control_sha256(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()

    stages = [("head", epochs_head), ("all", epochs_ft)]
    global_epoch = 0
    resume_checkpoint = None
    resume_source = Path(session_control["resume_checkpoint"]).expanduser() if session_control.get("resume_checkpoint") else last_path
    if bool(train_cfg.get("resume", True)) and not resume_source.exists():
        raise RuntimeError(f"Resume was requested but the checkpoint does not exist: {resume_source}")
    if bool(train_cfg.get("resume", True)) and resume_source.exists():
        # Checkpoints are accepted only after identity gates and are generated by this pipeline.
        # RNG state contains Python/NumPy objects that PyTorch's weights-only loader rejects.
        resume_checkpoint = torch.load(resume_source, map_location=device, weights_only=False)
        validate_resume_checkpoint(resume_checkpoint, cfg, architecture)
        expected_sha = session_control.get("resume_checkpoint_sha256")
        if expected_sha and sha256_file(resume_source) != expected_sha:
            raise RuntimeError("Resume checkpoint SHA-256 does not match session_control.resume_checkpoint_sha256")
        model.load_state_dict(resume_checkpoint["model"])
        global_epoch = int(resume_checkpoint.get("epoch", 0))
        best_map = float(resume_checkpoint.get("best_map", resume_checkpoint.get("val_metrics", {}).get("map", -1.0)))
        best_epoch = int(resume_checkpoint.get("best_epoch", global_epoch))
        history = list(resume_checkpoint["training_history"])
        if history and int(history[-1]["epoch"]) != global_epoch:
            raise RuntimeError("Resume checkpoint history does not end at completed_epoch")
        restore_rng_state(resume_checkpoint)
        if not best_path.exists():
            warnings.warn("best.pt is missing; using the resumed last.pt as the initial best checkpoint.")
            best_map = float(resume_checkpoint.get("val_metrics", {}).get("map", best_map))
            torch.save({
                "model": model.state_dict(), "cfg": cfg, "epoch": global_epoch,
                "val_metrics": resume_checkpoint.get("val_metrics", {}),
            }, best_path)
        print(f"Resuming {model_name} after completed epoch {global_epoch} from {resume_source}")

    completed_before_stage = 0
    for stage_name, epochs in stages:
        completed_in_stage = min(max(global_epoch - completed_before_stage, 0), epochs)
        completed_before_stage += epochs
        if completed_in_stage >= epochs:
            continue
        set_torchvision_trainable(model, architecture, stage=stage_name)
        optimizer = torch.optim.SGD(trainable_params(model), lr=lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        if resume_checkpoint is not None and resume_checkpoint.get("stage") == stage_name:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            scaler.load_state_dict(resume_checkpoint["scaler"])
            resume_checkpoint = None
        for epoch in range(completed_in_stage + 1, epochs + 1):
            global_epoch += 1
            epoch_generator = torch.Generator()
            epoch_generator.manual_seed(seed + global_epoch)
            epoch_train_loader = make_loader(
                data_root, "train", input_size, len(class_names), batch, workers, shuffle=True,
                augment=bool(cfg.get("augmentation", {}).get("enabled", True)),
                augmentation=cfg.get("augmentation", {}), drop_last=True, generator=epoch_generator,
            )
            model.train()
            loss_sum = 0.0
            n_batches = 0
            pbar = tqdm(epoch_train_loader, desc=f"{model_name} {stage_name} epoch {epoch}/{epochs}")
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
                model, val_loader, device, conf=confidence_threshold(cfg),
                iou_thr=float(thr.get("iou_match", 0.5)), class_agnostic=False,
                desc=f"{model_name} val"
            )
            row = {"epoch": global_epoch, "stage": stage_name, "loss": loss_sum / max(n_batches, 1), **val_metrics}
            row["lr"] = optimizer.param_groups[0]["lr"]
            history.append(row)
            pd.DataFrame(history).to_csv(out_model_dir / "history.csv", index=False)
            if val_metrics["map"] > best_map:
                best_map = val_metrics["map"]
                best_epoch = global_epoch
                torch.save({
                    "model": model.state_dict(), "cfg": cfg, "epoch": global_epoch,
                    "architecture": architecture, "stage": stage_name, "stage_epoch": epoch,
                    "val_metrics": val_metrics, "best_map": best_map,
                    "best_epoch": best_epoch, "checkpoint_identity": identity,
                    "training_config_sha256": training_config_digest,
                }, best_path)
                print(f"Saved new best {model_name}: val mAP@0.5:0.95={best_map:.4f}")
            resume_state = capture_rng_state()
            torch.save({
                "model": model.state_dict(), "cfg": cfg, "epoch": global_epoch,
                "completed_epoch": global_epoch, "next_epoch": global_epoch + 1,
                "architecture": architecture,
                "stage": stage_name, "stage_epoch": epoch, "val_metrics": val_metrics,
                "best_map": best_map, "best_epoch": best_epoch,
                "checkpoint_identity": identity, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "training_history": history, "training_config_sha256": training_config_digest,
                "session_control_sha256": session_control_digest, "session_id": session_id,
                "seed": seed, "experiment_id": identity["experiment_id"], "git_commit": identity["git_commit"],
                "dataset_sha256": identity["dataset_sha256"], "split_sha256": identity["split_manifest_sha256"],
                "dataloader_generator_state": epoch_generator.get_state(),
                "sampler_state": {"strategy": "epoch_seeded_generator", "seed": seed + global_epoch, "next_global_epoch": global_epoch + 1},
                **resume_state,
            }, last_path)
            if stage_name == "all" and epoch == stop_after_stage2_epoch and epoch < epochs_ft:
                status = {
                    "status": "SESSION_A_COMPLETE_READY_FOR_RESUME",
                    "training_status": "TRAINING_NOT_COMPLETE",
                    "session_id": session_id,
                    "completed_epoch": global_epoch,
                    "completed_stage2_epoch": epoch,
                    "next_stage2_epoch": epoch + 1,
                    "last_checkpoint_sha256": sha256_file(last_path),
                    "training_config_sha256": training_config_digest,
                    "session_control_sha256": session_control_digest,
                }
                (out_dir / "session_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
                return model, {}, math.nan, math.nan

    if not bool(cfg.get("run", {}).get("evaluate", True)):
        return model, {}, math.nan, math.nan

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate_torchvision_model(
        model, test_loader, device, conf=confidence_threshold(cfg),
        iou_thr=float(thr.get("iou_match", 0.5)), class_agnostic=False,
        desc=f"{model_name} TEST",
        error_dir=out_dir / "qualitative_errors" / architecture,
        error_limit=int(cfg.get("qualitative_analysis", {}).get("max_images", 20)),
    )
    fps, ms = measure_torchvision_fps(model, fps_loader, device)
    profile_image = next(iter(fps_loader))[0][0].to(device)
    gflops = estimate_forward_gflops(lambda: model([profile_image]))
    test_metrics.update({
        "training_time_s": time.perf_counter() - training_started,
        "model_size_mb": model_weights_size_mb(model),
        "checkpoint_size_mb": best_path.stat().st_size / (1024 ** 2),
        "parameters": sum(p.numel() for p in model.parameters()),
        "gflops_per_frame": gflops,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0,
    })
    return model, test_metrics, fps, ms


def train_yolo_detector(paths: PreparedPaths, out_dir: Path, cfg: dict) -> Tuple[object, dict, float, float]:
    try:
        from ultralytics import YOLO
    except Exception as e:
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            print("Installing Ultralytics in the Kaggle runtime...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.0"])
            from ultralytics import YOLO
        else:
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
    aug = cfg.get("augmentation", {})
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    resume_enabled = bool(train_cfg.get("resume", True))
    stage1_dir = project / "yolov8s_stage1"
    stage2_dir = project / "yolov8s_stage2"

    def completed_epochs(run_dir: Path) -> int:
        results_path = run_dir / "results.csv"
        if not results_path.exists():
            return 0
        try:
            return len(pd.read_csv(results_path))
        except Exception:
            return 0

    def primary_map_checkpoint(run_dir: Path) -> Path:
        """Select the saved epoch with the highest validation mAP@0.5:0.95."""
        results_path = run_dir / "results.csv"
        if results_path.exists():
            try:
                frame = pd.read_csv(results_path)
                frame.columns = [str(column).strip() for column in frame.columns]
                columns = [column for column in frame.columns if "mAP50-95" in column]
                if columns:
                    epoch_index = int(pd.to_numeric(frame[columns[0]], errors="coerce").idxmax())
                    epoch_path = run_dir / "weights" / f"epoch{epoch_index}.pt"
                    if epoch_path.exists():
                        return epoch_path
            except Exception as exc:
                warnings.warn(f"Could not select YOLO checkpoint by mAP@0.5:0.95: {exc}")
        fallback = run_dir / "weights" / "best.pt"
        return fallback if fallback.exists() else run_dir / "weights" / "last.pt"

    stage1_last = stage1_dir / "weights" / "last.pt"
    stage1_best = stage1_dir / "weights" / "best.pt"
    if resume_enabled and stage1_last.exists() and completed_epochs(stage1_dir) < epochs_head:
        print(f"[{now()}] Resuming YOLOv8s Stage 1 from {stage1_last}")
        res1 = YOLO(str(stage1_last)).train(resume=True)
        stage1_dir = Path(res1.save_dir)
        stage1_best = primary_map_checkpoint(stage1_dir)
    elif not resume_enabled or completed_epochs(stage1_dir) < epochs_head:
        model = YOLO("yolov8s.pt")
        print(f"[{now()}] YOLOv8s Stage 1: frozen adaptation for {epochs_head} epochs")
        res1 = model.train(
            data=str(paths.trash_yaml), epochs=epochs_head, imgsz=imgsz, batch=batch,
            project=str(project), name="yolov8s_stage1", exist_ok=True, seed=seed,
            deterministic=True, workers=workers, optimizer="AdamW", lr0=lr0,
            freeze=10, amp=bool(train_cfg.get("amp", True)), cos_lr=True, save_period=1,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0,
            fliplr=float(aug.get("horizontal_flip", 0.5)), flipud=0.0,
            hsv_v=float(aug.get("brightness", 0.0)), hsv_s=float(aug.get("saturation", 0.0)), hsv_h=0.0,
        )
        stage1_dir = Path(res1.save_dir)
        stage1_best = primary_map_checkpoint(stage1_dir)
    elif resume_enabled:
        stage1_best = primary_map_checkpoint(stage1_dir)
    if not stage1_best.exists():
        stage1_best = stage1_dir / "weights" / "last.pt"
    if not stage1_best.exists():
        raise RuntimeError("YOLO Stage 1 produced no checkpoint.")

    stage2_last = stage2_dir / "weights" / "last.pt"
    if resume_enabled and stage2_last.exists() and completed_epochs(stage2_dir) < epochs_ft:
        print(f"[{now()}] Resuming YOLOv8s Stage 2 from {stage2_last}")
        res2 = YOLO(str(stage2_last)).train(resume=True)
        stage2_dir = Path(res2.save_dir)
    elif not resume_enabled or completed_epochs(stage2_dir) < epochs_ft:
        print(f"[{now()}] YOLOv8s Stage 2: full fine-tuning for {epochs_ft} epochs")
        model = YOLO(str(stage1_best))
        res2 = model.train(
            data=str(paths.trash_yaml), epochs=epochs_ft, imgsz=imgsz, batch=batch,
            project=str(project), name="yolov8s_stage2", exist_ok=True, seed=seed,
            deterministic=True, workers=workers, optimizer="AdamW", lr0=lr0,
            freeze=0, amp=bool(train_cfg.get("amp", True)), cos_lr=True, save_period=1,
            close_mosaic=0, patience=0,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, degrees=0.0, translate=0.0, scale=0.0,
            fliplr=float(aug.get("horizontal_flip", 0.5)), flipud=0.0,
            hsv_v=float(aug.get("brightness", 0.0)), hsv_s=float(aug.get("saturation", 0.0)), hsv_h=0.0,
        )
        stage2_dir = Path(res2.save_dir)
    best = primary_map_checkpoint(stage2_dir)
    if not best.exists():
        best = stage2_dir / "weights" / "last.pt"
    if not best.exists():
        raise RuntimeError("YOLO Stage 2 produced no checkpoint.")
    yolo_model = YOLO(str(best))

    if not bool(cfg.get("run", {}).get("evaluate", True)):
        return yolo_model, {}, math.nan, math.nan

    metrics, fps, ms = evaluate_yolo_model(
        yolo_model, paths.trash_dir, "test", img_size=imgsz,
        conf=confidence_threshold(cfg),
        iou_thr=float(cfg.get("thresholds", {}).get("iou_match", 0.5)),
        class_agnostic=False,
        num_classes=len(cfg.get("class_names") or DEFAULT_CLASSES),
        desc="YOLOv8s TEST",
        error_dir=out_dir / "qualitative_errors" / "yolov8s",
        error_limit=int(cfg.get("qualitative_analysis", {}).get("max_images", 20)),
    )
    yolo_device = next(yolo_model.model.parameters()).device
    profile_tensor = torch.zeros((1, 3, imgsz, imgsz), device=yolo_device)
    gflops = estimate_forward_gflops(lambda: yolo_model.model(profile_tensor))
    metrics.update({
        "training_time_s": time.perf_counter() - training_started,
        "model_size_mb": model_weights_size_mb(yolo_model.model),
        "checkpoint_size_mb": best.stat().st_size / (1024 ** 2),
        "parameters": sum(p.numel() for p in yolo_model.model.parameters()),
        "gflops_per_frame": gflops,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0,
    })
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
    num_classes: int,
    desc: str,
    class_agnostic_nms_iou: float = 0.5,
    error_dir: Optional[Path] = None,
    error_limit: int = 20,
) -> Tuple[dict, float, float]:
    # We use the same dataset class for consistent target geometry.
    dataset_num_classes = 1 if class_agnostic else int(num_classes)
    ds = YoloDetectionDataset(yolo_root, split, img_size, num_classes=dataset_num_classes, class_agnostic=class_agnostic)
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    preds_raw: List[dict] = []
    targets_raw: List[dict] = []

    for i in tqdm(range(len(ds)), desc=desc):
        img_tensor, target = ds[i]
        # yolo_model.predict accepts numpy images, but target boxes are in letterboxed coords.
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        result = yolo_model.predict(
            img_np, imgsz=img_size, conf=0.001, verbose=False,
            agnostic_nms=class_agnostic,
            iou=class_agnostic_nms_iou if class_agnostic else 0.7,
        )[0]
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
        targ = {"boxes": target["boxes"].detach().cpu(), "labels": target["labels"].detach().cpu(), "input_size": img_size}
        preds_raw.append(pred)
        targets_raw.append(targ)
        metric.update(
            convert_preds_for_map([pred], conf=0.0, class_agnostic=class_agnostic, is_torchvision=False),
            convert_targets_for_map([targ], class_agnostic=class_agnostic),
        )

    yolo_device = next(yolo_model.model.parameters()).device
    fps, ms = measure_yolo_fps(yolo_model, ds, yolo_device, img_size)
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
        classes = np.atleast_1d(m["classes"].detach().cpu().numpy()).tolist()
        map_pc = np.atleast_1d(m["map_per_class"].detach().cpu().numpy()).tolist()
        results["per_class_map"] = {int(c): float(v) for c, v in zip(classes, map_pc) if float(v) >= 0}
    if error_dir is not None:
        save_qualitative_errors(ds.images, preds_raw, targets_raw, error_dir, conf, iou_thr,
                                class_agnostic, False, error_limit)
        save_stratified_performance(
            ds.images, preds_raw, targets_raw, error_dir / "stratified_performance.csv",
            conf, iou_thr, class_agnostic, False,
        )
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
    nms_iou = float(thr.get("class_agnostic_nms_iou", 0.5))

    if yolo_model is not None:
        m, fps, ms = evaluate_yolo_model(
            yolo_model, paths.river_dir, "test", int(train_cfg.get("imgsz_yolo", 640)),
            conf=confidence_threshold(cfg), iou_thr=iou_thr,
            class_agnostic=True, num_classes=len(cfg.get("class_names") or DEFAULT_CLASSES),
            desc="YOLOv8s River class-agnostic", class_agnostic_nms_iou=nms_iou,
        )
        rows.append(metrics_row("YOLOv8s", "River cross-domain", m, fps, ms))

    if frcnn_model is not None:
        loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_frcnn", 640)), 1, 1, workers, False, class_agnostic=True)
        m = evaluate_torchvision_model(
            frcnn_model, loader, device, conf=confidence_threshold(cfg), iou_thr=iou_thr,
            class_agnostic=True, desc="Faster R-CNN River class-agnostic",
            class_agnostic_nms_iou=nms_iou,
        )
        fps_loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_frcnn", 640)), 1, 1, workers, False, class_agnostic=True)
        fps, ms = measure_torchvision_fps(frcnn_model, fps_loader, device)
        rows.append(metrics_row("Faster R-CNN", "River cross-domain", m, fps, ms))

    if ssd_model is not None:
        loader = make_loader(paths.river_dir, "test", int(train_cfg.get("imgsz_ssd", 320)), 1, 1, workers, False, class_agnostic=True)
        m = evaluate_torchvision_model(
            ssd_model, loader, device, conf=confidence_threshold(cfg), iou_thr=iou_thr,
            class_agnostic=True, desc="MobileNet SSD River class-agnostic",
            class_agnostic_nms_iou=nms_iou,
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
        md.append(dataframe_markdown(ds))
    except Exception:
        md.append("Dataset summary unavailable.")
    md.append("\n\n## Overall in-domain test results\n")
    md.append(dataframe_markdown(overall) if not overall.empty else "No results.")
    md.append("\n\n## Per-class mAP\n")
    md.append(dataframe_markdown(per_class) if not per_class.empty else "No per-class results.")
    md.append("\n\n## Cross-domain river results\n")
    md.append(dataframe_markdown(cross) if not cross.empty else "Cross-domain results unavailable or skipped.")
    md.append("\n\n## Important paper wording\n")
    md.append(
        "- Report these results as **test-set results** only if this script completed evaluation on the `test` split.\n"
        "- Cross-domain river evaluation here is **class-agnostic IoU matching**: all river labels are remapped to one `trash` class and model predictions are evaluated as object localization, not class-name compatibility.\n"
        "- One common class-agnostic NMS is applied to every model after River class collapsing.\n"
        "- Precision/Recall/F1 use one configured confidence threshold for every model; mAP remains threshold-independent.\n"
        "- The controlled paper configuration uses horizontal flip only; model-specific color augmentation is disabled.\n"
        "- FPS uses batch 1, 20 warm-up frames, up to 120 timed frames, and includes host-to-device transfer plus detector post-processing.\n"
        "- Object-size mAP uses COCO area bins after scaling every model's boxes to a canonical 640x640 canvas.\n"
        "- Model size compares state-dictionary tensor bytes; checkpoint file size is reported separately.\n"
        "- Do not copy old notebook numbers if they differ from these CSV files. Update the paper tables from this folder.\n"
    )
    (out_dir / "results_for_paper.md").write_text("\n".join(md), encoding="utf-8")


# -----------------------------
# Main
# -----------------------------


def load_config(args) -> dict:
    if args.config and Path(args.config).exists():
        cfg = read_yaml(Path(args.config))
    else:
        cfg = {}
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not cfg:
        raise RuntimeError(
            "Canonical runs require an explicit immutable YAML config; "
            "the legacy auto-generated Kaggle configuration is disabled."
        )
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


def run_single_seed(paths: PreparedPaths, out_dir: Path, cfg: dict, device: torch.device) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_cfg = cfg.get("run", {})
    yolo_model = frcnn_model = ssd_model = None
    results_by_model: Dict[str, dict] = {}
    overall_rows = []
    seed = int(cfg.get("seed", 42))

    evaluate_enabled = bool(run_cfg.get("evaluate", True))
    if bool(run_cfg.get("train_yolo", True)):
        yolo_model, metrics, fps, ms = train_yolo_detector(paths, out_dir, cfg)
        if evaluate_enabled:
            results_by_model["YOLOv8s"] = metrics
            row = metrics_row("YOLOv8s", "Trash-ICRA19 test", metrics, fps, ms)
            row["seed"] = seed
            overall_rows.append(row)
    if bool(run_cfg.get("train_frcnn", True)):
        frcnn_model, metrics, fps, ms = train_torchvision_detector("frcnn", paths.trash_dir, out_dir, cfg, device)
        if evaluate_enabled:
            results_by_model["Faster R-CNN"] = metrics
            row = metrics_row("Faster R-CNN", "Trash-ICRA19 test", metrics, fps, ms)
            row["seed"] = seed
            overall_rows.append(row)
    if bool(run_cfg.get("train_ssd", True)):
        ssd_model, metrics, fps, ms = train_torchvision_detector("ssd", paths.trash_dir, out_dir, cfg, device)
        if evaluate_enabled:
            results_by_model["MobileNet SSD"] = metrics
            row = metrics_row("MobileNet SSD", "Trash-ICRA19 test", metrics, fps, ms)
            row["seed"] = seed
            overall_rows.append(row)

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(out_dir / "results_overall_test.csv", index=False)
    per_class = save_per_class(results_by_model, list(cfg.get("class_names") or DEFAULT_CLASSES), out_dir)
    cross = evaluate_cross_domain(yolo_model, frcnn_model, ssd_model, paths, cfg, device, out_dir) if evaluate_enabled else pd.DataFrame()
    if not cross.empty:
        cross["seed"] = seed
        cross.to_csv(out_dir / "results_cross_domain.csv", index=False)
    plot_results(overall, out_dir)
    write_markdown_report(out_dir, paths.summary_csv, overall, per_class, cross, cfg)
    return overall, per_class, cross


def summarize_runs(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    numeric = [c for c in runs.select_dtypes(include=[np.number]).columns if c != "seed"]
    summary_rows = []
    distribution_rows = []
    for (model, split), group in runs.groupby(["model", "split"], dropna=False):
        row = {"model": model, "split": split, "n_runs": int(len(group))}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce").dropna().astype(float)
            n = len(values)
            mean = float(values.mean()) if n else math.nan
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            margin = t_critical_95(n - 1) * sd / math.sqrt(n) if n > 1 else math.nan
            row.update({
                f"{column}_mean": mean,
                f"{column}_std": sd,
                f"{column}_ci95_low": mean - margin if n > 1 else math.nan,
                f"{column}_ci95_high": mean + margin if n > 1 else math.nan,
            })
            for seed, value in zip(group.loc[values.index, "seed"], values):
                distribution_rows.append({
                    "model": model, "split": split, "metric": column,
                    "seed": int(seed), "value": float(value),
                })
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "results_mean_sd.csv", index=False)
    distribution = pd.DataFrame(distribution_rows)
    distribution.to_csv(out_dir / "results_metric_distributions.csv", index=False)
    report_lines = [
        "# Statistical robustness across seeds",
        "",
        "Intervals are two-sided 95% Student-t confidence intervals. At least two runs are required; three or more are recommended.",
        "",
        dataframe_markdown(summary),
    ]
    (out_dir / "statistical_summary.md").write_text("\n".join(report_lines), encoding="utf-8")

    selected = [c for c in ("mAP@0.5_%", "mAP@0.5:0.95_%", "precision_%", "recall_%", "f1_%", "FPS") if c in runs]
    if selected:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(math.ceil(len(selected) / 2), 2, figsize=(12, 3.8 * math.ceil(len(selected) / 2)))
        axes = np.atleast_1d(axes).reshape(-1)
        labels = runs["model"].astype(str) + " | " + runs["split"].astype(str)
        for axis, metric in zip(axes, selected):
            categories = list(dict.fromkeys(labels))
            values_by_category = [pd.to_numeric(runs.loc[labels == category, metric], errors="coerce").dropna() for category in categories]
            try:
                axis.boxplot(values_by_category, tick_labels=categories, showmeans=True)
            except TypeError:  # Matplotlib < 3.9
                axis.boxplot(values_by_category, labels=categories, showmeans=True)
            axis.set_title(f"{metric} across seeds")
            axis.tick_params(axis="x", rotation=25)
            axis.grid(axis="y", alpha=0.25)
        for axis in axes[len(selected):]:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_metric_distributions.png", dpi=200)
        plt.close(fig)
    return summary


def aggregate_completed_runs(search_root: Path, out_dir: Path) -> pd.DataFrame:
    """Combine independently downloaded seed runs and regenerate statistics."""
    frames = []
    for filename in ("results_overall_test.csv", "results_cross_domain.csv"):
        for path in search_root.rglob(filename):
            try:
                frame = pd.read_csv(path)
            except Exception as exc:
                warnings.warn(f"Skipping unreadable result file {path}: {exc}")
                continue
            if frame.empty or not {"model", "split", "seed"}.issubset(frame.columns):
                continue
            frame["source_file"] = str(path.resolve())
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No completed per-seed result CSV files found under {search_root}")
    combined = pd.concat(frames, ignore_index=True)
    combined["seed"] = pd.to_numeric(combined["seed"], errors="raise").astype(int)
    combined = combined.drop_duplicates(subset=["model", "split", "seed"], keep="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "results_all_runs.csv", index=False)
    summarize_runs(combined.drop(columns=["source_file"]), out_dir)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled marine debris detection comparison on RunPod.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--trash-root", type=str, default=None, help="Override Trash-ICRA19 root path")
    parser.add_argument("--river-root", type=str, default=None, help="Override River Floating Trash root path")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--quick-debug", action="store_true", help="Run a tiny 1-epoch test for debugging")
    parser.add_argument("--check-sequences-only", action="store_true",
                        help="Validate sequence_regex for every Trash image, write an audit CSV, then exit")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate canonical provenance and approval gates, then exit without training")
    parser.add_argument("--aggregate-root", type=str, default=None,
                        help="Only aggregate completed result CSVs found recursively under this directory")
    args = parser.parse_args()

    if args.aggregate_root:
        aggregate_root = Path(args.aggregate_root).expanduser().resolve()
        aggregate_out = Path(args.out_dir).expanduser().resolve() if args.out_dir else aggregate_root / "aggregated_statistics"
        combined = aggregate_completed_runs(aggregate_root, aggregate_out)
        print(f"Aggregated {len(combined)} unique model/split/seed rows into {aggregate_out}")
        return

    cfg = load_config(args)
    required = ["trash_root", "out_dir"]
    missing = [k for k in required if k not in cfg or not cfg[k]]
    if missing:
        raise RuntimeError(f"Missing config keys: {missing}. Edit config.yaml or pass CLI paths.")

    provenance = cfg.get("provenance", {})
    repo_root = Path(__file__).resolve().parent
    manifest_value = provenance.get("canonical_manifest")
    fingerprint_value = provenance.get("fingerprint_file")
    if not manifest_value or not fingerprint_value:
        raise RuntimeError("Canonical provenance paths are required before training.")
    manifest_path = Path(manifest_value)
    fingerprint_path = Path(fingerprint_value)
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if not fingerprint_path.is_absolute():
        fingerprint_path = repo_root / fingerprint_path
    if not manifest_path.is_file() or not fingerprint_path.is_file():
        raise RuntimeError("Canonical manifest or fingerprint file is missing.")
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    for key in ("dataset_sha256", "split_manifest_sha256"):
        if not provenance.get(key) or provenance[key] != fingerprint.get(key):
            raise RuntimeError(f"Configured provenance.{key} does not match the canonical fingerprint.")
    river_fingerprint_value = provenance.get("river_fingerprint_file")
    if not river_fingerprint_value:
        raise RuntimeError("Canonical River fingerprint path is required before training.")
    river_fingerprint_path = Path(river_fingerprint_value)
    if not river_fingerprint_path.is_absolute():
        river_fingerprint_path = repo_root / river_fingerprint_path
    if not river_fingerprint_path.is_file():
        raise RuntimeError("Canonical River fingerprint file is missing.")
    river_fingerprint = json.loads(river_fingerprint_path.read_text(encoding="utf-8"))
    river_expected = {
        "river_source_sha256": river_fingerprint.get("dataset_sha256"),
        "river_evaluation_sha256": river_fingerprint.get("evaluation_dataset_sha256"),
    }
    for key, actual in river_expected.items():
        if not provenance.get(key) or provenance[key] != actual:
            raise RuntimeError(f"Configured provenance.{key} does not match the canonical River fingerprint.")
    verifier = repo_root / "tools" / "verify_dataset.py"
    subprocess.run(
        [sys.executable, str(verifier), "--manifest", str(manifest_path),
         "--data-root", str(Path(cfg["trash_root"]).expanduser().resolve())],
        cwd=repo_root, check=True,
    )
    approval = cfg.get("approval", {})
    gate_errors = []
    if approval.get("training_authorized") is not True:
        gate_errors.append("training_authorized is false")
    if approval.get("river_duplicate_policy") != "exclude_all_conflicting_groups":
        gate_errors.append("River duplicate policy is not exclude_all_conflicting_groups")
    if approval.get("river_nms_protocol") != "framework_class_agnostic_once":
        gate_errors.append("River NMS protocol is not framework_class_agnostic_once")
    if gate_errors and not args.check_sequences_only:
        raise RuntimeError("Training approval gate is closed: " + "; ".join(gate_errors))
    if not args.check_sequences_only and list(cfg.get("experiment", {}).get("seeds", [])) != [42]:
        raise RuntimeError("The initial canonical run must use experiment.seeds: [42].")
    run_cfg = cfg.get("run", {})
    training_requested = any(bool(run_cfg.get(key, False)) for key in ("train_yolo", "train_frcnn", "train_ssd"))
    out_dir = Path(cfg["out_dir"]).expanduser().resolve()
    if training_requested and not bool(cfg.get("training", {}).get("resume", False)):
        if out_dir.exists() and any(out_dir.iterdir()):
            raise RuntimeError(f"Clean canonical output directory is not empty: {out_dir}")
    if args.preflight_only:
        if not torch.cuda.is_available():
            raise RuntimeError("Canonical training requires a CUDA GPU; this environment is CPU-only.")
        print(f"Canonical training preflight passed for new output directory {out_dir}; no training was started.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.check_sequences_only:
        trash_root = Path(cfg["trash_root"]).expanduser().resolve()
        records = scan_yolo_records(trash_root)
        if not records:
            raise RuntimeError(f"No images found under {trash_root}")
        audit_path = out_dir / "sequence_regex_audit.csv"
        audit = save_sequence_regex_audit(records, cfg, audit_path)
        print(
            f"Sequence regex matched all {len(audit)} images across "
            f"{audit['sequence'].nunique()} sequences. Audit: {audit_path}"
        )
        return
    write_yaml(out_dir / "used_config.yaml", cfg)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device()
    if training_requested and device.type != "cuda" and not bool(run_cfg.get("quick_debug", False)):
        raise RuntimeError("Canonical training requires a CUDA GPU; refusing an accidental CPU training run.")
    print(f"[{now()}] Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    system_details = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": sys.version,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "torchvision_version": importlib.metadata.version("torchvision"),
        "ultralytics_version": importlib.metadata.version("ultralytics"),
    }
    with (out_dir / "system_details.json").open("w", encoding="utf-8") as stream:
        json.dump(system_details, stream, indent=2)
    with (out_dir / "python_environment.txt").open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True, stdout=stream, text=True)
    experiment_protocol = {
        "confidence_threshold": confidence_threshold(cfg),
        "iou_match_threshold": float(cfg.get("thresholds", {}).get("iou_match", 0.5)),
        "class_agnostic_nms_iou": float(cfg.get("thresholds", {}).get("class_agnostic_nms_iou", 0.5)),
        "augmentation": cfg.get("augmentation", {}),
        "latency": {
            "batch_size": 1, "warmup_frames": 20, "max_timed_frames": 120,
            "input": "predecoded_and_resized_cpu_tensor",
            "includes": ["host_to_device_transfer", "model_forward", "postprocessing_and_nms"],
        },
        "object_size": {
            "canonical_canvas": "640x640", "standard": "COCO area bins",
            "small": "area < 32^2 pixels", "medium": "32^2 <= area < 96^2 pixels",
            "large": "area >= 96^2 pixels",
        },
        "checkpoint_selection": "highest validation mAP@0.5:0.95",
        "model_size": "state_dict tensor bytes only; checkpoint size reported separately",
    }
    with (out_dir / "experiment_protocol.json").open("w", encoding="utf-8") as stream:
        json.dump(experiment_protocol, stream, indent=2)

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
        if cfg.get("split_mode") == "sequence_70_15_15":
            validate_prepared_sequence_split(paths.trash_dir, out_dir, cfg)

    seeds = list(cfg.get("experiment", {}).get("seeds", [cfg.get("seed", 42)]))
    all_runs = []
    for seed_value in seeds:
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = int(seed_value)
        run_dir = out_dir / "runs" / f"seed_{int(seed_value)}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_yaml(run_dir / "used_config.yaml", seed_cfg)
        set_seed(int(seed_value))
        print(f"\n[{now()}] Starting training seed {seed_value}")
        overall_df, _, cross_df = run_single_seed(paths, run_dir, seed_cfg, device)
        all_runs.append(overall_df)
        if not cross_df.empty:
            all_runs.append(cross_df)
    runs_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    runs_df.to_csv(out_dir / "results_all_runs.csv", index=False)
    summarize_runs(runs_df, out_dir)

    print("\nFinished. Main outputs:")
    for p in [
        out_dir / "dataset_summary.csv",
        out_dir / "results_all_runs.csv",
        out_dir / "results_mean_sd.csv",
        out_dir / "results_metric_distributions.csv",
        out_dir / "statistical_summary.md",
        out_dir / "fig_metric_distributions.png",
        out_dir / "deduplication_report.csv",
        out_dir / "sequence_split_audit.csv",
        out_dir / "system_details.json",
        out_dir / "experiment_protocol.json",
    ]:
        if p.exists():
            print(f" - {p}")


if __name__ == "__main__":
    main()
