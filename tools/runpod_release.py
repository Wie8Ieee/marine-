"""Shared fail-closed contracts for the three-model RunPod release."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

RELEASE_BASE_COMMIT = "74e0d7d87b52b6fc6a6b7af3c5d6136cdd37c291"
DATASET_SHA256 = "5e0f560955eaf8ae4c517aa7eb80215f273dc976e9385f71e87d22f6ffa9e4cf"
SPLIT_SHA256 = "a0d4ad351b536dbfde96926c7500b09c62d2d24bfacc01831c7cf2ed65fa3d94"
MODEL_CONFIGS = {
    "yolo": "config_runpod_yolo_seed42.yaml",
    "frcnn": "config_runpod_frcnn_seed42.yaml",
    "ssd": "config_runpod_ssd_seed42.yaml",
}
MODEL_OUTPUTS = {
    "yolo": "/workspace/persistent/marine_canonical/yolov8s_seed42",
    "frcnn": "/workspace/persistent/marine_canonical/frcnn_seed42",
    "ssd": "/workspace/persistent/marine_canonical/ssd_seed42",
}

def source_commit(repo: Path) -> str:
    import subprocess
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    marker = repo / "RELEASE_COMMIT.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    raise RuntimeError("Cannot establish source commit from Git or RELEASE_COMMIT.txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_and_validate_config(path: Path, model: str, *, allow_preflight: bool = False,
                             allow_output_override: bool = False) -> dict:
    if model not in MODEL_CONFIGS:
        raise RuntimeError(f"Unsupported model: {model}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = []
    run = cfg.get("run", {})
    training = cfg.get("training", {})
    provenance = cfg.get("provenance", {})
    expected_flags = {"train_yolo": model == "yolo", "train_frcnn": model == "frcnn",
                      "train_ssd": model == "ssd", "evaluate": False}
    for key, expected in expected_flags.items():
        if run.get(key) is not expected:
            errors.append(f"run.{key} must be {expected}")
    if not allow_preflight and run.get("quick_debug") is not False:
        errors.append("run.quick_debug must be false")
    if allow_preflight and run.get("quick_debug") is not True:
        errors.append("preflight run.quick_debug must be true")
    if cfg.get("session_control"):
        errors.append("continuous RunPod execution must not define session_control")
    if run.get("resume_smoke_test"):
        errors.append("continuous RunPod execution must not enable resume_smoke_test")
    checks = {
        "seed": (cfg.get("seed"), 42),
        "split_seed": (cfg.get("split_seed"), 42),
        "split_mode": (cfg.get("split_mode"), "sequence_70_15_15"),
        "experiment.seeds": (cfg.get("experiment", {}).get("seeds"), [42]),
        "training.resume": (training.get("resume"), False),
        "training.epochs_head": (training.get("epochs_head"), 10 if not allow_preflight else 1),
        "training.epochs_finetune": (training.get("epochs_finetune"), 100 if not allow_preflight else 1),
        "training.imgsz_yolo": (training.get("imgsz_yolo"), 640),
        "training.imgsz_frcnn": (training.get("imgsz_frcnn"), 640),
        "training.imgsz_ssd": (training.get("imgsz_ssd"), 320),
        "training.batch_yolo": (training.get("batch_yolo"), 16),
        "training.batch_frcnn": (training.get("batch_frcnn"), 8),
        "training.batch_ssd": (training.get("batch_ssd"), 16),
        "training.lr_yolo": (training.get("lr_yolo"), 0.01),
        "training.lr_torch": (training.get("lr_torch"), 0.001),
        "training.workers": (training.get("workers"), 8),
        "training.amp": (training.get("amp"), True),
        "augmentation.enabled": (cfg.get("augmentation", {}).get("enabled"), True),
        "augmentation.horizontal_flip": (cfg.get("augmentation", {}).get("horizontal_flip"), 0.5),
        "augmentation.brightness": (cfg.get("augmentation", {}).get("brightness"), 0.0),
        "augmentation.saturation": (cfg.get("augmentation", {}).get("saturation"), 0.0),
        "provenance.dataset_sha256": (provenance.get("dataset_sha256"), DATASET_SHA256),
        "provenance.split_manifest_sha256": (provenance.get("split_manifest_sha256"), SPLIT_SHA256),
        "class_names": (cfg.get("class_names"), ["plastic", "bio", "rov"]),
        "thresholds.confidence": (cfg.get("thresholds", {}).get("confidence"), 0.25),
        "thresholds.iou_match": (cfg.get("thresholds", {}).get("iou_match"), 0.5),
        "approval.training_authorized": (cfg.get("approval", {}).get("training_authorized"), True),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{label}: {actual!r} != {expected!r}")
    expected_output = MODEL_OUTPUTS[model]
    actual_output = str(cfg.get("out_dir", ""))
    if not allow_preflight and not allow_output_override and actual_output != expected_output:
        errors.append(f"out_dir must be {expected_output}")
    if allow_output_override and not actual_output.startswith("/workspace/persistent/"):
        errors.append("runtime out_dir must be below /workspace/persistent")
    if errors:
        raise RuntimeError("Canonical config contract failed: " + "; ".join(errors))
    return cfg


def config_sha256(path: Path) -> str:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                                 default=str).encode("utf-8")).hexdigest()


def training_config_sha256(cfg: dict) -> str:
    stable = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    for key in ("trash_root", "river_root", "out_dir"):
        stable.pop(key, None)
    stable.get("training", {}).pop("resume", None)
    stable.get("run", {}).pop("quick_debug", None)
    stable.get("run", {}).pop("preflight_sample_limit", None)
    stable.pop("session_control", None)
    return hashlib.sha256(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
