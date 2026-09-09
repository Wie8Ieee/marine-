#!/usr/bin/env python3
"""Evaluation-only runner for the accepted canonical three-model run."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml


EXPECTED_COMMIT = "aed8bedd74010a0c29b70e6bf65ea8fdecc4fb44"
EXPECTED_DATASET = "5e0f560955eaf8ae4c517aa7eb80215f273dc976e9385f71e87d22f6ffa9e4cf"
EXPECTED_SPLIT = "a0d4ad351b536dbfde96926c7500b09c62d2d24bfacc01831c7cf2ed65fa3d94"
EXPECTED_RIVER_SOURCE = "5a8f7762c09d88e9fb823a03cf880e20a707b7c1f0e8253c567a98c086563ce1"
EXPECTED_RIVER_EVAL = "5f6f1a7ca2f8edef49c1b8049b405d9065a4cb55c7118e690e60a4d6c17a8af1"
MODELS = ("yolo", "frcnn", "ssd")
CLASS_NAMES = ("plastic", "bio", "rov")


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def snapshot_training_artifacts(group: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for model in MODELS:
        root = group / model
        manifest_path = root / "training_artifact_manifest.json"
        require(manifest_path.is_file(), f"Missing training manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(manifest.get("status") == "VERIFIED_COMPLETE", f"{model} is not VERIFIED_COMPLETE")
        require(manifest.get("source_commit") == EXPECTED_COMMIT, f"{model} commit mismatch")
        provenance = manifest.get("provenance", {})
        require(provenance.get("dataset_sha256") == EXPECTED_DATASET, f"{model} dataset SHA mismatch")
        require(provenance.get("split_manifest_sha256") == EXPECTED_SPLIT, f"{model} split SHA mismatch")
        files = {"training_artifact_manifest.json": manifest_path}
        for checkpoint_name in ("best.pt", "last.pt"):
            item = manifest.get("checkpoints", {}).get(checkpoint_name, {})
            path = Path(str(item.get("path", "")))
            require(path.is_file() and path.stat().st_size > 0, f"Missing/empty {model} {checkpoint_name}")
            require(sha256(path) == item.get("sha256"), f"{model} {checkpoint_name} SHA mismatch")
            files[checkpoint_name] = path
        result[model] = {
            "manifest": manifest,
            "files": {name: {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in files.items()},
        }
    return result


def load_best_val(group: Path, snapshots: dict) -> dict[str, dict[str, float | int]]:
    values: dict[str, dict[str, float | int]] = {}
    for model in ("frcnn", "ssd"):
        manifest = snapshots[model]["manifest"]
        epoch = int(manifest["details"]["best_epoch"])
        history = group / model / "runs" / "seed_42" / "torchvision" / model / "history.csv"
        frame = pd.read_csv(history)
        row = frame.loc[pd.to_numeric(frame["epoch"]) == epoch]
        require(len(row) == 1, f"Cannot find unique best epoch for {model}")
        values[model] = {"best_epoch": epoch, "map": float(row.iloc[0]["map"])}
    selection = group / "yolo" / "output" / "runs" / "seed_42" / "yolo" / "yolov8s_stage2" / "checkpoint_selection.json"
    if not selection.is_file():
        candidates = list((group / "yolo").rglob("checkpoint_selection.json"))
        require(len(candidates) == 1, "Cannot locate unique YOLO checkpoint_selection.json")
        selection = candidates[0]
    selected = json.loads(selection.read_text(encoding="utf-8"))
    require(selected.get("status") == "SELECTED_BY_VALIDATION_MAP50_95", "YOLO best selection contract mismatch")
    values["yolo"] = {"best_epoch": int(selected["selected_epoch"]), "map": float(selected["selected_metric"])}
    return values


def metrics_payload(metrics: dict, fps: float, ms: float) -> dict[str, object]:
    return {
        "map_50": float(metrics["map_50"]),
        "map_50_95": float(metrics["map"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "per_class_ap_50_95": {CLASS_NAMES[int(k)]: float(v) for k, v in (metrics.get("per_class_map") or {}).items()},
        "fps": float(fps),
        "ms_per_frame": float(ms),
    }


def river_payload(metrics: dict) -> dict[str, object]:
    return {
        "map_50": float(metrics["map_50"]),
        "map_50_95": float(metrics["map"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--river-root", type=Path, required=True)
    parser.add_argument("--river-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo, group, river_root, river_archive, out = (
        p.resolve() for p in (args.repo, args.group, args.river_root, args.river_archive, args.out)
    )
    require(not out.exists(), f"Clean evaluation output already exists: {out}")
    out.mkdir(parents=True)
    status_path = out / "evaluation_status.json"
    atomic_json(status_path, {"status": "RUNNING", "started_utc": utc()})
    try:
        sys.path.insert(0, str(repo))
        import marine_3model_experiment as exp
        sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))
        sys.path.insert(0, str(repo / "tools"))
        import build_river_manifest
        import verify_dataset
        from ultralytics import YOLO

        require(torch.cuda.is_available(), "CUDA unavailable")
        release_commit = repo / "RELEASE_COMMIT.txt"
        if (repo / ".git").exists():
            actual_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True
            ).stdout.strip()
        else:
            require(release_commit.is_file(), "Release has neither .git nor RELEASE_COMMIT.txt")
            actual_commit = release_commit.read_text(encoding="utf-8").strip()
        require(actual_commit == EXPECTED_COMMIT, f"Pinned commit mismatch: {actual_commit}")
        snapshots_before = snapshot_training_artifacts(group)
        best_val = load_best_val(group, snapshots_before)

        trash_manifest = repo / "manifests" / "trash_icra19" / "canonical_split_manifest.csv"
        trash_fingerprint, _, trash_errors = verify_dataset.inspect(
            trash_manifest, data_root=Path("/workspace/datasets/trash-icra19")
        )
        require(not trash_errors, f"Trash integrity errors: {trash_errors[:3]}")
        require(trash_fingerprint.get("dataset_sha256") == EXPECTED_DATASET, "Computed Trash dataset SHA mismatch")
        require(trash_fingerprint.get("split_manifest_sha256") == EXPECTED_SPLIT, "Computed Trash split SHA mismatch")
        require(trash_fingerprint.get("sequence_overlap") == 0, "Trash sequence overlap is not zero")
        require(trash_fingerprint.get("splits", {}).get("test", {}).get("images") == 1153, "Trash Test membership count mismatch")
        # The checker derives archive-relative provenance from REPO_ROOT and also
        # reads the canonical Trash file hashes there for the overlap gate.
        build_river_manifest.REPO_ROOT = river_archive.parent
        _, river_fingerprint, river_errors = build_river_manifest.inspect(river_archive)
        require(not river_errors, f"River integrity errors: {river_errors[:3]}")
        require(river_fingerprint.get("dataset_sha256") == EXPECTED_RIVER_SOURCE, "Computed River source SHA mismatch")
        require(river_fingerprint.get("evaluation_dataset_sha256") == EXPECTED_RIVER_EVAL, "Computed River evaluation SHA mismatch")
        require(river_fingerprint.get("evaluation_images") == 2185, "River evaluation membership count mismatch")
        atomic_json(out / "pre_evaluation_gate.json", {
            "status": "PASS", "all_model_statuses": "VERIFIED_COMPLETE",
            "dataset_sha256": trash_fingerprint["dataset_sha256"],
            "split_sha256": trash_fingerprint["split_manifest_sha256"],
            "test_images": trash_fingerprint["splits"]["test"]["images"],
            "sequence_overlap": trash_fingerprint["sequence_overlap"],
            "river_source_sha256": river_fingerprint["dataset_sha256"],
            "river_evaluation_sha256": river_fingerprint["evaluation_dataset_sha256"],
            "river_evaluation_images": river_fingerprint["evaluation_images"],
            "timestamp_utc": utc(),
        })

        config_path = group / "yolo" / "used_config.yaml"
        require(config_path.is_file(), f"Missing canonical config: {config_path}")
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg["out_dir"] = str(out)
        cfg["river_root"] = str(river_root)
        cfg["copy_files"] = False
        cfg.setdefault("run", {}).update({"prepare_data": True, "train_yolo": False, "train_frcnn": False, "train_ssd": False, "evaluate": True, "quick_debug": False})
        cfg.setdefault("training", {})["resume"] = False
        require(float(cfg["thresholds"]["confidence"]) == 0.25, "Confidence threshold mismatch")
        require(float(cfg["thresholds"]["iou_match"]) == 0.50, "IoU threshold mismatch")
        require(float(cfg["thresholds"]["class_agnostic_nms_iou"]) == 0.50, "River NMS threshold mismatch")
        require(int(cfg["seed"]) == 42, "Seed mismatch")
        atomic_json(out / "evaluation_config.json", cfg)

        paths = exp.prepare_datasets(cfg)
        trash_count = len(list((paths.trash_dir / "images" / "test").glob("*")))
        river_count = len(list((paths.river_dir / "images" / "test").glob("*"))) if paths.river_dir else 0
        require(trash_count == 1153, f"Canonical Test count mismatch: {trash_count}")
        require(river_count == 2185, f"Canonical River evaluation count mismatch: {river_count}")

        device = torch.device("cuda:0")
        best_paths = {model: Path(snapshots_before[model]["files"]["best.pt"]["path"]) for model in MODELS}
        yolo = YOLO(str(best_paths["yolo"]))
        yolo.to(device)
        frcnn = exp.build_faster_rcnn(len(CLASS_NAMES) + 1)
        frcnn_ckpt = torch.load(best_paths["frcnn"], map_location="cpu", weights_only=False)
        frcnn.load_state_dict(frcnn_ckpt["model"], strict=True)
        frcnn.to(device).eval()
        ssd = exp.build_ssdlite(len(CLASS_NAMES) + 1)
        ssd_ckpt = torch.load(best_paths["ssd"], map_location="cpu", weights_only=False)
        ssd.load_state_dict(ssd_ckpt["model"], strict=True)
        ssd.to(device).eval()
        atomic_json(out / "checkpoint_load_gate.json", {"status": "PASS", "models": list(MODELS), "timestamp_utc": utc()})

        conf, iou, nms_iou = 0.25, 0.50, 0.50
        test: dict[str, dict[str, object]] = {}
        river: dict[str, dict[str, object]] = {}

        ym, yfps, yms = exp.evaluate_yolo_model(yolo, paths.trash_dir, "test", 640, conf, iou, False, 3, "YOLO canonical Test")
        test["yolo"] = metrics_payload(ym, yfps, yms)

        fl = exp.make_loader(paths.trash_dir, "test", 640, 3, 8, 8, False)
        fm = exp.evaluate_torchvision_model(frcnn, fl, device, conf, iou, desc="Faster R-CNN canonical Test")
        ffl = exp.make_loader(paths.trash_dir, "test", 640, 3, 1, 8, False)
        ffps, fms = exp.measure_torchvision_fps(frcnn, ffl, device, warmup=20, samples=120)
        test["frcnn"] = metrics_payload(fm, ffps, fms)

        sl = exp.make_loader(paths.trash_dir, "test", 320, 3, 16, 8, False)
        sm = exp.evaluate_torchvision_model(ssd, sl, device, conf, iou, desc="SSDLite canonical Test")
        sfl = exp.make_loader(paths.trash_dir, "test", 320, 3, 1, 8, False)
        sfps, sms = exp.measure_torchvision_fps(ssd, sfl, device, warmup=20, samples=120)
        test["ssd"] = metrics_payload(sm, sfps, sms)
        atomic_json(out / "test_results.json", test)
        pd.DataFrame([{"model": model, **{k: v for k, v in row.items() if k != "per_class_ap_50_95"}} for model, row in test.items()]).to_csv(out / "test_results.csv", index=False)

        yrm, _, _ = exp.evaluate_yolo_model(yolo, paths.river_dir, "test", 640, conf, iou, True, 3, "YOLO River", class_agnostic_nms_iou=nms_iou)
        river["yolo"] = river_payload(yrm)
        frl = exp.make_loader(paths.river_dir, "test", 640, 1, 8, 8, False, class_agnostic=True)
        frm = exp.evaluate_torchvision_model(frcnn, frl, device, conf, iou, class_agnostic=True, desc="Faster R-CNN River", class_agnostic_nms_iou=nms_iou)
        river["frcnn"] = river_payload(frm)
        srl = exp.make_loader(paths.river_dir, "test", 320, 1, 16, 8, False, class_agnostic=True)
        srm = exp.evaluate_torchvision_model(ssd, srl, device, conf, iou, class_agnostic=True, desc="SSDLite River", class_agnostic_nms_iou=nms_iou)
        river["ssd"] = river_payload(srm)
        atomic_json(out / "river_results.json", river)
        pd.DataFrame([{"model": model, **row} for model, row in river.items()]).to_csv(out / "river_results.csv", index=False)

        environment = {
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "pytorch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
            "ultralytics": importlib.metadata.version("ultralytics"), "python": platform.python_version(),
            "commit": actual_commit, "device": str(device), "timestamp_utc": utc(),
            "fps_protocol": {"batch_size": 1, "warmup": 20, "timed_frames": 120, "preprocessing": "excluded (predecoded/resized)", "included": ["host_to_device", "forward", "postprocessing_and_nms"]},
        }
        atomic_json(out / "environment.json", environment)
        atomic_json(out / "best_validation.json", best_val)

        snapshots_after = snapshot_training_artifacts(group)
        require(snapshots_before == snapshots_after, "Training artifacts changed during evaluation")
        atomic_json(out / "training_artifact_immutability.json", {"status": "PASS", "before": snapshots_before, "after": snapshots_after})

        artifact_names = ["pre_evaluation_gate.json", "evaluation_config.json", "checkpoint_load_gate.json", "test_results.json", "test_results.csv", "river_results.json", "river_results.csv", "environment.json", "best_validation.json", "training_artifact_immutability.json", "dataset_summary.csv", "canonical_materialization_audit.json", "canonical_source_membership.json", "sequence_split_audit.csv"]
        artifacts = {}
        for name in artifact_names:
            path = out / name
            require(path.is_file() and path.stat().st_size > 0, f"Missing evaluation artifact: {name}")
            artifacts[name] = {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
        manifest = {
            "schema_version": 1, "status": "VERIFIED_COMPLETE", "evaluation_only": True,
            "training_performed": False, "checkpoints": "best.pt only", "models": list(MODELS),
            "dataset_sha256": EXPECTED_DATASET, "split_sha256": EXPECTED_SPLIT,
            "river_source_sha256": EXPECTED_RIVER_SOURCE, "river_evaluation_sha256": EXPECTED_RIVER_EVAL,
            "thresholds": {"confidence": conf, "iou_match": iou, "river_class_agnostic_nms": nms_iou},
            "artifacts": artifacts, "completed_utc": utc(),
        }
        atomic_json(out / "evaluation_artifact_manifest.json", manifest)
        for name, item in artifacts.items():
            require(sha256(out / name) == item["sha256"], f"Final artifact SHA mismatch: {name}")
        atomic_json(status_path, {"status": "VERIFIED_COMPLETE", "return_code": 0, "ended_utc": utc(), "manifest": str(out / "evaluation_artifact_manifest.json")})
        print("FINAL_EVALUATION_VERIFIED_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        atomic_json(status_path, {"status": "FAILED", "return_code": 1, "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(), "ended_utc": utc()})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
