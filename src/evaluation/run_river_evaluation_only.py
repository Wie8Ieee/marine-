#!/usr/bin/env python3
"""Complete only the failed River phase of an existing final evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
import torch
import yaml


def load_final_helpers(path: Path):
    spec = importlib.util.spec_from_file_location("final_evaluation_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load final-evaluation helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--group", type=Path, required=True)
    parser.add_argument("--river-root", type=Path, required=True)
    parser.add_argument("--river-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--final-evaluator", type=Path, required=True)
    args = parser.parse_args()
    repo, group, river_root, river_archive, out = (
        p.resolve() for p in (args.repo, args.group, args.river_root, args.river_archive, args.out)
    )
    helpers = load_final_helpers(args.final_evaluator.resolve())
    status_path = out / "evaluation_status.json"
    retry_status = out / "river_retry_status.json"
    before_test_hashes = {
        name: helpers.sha256(out / name) for name in ("test_results.json", "test_results.csv")
    }
    helpers.atomic_json(retry_status, {
        "status": "RUNNING", "river_only": True, "training_performed": False,
        "test_rerun": False, "fps_rerun": False, "started_utc": helpers.utc(),
        "test_hashes_before": before_test_hashes,
    })
    try:
        helpers.require(out.is_dir(), f"Existing evaluation directory missing: {out}")
        helpers.require(torch.cuda.is_available(), "CUDA unavailable")
        for name in ("test_results.json", "test_results.csv", "pre_evaluation_gate.json", "evaluation_config.json"):
            helpers.require((out / name).is_file(), f"Existing evaluation artifact missing: {name}")

        sys.path.insert(0, str(repo))
        import marine_3model_experiment as exp
        sys.path.insert(0, str(args.final_evaluator.resolve().parent / "tools"))
        import build_river_manifest
        import verify_dataset
        from ultralytics import YOLO

        release_commit = repo / "RELEASE_COMMIT.txt"
        actual_commit = (
            subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()
            if (repo / ".git").exists() else release_commit.read_text(encoding="utf-8").strip()
        )
        helpers.require(actual_commit == helpers.EXPECTED_COMMIT, f"Pinned commit mismatch: {actual_commit}")
        snapshots_before = helpers.snapshot_training_artifacts(group)

        trash_manifest = repo / "manifests" / "trash_icra19" / "canonical_split_manifest.csv"
        trash_fingerprint, _, trash_errors = verify_dataset.inspect(
            trash_manifest, data_root=Path("/workspace/datasets/trash-icra19")
        )
        helpers.require(not trash_errors, f"Trash integrity errors: {trash_errors[:3]}")
        helpers.require(trash_fingerprint.get("dataset_sha256") == helpers.EXPECTED_DATASET, "Computed Trash dataset SHA mismatch")
        helpers.require(trash_fingerprint.get("split_manifest_sha256") == helpers.EXPECTED_SPLIT, "Computed Trash split SHA mismatch")
        helpers.require(trash_fingerprint.get("sequence_overlap") == 0, "Trash sequence overlap is not zero")
        build_river_manifest.REPO_ROOT = river_archive.parent
        _, river_fingerprint, river_errors = build_river_manifest.inspect(river_archive)
        helpers.require(not river_errors, f"River integrity errors: {river_errors[:3]}")
        helpers.require(river_fingerprint.get("dataset_sha256") == helpers.EXPECTED_RIVER_SOURCE, "Computed River source SHA mismatch")
        helpers.require(river_fingerprint.get("evaluation_dataset_sha256") == helpers.EXPECTED_RIVER_EVAL, "Computed River evaluation SHA mismatch")

        cfg = json.loads((out / "evaluation_config.json").read_text(encoding="utf-8"))
        cfg["out_dir"] = str(out)
        cfg["river_root"] = str(river_root)
        cfg["copy_files"] = False
        cfg.setdefault("run", {}).update({
            "prepare_data": True, "train_yolo": False, "train_frcnn": False,
            "train_ssd": False, "evaluate": True, "quick_debug": False,
        })
        cfg.setdefault("training", {})["resume"] = False
        helpers.require(float(cfg["thresholds"]["confidence"]) == 0.25, "Confidence threshold mismatch")
        helpers.require(float(cfg["thresholds"]["iou_match"]) == 0.50, "IoU threshold mismatch")
        helpers.require(float(cfg["thresholds"]["class_agnostic_nms_iou"]) == 0.50, "River NMS threshold mismatch")
        paths = exp.prepare_datasets(cfg)
        helpers.require(paths.river_dir is not None, "River materialization missing")
        helpers.require(len(list((paths.river_dir / "images" / "test").glob("*"))) == 2185, "River membership count mismatch")

        device = torch.device("cuda:0")
        best_paths = {m: Path(snapshots_before[m]["files"]["best.pt"]["path"]) for m in helpers.MODELS}
        yolo = YOLO(str(best_paths["yolo"])); yolo.to(device)
        frcnn = exp.build_faster_rcnn(len(helpers.CLASS_NAMES) + 1)
        frcnn.load_state_dict(torch.load(best_paths["frcnn"], map_location="cpu", weights_only=False)["model"], strict=True)
        frcnn.to(device).eval()
        ssd = exp.build_ssdlite(len(helpers.CLASS_NAMES) + 1)
        ssd.load_state_dict(torch.load(best_paths["ssd"], map_location="cpu", weights_only=False)["model"], strict=True)
        ssd.to(device).eval()

        conf, iou, nms_iou = 0.25, 0.50, 0.50
        river = {}
        yrm, _, _ = exp.evaluate_yolo_model(yolo, paths.river_dir, "test", 640, conf, iou, True, 3, "YOLO River", class_agnostic_nms_iou=nms_iou)
        river["yolo"] = helpers.river_payload(yrm)
        frl = exp.make_loader(paths.river_dir, "test", 640, 1, 8, 8, False, class_agnostic=True)
        river["frcnn"] = helpers.river_payload(exp.evaluate_torchvision_model(frcnn, frl, device, conf, iou, class_agnostic=True, desc="Faster R-CNN River", class_agnostic_nms_iou=nms_iou))
        srl = exp.make_loader(paths.river_dir, "test", 320, 1, 16, 8, False, class_agnostic=True)
        river["ssd"] = helpers.river_payload(exp.evaluate_torchvision_model(ssd, srl, device, conf, iou, class_agnostic=True, desc="SSDLite River", class_agnostic_nms_iou=nms_iou))
        helpers.atomic_json(out / "river_results.json", river)
        pd.DataFrame([{"model": model, **row} for model, row in river.items()]).to_csv(out / "river_results.csv", index=False)

        environment = {
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
            "pytorch": torch.__version__, "torchvision": importlib.metadata.version("torchvision"),
            "ultralytics": importlib.metadata.version("ultralytics"), "python": platform.python_version(),
            "commit": actual_commit, "device": str(device), "timestamp_utc": helpers.utc(),
            "fps_protocol": {"batch_size": 1, "warmup": 20, "timed_frames": 120, "preprocessing": "excluded (predecoded/resized)", "included": ["host_to_device", "forward", "postprocessing_and_nms"]},
            "fps_reused_from_existing_test_results": True,
        }
        helpers.atomic_json(out / "environment.json", environment)
        helpers.atomic_json(out / "best_validation.json", helpers.load_best_val(group, snapshots_before))
        snapshots_after = helpers.snapshot_training_artifacts(group)
        helpers.require(snapshots_before == snapshots_after, "Training artifacts changed during River evaluation")
        after_test_hashes = {name: helpers.sha256(out / name) for name in before_test_hashes}
        helpers.require(before_test_hashes == after_test_hashes, "Existing Test/FPS artifacts changed")
        helpers.atomic_json(out / "training_artifact_immutability.json", {
            "status": "PASS", "before": snapshots_before, "after": snapshots_after,
            "test_hashes_before": before_test_hashes, "test_hashes_after": after_test_hashes,
        })

        artifact_names = ["pre_evaluation_gate.json", "evaluation_config.json", "checkpoint_load_gate.json", "test_results.json", "test_results.csv", "river_results.json", "river_results.csv", "environment.json", "best_validation.json", "training_artifact_immutability.json", "dataset_summary.csv", "canonical_materialization_audit.json", "canonical_source_membership.json", "sequence_split_audit.csv"]
        artifacts = {}
        for name in artifact_names:
            path = out / name
            helpers.require(path.is_file() and path.stat().st_size > 0, f"Missing evaluation artifact: {name}")
            artifacts[name] = {"size_bytes": path.stat().st_size, "sha256": helpers.sha256(path)}
        manifest = {
            "schema_version": 1, "status": "VERIFIED_COMPLETE", "evaluation_only": True,
            "training_performed": False, "test_rerun": False, "fps_rerun": False,
            "river_retry_after_numeric_boundary_fix": True, "checkpoints": "best.pt only",
            "models": list(helpers.MODELS), "dataset_sha256": helpers.EXPECTED_DATASET,
            "split_sha256": helpers.EXPECTED_SPLIT, "river_source_sha256": helpers.EXPECTED_RIVER_SOURCE,
            "river_evaluation_sha256": helpers.EXPECTED_RIVER_EVAL,
            "thresholds": {"confidence": conf, "iou_match": iou, "river_class_agnostic_nms": nms_iou},
            "artifacts": artifacts, "completed_utc": helpers.utc(),
        }
        helpers.atomic_json(out / "evaluation_artifact_manifest.json", manifest)
        for name, item in artifacts.items():
            helpers.require(helpers.sha256(out / name) == item["sha256"], f"Final artifact SHA mismatch: {name}")
        helpers.atomic_json(retry_status, {
            "status": "VERIFIED_COMPLETE", "return_code": 0, "river_only": True,
            "training_performed": False, "test_rerun": False, "fps_rerun": False,
            "test_hashes_before": before_test_hashes, "test_hashes_after": after_test_hashes,
            "ended_utc": helpers.utc(),
        })
        helpers.atomic_json(status_path, {"status": "VERIFIED_COMPLETE", "return_code": 0, "ended_utc": helpers.utc(), "manifest": str(out / "evaluation_artifact_manifest.json")})
        print("RIVER_ONLY_AND_FINAL_VERIFICATION_COMPLETE", flush=True)
        return 0
    except Exception as exc:
        helpers.atomic_json(retry_status, {
            "status": "FAILED", "return_code": 1, "river_only": True,
            "training_performed": False, "test_rerun": False, "fps_rerun": False,
            "error_type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(), "ended_utc": helpers.utc(),
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
