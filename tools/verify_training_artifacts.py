"""Fail-closed, framework-aware verification of continuous training artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import atomic_json, load_and_validate_config, sha256_file, training_config_sha256  # noqa: E402


def required(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty artifact: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def finite_csv(path: Path, rows: int) -> pd.DataFrame:
    required(path)
    frame = pd.read_csv(path)
    if len(frame) != rows:
        raise RuntimeError(f"History row count mismatch: {path}: {len(frame)} != {rows}")
    numeric = frame.select_dtypes(include="number")
    if numeric.empty or not numeric.map(math.isfinite).all().all():
        raise RuntimeError(f"History contains non-finite values: {path}")
    return frame


def ensure_no_evaluation(out_dir: Path) -> None:
    forbidden = (
        "results_overall_test.csv", "results_cross_domain.csv", "fps_results.json",
        "domain_feature_shift.csv", "river_excluded_conflicting_duplicates.csv",
    )
    found = [str(path) for name in forbidden for path in out_dir.rglob(name) if path.is_file()]
    found.extend(str(path) for part in ("qualitative_errors", "river_trash_class_agnostic")
                 for path in out_dir.rglob(part) if path.exists())
    if found:
        raise RuntimeError(f"Training-only run contains evaluation output: {found}")


def verify_yolo(out_dir: Path, expected_rows: tuple[int, int]) -> tuple[dict, dict]:
    from ultralytics import YOLO
    root = out_dir / "runs/seed_42/yolo"
    stage1, stage2 = root / "yolov8s_stage1", root / "yolov8s_stage2"
    f1 = finite_csv(stage1 / "results.csv", expected_rows[0])
    f2 = finite_csv(stage2 / "results.csv", expected_rows[1])
    transition_path = stage2 / "stage2_initialization.json"
    required(transition_path)
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    stage1_last = stage1 / "weights" / "last.pt"
    if transition.get("status") != "STAGE2_INITIALIZED_FROM_STAGE1_LAST":
        raise RuntimeError("YOLO Stage 2 was not initialized from the final Stage-1 state")
    if transition.get("source_checkpoint_sha256") != required(stage1_last)["sha256"]:
        raise RuntimeError("YOLO Stage-1-to-Stage-2 checkpoint SHA-256 mismatch")
    selection_path = stage2 / "checkpoint_selection.json"
    required(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED_BY_VALIDATION_MAP50_95" or selection.get("stage") != "stage2":
        raise RuntimeError("YOLO Stage-2 checkpoint selection contract is invalid")
    checkpoints = {}
    checkpoint_payloads = {}
    for name in ("best.pt", "last.pt"):
        path = stage2 / "weights" / name
        checkpoints[name] = required(path)
        loaded = YOLO(str(path))
        if getattr(loaded, "model", None) is None:
            raise RuntimeError(f"Ultralytics could not load {path}")
        payload = getattr(loaded, "ckpt", None)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Ultralytics checkpoint metadata missing: {path}")
        train_args = payload.get("train_args", {})
        expected = {"seed": 42, "imgsz": 640, "batch": 16, "epochs": expected_rows[1]}
        for key, value in expected.items():
            if train_args.get(key) != value:
                raise RuntimeError(f"YOLO Stage-2 checkpoint {key} mismatch: {train_args.get(key)!r} != {value!r}")
        if train_args.get("name") != "yolov8s_stage2":
            raise RuntimeError("YOLO checkpoint does not identify the current Stage-2 run")
        checkpoint_payloads[name] = payload
    if selection.get("best_checkpoint_sha256") != checkpoints["best.pt"]["sha256"]:
        raise RuntimeError("YOLO selected best checkpoint SHA-256 mismatch")
    if selection.get("last_checkpoint_sha256") != checkpoints["last.pt"]["sha256"]:
        raise RuntimeError("YOLO Stage-2 last checkpoint SHA-256 mismatch")
    selected_index = int(selection.get("selected_epoch_index", -1))
    if not 0 <= selected_index < len(f2):
        raise RuntimeError("YOLO selected epoch index is outside Stage-2 history")
    columns = [str(column).strip() for column in f2.columns if "mAP50-95" in str(column)]
    if len(columns) != 1 or selected_index != int(pd.to_numeric(f2[columns[0]], errors="coerce").idxmax()):
        raise RuntimeError("YOLO best.pt was not selected by maximum validation mAP@0.5:0.95")
    source = stage2 / "weights" / str(selection.get("source_checkpoint", ""))
    if selection.get("source_checkpoint_sha256") != required(source)["sha256"]:
        raise RuntimeError("YOLO selected epoch source SHA-256 mismatch")
    if int(checkpoint_payloads["best.pt"].get("epoch", -1)) != selected_index:
        raise RuntimeError("YOLO best.pt epoch metadata does not match selection")
    if int(checkpoint_payloads["last.pt"].get("epoch", -1)) != expected_rows[1] - 1:
        raise RuntimeError("YOLO last.pt does not represent the final Stage-2 epoch")
    return checkpoints, {"stage1_rows": len(f1), "stage2_rows": len(f2), "selection": selection}


def verify_torchvision(model: str, out_dir: Path, cfg: dict, expected_commit: str, expected_rows: int) -> tuple[dict, dict]:
    root = out_dir / f"runs/seed_42/torchvision/{model}"
    history = finite_csv(root / "history.csv", expected_rows)
    expected_head = 10 if expected_rows == 110 else 1
    if history.iloc[:expected_head]["stage"].tolist() != ["head"] * expected_head:
        raise RuntimeError("Head-stage history is invalid")
    if history.iloc[expected_head:]["stage"].tolist() != ["all"] * (expected_rows - expected_head):
        raise RuntimeError("Full fine-tuning history is invalid")
    checkpoints, loaded = {}, {}
    for name in ("best.pt", "last.pt"):
        path = root / name
        checkpoints[name] = required(path)
        loaded[name] = torch.load(path, map_location="cpu", weights_only=False)
        if not loaded[name].get("model"):
            raise RuntimeError(f"Checkpoint has no model state: {path}")
    last = loaded["last.pt"]
    needed = {"model", "optimizer", "scheduler", "scaler", "epoch", "completed_epoch", "next_epoch", "stage", "stage_epoch", "architecture", "cfg", "best_epoch", "training_history", "training_config_sha256", "dataset_sha256", "split_sha256", "git_commit", "seed", "experiment_id", "python_random_state", "numpy_random_state", "torch_cpu_rng_state", "torch_cuda_rng_states", "dataloader_generator_state", "sampler_state"}
    missing = sorted(needed - set(last))
    if missing:
        raise RuntimeError(f"Last checkpoint is incomplete: {missing}")
    if last["architecture"] != model or int(last["seed"]) != 42:
        raise RuntimeError("Checkpoint architecture or seed mismatch")
    if int(last["completed_epoch"]) != expected_rows or len(last["training_history"]) != expected_rows:
        raise RuntimeError("TorchVision checkpoint is not training-complete")
    final_stage_epoch = 100 if expected_rows == 110 else 1
    if last["stage"] != "all" or int(last["stage_epoch"]) != final_stage_epoch:
        raise RuntimeError("TorchVision final stage is incomplete")
    if last["git_commit"] != expected_commit:
        raise RuntimeError("Checkpoint source commit mismatch")
    if last["training_config_sha256"] != training_config_sha256(cfg):
        raise RuntimeError("Checkpoint canonical configuration hash mismatch")
    provenance = cfg["provenance"]
    if last["dataset_sha256"] != provenance["dataset_sha256"] or last["split_sha256"] != provenance["split_manifest_sha256"]:
        raise RuntimeError("Checkpoint dataset/split fingerprint mismatch")
    maps = pd.to_numeric(history["map"], errors="coerce")
    if int(last["best_epoch"]) != int(history.loc[maps.idxmax(), "epoch"]):
        raise RuntimeError("Best epoch is inconsistent with validation mAP@0.5:0.95")
    if int(loaded["best.pt"].get("epoch", -1)) != int(last["best_epoch"]):
        raise RuntimeError("best.pt does not belong to the selected best epoch")
    if training_config_sha256(loaded["best.pt"].get("cfg", {})) != training_config_sha256(cfg):
        raise RuntimeError("best.pt configuration mismatch")
    return checkpoints, {"history_rows": len(history), "best_epoch": int(last["best_epoch"])}


def verify(model: str, out_dir: Path, config_path: Path, expected_commit: str, preflight: bool = False, execution_record: Path | None = None) -> dict:
    cfg = load_and_validate_config(config_path, model, allow_preflight=preflight)
    expected_rows = 2 if preflight else 110
    evidence = {name: required(out_dir / name) for name in (
        "used_config.yaml", "system_details.json", "python_environment.txt", "experiment_protocol.json",
        "environment.txt", "canonical_source_membership.json", "canonical_materialization_audit.json",
    )}
    source_membership = json.loads((out_dir / "canonical_source_membership.json").read_text(encoding="utf-8"))
    materialization = json.loads((out_dir / "canonical_materialization_audit.json").read_text(encoding="utf-8"))
    if source_membership.get("status") != "PASS" or any(int(source_membership.get(key, -1)) != 0 for key in (
        "missing_images", "extra_images", "missing_labels", "extra_labels", "sequence_overlap",
    )):
        raise RuntimeError("Canonical source membership evidence is invalid")
    if materialization.get("status") != "PASS" or any(int(materialization.get(key, -1)) != 0 for key in (
        "images_moved_from_canonical_split", "sequences_moved_from_canonical_split", "sequence_overlap",
        "missing_materialized_images", "extra_materialized_images",
    )):
        raise RuntimeError("Canonical materialized split evidence is invalid")
    used = yaml.safe_load((out_dir / "used_config.yaml").read_text(encoding="utf-8"))
    if training_config_sha256(used) != training_config_sha256(cfg):
        raise RuntimeError("Resolved configuration does not match requested configuration")
    system = json.loads((out_dir / "system_details.json").read_text(encoding="utf-8"))
    if system.get("git_commit") != expected_commit:
        raise RuntimeError("Runtime source commit mismatch")
    if execution_record:
        evidence["execution"] = required(execution_record)
        if json.loads(execution_record.read_text(encoding="utf-8")).get("return_code") != 0:
            raise RuntimeError("Training subprocess did not exit successfully")
    ensure_no_evaluation(out_dir)
    checkpoints, details = (verify_yolo(out_dir, (1, 1) if preflight else (10, 100)) if model == "yolo" else verify_torchvision(model, out_dir, cfg, expected_commit, expected_rows))
    report = {"schema_version": 2, "status": "VERIFIED_COMPLETE", "execution_mode": "gpu_preflight" if preflight else "continuous_official_training", "exact_resume_comparison": "NOT_APPLICABLE", "model": model, "source_commit": expected_commit, "config": required(config_path), "provenance": cfg["provenance"], "evidence": evidence, "checkpoints": checkpoints, "details": details}
    atomic_json(out_dir / "training_artifact_manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("yolo", "frcnn", "ssd"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execution-record", type=Path)
    args = parser.parse_args()
    verify(args.model, args.out_dir.resolve(), args.config.resolve(), args.expected_commit, args.preflight, args.execution_record.resolve() if args.execution_record else None)
    print(f"VERIFIED {args.model}: VERIFIED_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
