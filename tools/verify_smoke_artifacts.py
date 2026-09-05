"""Verify non-canonical smoke artifacts without weakening the canonical verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marine_3model_experiment import config_sha256


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def required(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty smoke artifact: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not cfg.get("run", {}).get("quick_debug") or cfg.get("run", {}).get("evaluate") is not False:
        raise RuntimeError("Smoke verifier accepts only quick_debug=true and evaluate=false runs")
    run_dir = args.out_dir / "runs" / "seed_42" / "torchvision" / "frcnn"
    best, last = run_dir / "best.pt", run_dir / "last.pt"
    manifest = {"best.pt": required(best), "last.pt": required(last), "environment": required(args.environment)}
    # Load both artifacts. A non-empty checkpoint alone is not evidence that it can be resumed.
    torch.load(best, map_location="cpu", weights_only=False)
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    needed = {
        "model", "optimizer", "scheduler", "scaler", "completed_epoch", "next_epoch",
        "best_map", "best_epoch", "checkpoint_identity", "training_config_sha256",
        "dataset_sha256", "split_sha256", "git_commit", "seed", "experiment_id",
        "training_history", "python_random_state", "numpy_random_state", "torch_cpu_rng_state",
        "torch_cuda_rng_states", "dataloader_generator_state", "sampler_state",
    }
    missing = sorted(needed - set(checkpoint))
    if missing:
        raise RuntimeError(f"Smoke checkpoint missing required state: {missing}")
    history = run_dir / "history.csv"
    manifest["history"] = required(history)
    history_df = pd.read_csv(history)
    epochs = history_df["epoch"].astype(int).tolist()
    expected_epochs = list(range(1, int(checkpoint["completed_epoch"]) + 1))
    if epochs != expected_epochs or len(epochs) != len(set(epochs)):
        raise RuntimeError(f"Smoke history is not continuous: {epochs}")
    if int(checkpoint["next_epoch"]) != int(checkpoint["completed_epoch"]) + 1:
        raise RuntimeError("Smoke checkpoint next_epoch is inconsistent with completed_epoch")
    if "lr" not in history_df or history_df["lr"].isna().any():
        raise RuntimeError("Smoke history is missing learning-rate continuity evidence")
    if checkpoint["training_config_sha256"] != config_sha256(cfg):
        raise RuntimeError("Smoke checkpoint training_config_sha256 does not match runtime configuration")
    if int(checkpoint["seed"]) != 42:
        raise RuntimeError("Smoke checkpoint seed is not 42")
    if bool(cfg.get("run", {}).get("resume_smoke_test", False)):
        if bool(cfg.get("canonical", False)) or bool(cfg.get("experiment", {}).get("canonical", False)):
            raise RuntimeError("BLOCKED — RESUME SMOKE CANNOT BE CANONICAL")
        comparison = args.out_dir / "resume_smoke_comparison.json"
        manifest["resume_smoke_comparison"] = required(comparison)
        comparison_data = json.loads(comparison.read_text(encoding="utf-8"))
        if comparison_data.get("status") != "RESUME_SMOKE_COMPARISON_PASS":
            raise RuntimeError("Resume smoke comparison did not pass")
    for name in ("results_overall_test.csv", "results_cross_domain.csv"):
        path = args.out_dir / name
        if path.exists() and path.stat().st_size > 1:
            raise RuntimeError(f"Smoke unexpectedly produced evaluation results: {path}")
    manifest.update({
        "status": "SMOKE_VERIFIED_FOR_PIPELINE_ONLY",
        "classification": "SMOKE_DEBUG_ONLY",
        "research_eligibility": "NOT_ELIGIBLE_FOR_RESEARCH_RESULTS",
        "canonical_comparison_eligibility": "NOT_ELIGIBLE_FOR_CANONICAL_COMPARISON",
    })
    target = args.out_dir / "smoke_artifact_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("SMOKE_VERIFIED_FOR_PIPELINE_ONLY")
    print("NOT_ELIGIBLE_FOR_RESEARCH_RESULTS")
    print("NOT_ELIGIBLE_FOR_CANONICAL_COMPARISON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
