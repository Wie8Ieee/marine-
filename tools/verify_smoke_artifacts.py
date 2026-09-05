"""Verify non-canonical smoke artifacts without weakening the canonical verifier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


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
    checkpoint = torch.load(last, map_location="cpu", weights_only=True)
    needed = {"model", "optimizer", "scheduler", "scaler", "completed_epoch", "next_epoch", "best_map", "best_epoch", "checkpoint_identity", "training_config_sha256", "dataset_sha256", "split_sha256", "git_commit", "seed", "experiment_id", "training_history"}
    missing = sorted(needed - set(checkpoint))
    if missing:
        raise RuntimeError(f"Smoke checkpoint missing required state: {missing}")
    history = run_dir / "history.csv"
    manifest["history"] = required(history)
    for name in ("results_overall_test.csv", "results_cross_domain.csv"):
        path = args.out_dir / name
        if path.exists() and path.stat().st_size > 1:
            raise RuntimeError(f"Smoke unexpectedly produced evaluation results: {path}")
    manifest.update({
        "status": "SMOKE_VERIFIED_FOR_PIPELINE_ONLY",
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
