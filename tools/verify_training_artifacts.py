"""Verify and fingerprint canonical training artifacts without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty required artifact: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("yolo", "frcnn", "ssd"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg["experiment"]["seeds"] != [42] or cfg["training"]["resume"] is not False:
        raise RuntimeError("Artifact verifier accepts only clean canonical seed-42 configs")
    if cfg["run"]["evaluate"] is not False or cfg["run"]["quick_debug"] is not False:
        raise RuntimeError("Official training config must disable evaluation and quick_debug")

    run_dir = out_dir / "runs" / "seed_42"
    artifacts: dict[str, object] = {
        "schema_version": 1,
        "model": args.model,
        "experiment_id": out_dir.name,
        "config": require_file(config_path),
        "provenance": cfg["provenance"],
        "checkpoints": {},
    }
    if args.model == "yolo":
        from ultralytics import YOLO

        checkpoint_dir = run_dir / "yolo" / "yolov8s_stage2" / "weights"
        for name in ("best.pt", "last.pt"):
            path = checkpoint_dir / name
            artifacts["checkpoints"][name] = require_file(path)
            loaded = YOLO(str(path))
            if getattr(loaded, "model", None) is None:
                raise RuntimeError(f"Ultralytics could not load {path}")
    else:
        checkpoint_dir = run_dir / "torchvision" / args.model
        required_last_keys = {
            "model", "optimizer", "scheduler", "scaler", "epoch", "stage",
            "stage_epoch", "architecture", "cfg",
        }
        for name in ("best.pt", "last.pt"):
            path = checkpoint_dir / name
            artifacts["checkpoints"][name] = require_file(path)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if "model" not in checkpoint or not checkpoint["model"]:
                raise RuntimeError(f"Checkpoint has no model state: {path}")
            if name == "last.pt":
                missing = required_last_keys - set(checkpoint)
                if missing:
                    raise RuntimeError(f"Last checkpoint is not resumable; missing {sorted(missing)}")
                if checkpoint["architecture"] != args.model:
                    raise RuntimeError(f"Architecture mismatch in {path}")
                recorded = checkpoint["cfg"].get("provenance", {})
                if recorded != cfg["provenance"]:
                    raise RuntimeError(f"Provenance mismatch in {path}")

    manifest_path = out_dir / "training_artifact_manifest.json"
    manifest_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(f"VERIFIED {args.model} artifacts: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
