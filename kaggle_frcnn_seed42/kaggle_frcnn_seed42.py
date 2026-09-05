"""Official canonical Faster R-CNN Seed-42 training on Kaggle GPU."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

REPOSITORY = "https://github.com/Wie8Ieee/marine-.git"
REQUIRED_ANCESTOR = "70d02255"
TRAINING_COMMIT = "b2d2d1bf16a37ec43669488536a9b512305bc181"
CONFIG_NAME = "config_runpod_frcnn_seed42.yaml"
MODEL = "frcnn"
EXPERIMENT_ID = f"frcnn_resnet50fpn_seed42_canonical_clean_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
WORK_ROOT = Path("/kaggle/working") / EXPERIMENT_ID


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def data_root() -> Path:
    roots = [p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one mounted Trash dataset, found {roots}")
    return roots[0]


def environment_record(runtime: Path, cfg: dict, git_commit: str) -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True)
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "running",
        "start_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "git_commit": git_commit,
        "config_sha256": sha256(runtime),
        "dataset_sha256": cfg["provenance"]["dataset_sha256"],
        "split_manifest_sha256": cfg["provenance"]["split_manifest_sha256"],
        "gpu": gpu.stdout.strip() if gpu.returncode == 0 else "unavailable",
    }


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=False)
    repo = WORK_ROOT / "repository"
    runtime = WORK_ROOT / "official_runtime_config.yaml"
    status_path = WORK_ROOT / "run_status.json"
    try:
        run("git", "clone", REPOSITORY, str(repo))
        run("git", "-C", str(repo), "checkout", "--detach", TRAINING_COMMIT)
        run("git", "-C", str(repo), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD")
        git_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if git_commit != TRAINING_COMMIT:
            raise RuntimeError(f"Pinned commit mismatch: expected {TRAINING_COMMIT}, got {git_commit}")
        run(sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))

        cfg = yaml.safe_load((repo / CONFIG_NAME).read_text(encoding="utf-8"))
        cfg["trash_root"] = str(data_root())
        cfg["out_dir"] = str(WORK_ROOT / "output")
        runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        record = environment_record(runtime, cfg, git_commit)
        (WORK_ROOT / "environment_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        status_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CANONICAL_GIT_COMMIT=git_commit, CANONICAL_EXPERIMENT_ID=EXPERIMENT_ID)
        run(sys.executable, str(repo / "marine_3model_experiment.py"), "--config", str(runtime), "--preflight-only")
        subprocess.run([sys.executable, str(repo / "marine_3model_experiment.py"), "--config", str(runtime)], check=True, cwd=repo, env=env)
        run(sys.executable, str(repo / "tools/verify_training_artifacts.py"), "--model", MODEL, "--out-dir", str(WORK_ROOT / "output"), "--config", str(runtime))
        record.update(status="completed", end_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        status_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        (WORK_ROOT / "OFFICIAL_SEED42_RUN.txt").write_text(
            f"model=frcnn\nseed=42\nresume=false\nevaluate=false\nexperiment_id={EXPERIMENT_ID}\ngit_commit={git_commit}\n",
            encoding="utf-8",
        )
    except Exception:
        record = {"experiment_id": EXPERIMENT_ID, "status": "failed", "end_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "traceback": traceback.format_exc()}
        status_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
