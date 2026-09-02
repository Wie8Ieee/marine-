"""Kaggle GPU smoke check for the canonical Seed-42 pipeline.

This kernel is intentionally excluded from official comparison: it trains on a
small quick-debug subset only and never evaluates Trash Test or River.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


REPOSITORY = "https://github.com/Wie8Ieee/marine-.git"
COMMIT = "1cbcfcd6"
INPUT_ROOT = Path("/kaggle/input/marine-trash-icra19-canonical-seed42")
WORK_ROOT = Path("/kaggle/working/marine_canonical_smoke")


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    if not INPUT_ROOT.is_dir():
        raise RuntimeError(f"Canonical Kaggle dataset is not mounted: {INPUT_ROOT}")
    if not (INPUT_ROOT / "images").is_dir() or not (INPUT_ROOT / "labels").is_dir():
        raise RuntimeError("Canonical Kaggle dataset must contain images/ and labels/")

    repo = WORK_ROOT / "repository"
    run("git", "clone", "--depth", "1", REPOSITORY, str(repo))
    run("git", "-C", str(repo), "fetch", "--depth", "1", "origin", COMMIT)
    run("git", "-C", str(repo), "checkout", "--detach", COMMIT)
    run(sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))

    config = yaml.safe_load((repo / "config_runpod_smoke.yaml").read_text(encoding="utf-8"))
    config["trash_root"] = str(INPUT_ROOT)
    config["out_dir"] = str(WORK_ROOT / "output")
    config["river_root"] = None
    runtime_config = WORK_ROOT / "smoke_runtime_config.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    subprocess.run(
        [sys.executable, str(repo / "marine_3model_experiment.py"), "--config", str(runtime_config), "--preflight-only"],
        check=True, cwd=repo, env=env,
    )
    subprocess.run(
        [sys.executable, str(repo / "marine_3model_experiment.py"), "--config", str(runtime_config)],
        check=True, cwd=repo, env=env,
    )
    (WORK_ROOT / "EXCLUDED_FROM_OFFICIAL_COMPARISON.txt").write_text(
        "quick_debug=true; smoke run only; no Trash Test or River evaluation.\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
