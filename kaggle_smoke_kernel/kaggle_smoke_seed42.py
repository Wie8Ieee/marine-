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
REQUIRED_ANCESTOR = "70d02255"
INPUT_ROOTS = (
    Path("/kaggle/input/marine-trash-icra19-canonical-seed42"),
    Path("/kaggle/input/marine-trash-icra19-canonical-seed-42"),
)
WORK_ROOT = Path("/kaggle/working/marine_canonical_smoke")


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def canonical_input_root() -> Path:
    for candidate in INPUT_ROOTS:
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate
    input_directory = Path("/kaggle/input")
    candidates = [
        candidate for candidate in input_directory.rglob("*")
        if candidate.is_dir() and (candidate / "images").is_dir() and (candidate / "labels").is_dir()
    ] if input_directory.is_dir() else []
    if len(candidates) == 1:
        return candidates[0]
    mounted = sorted(str(candidate.relative_to(input_directory)) for candidate in input_directory.rglob("*") if candidate.is_dir()) if input_directory.is_dir() else []
    raise RuntimeError(
        "Canonical Kaggle dataset is not mounted with images/ and labels/. "
        f"Checked: {', '.join(map(str, INPUT_ROOTS))}; mounted roots: {mounted}"
    )


def main() -> None:
    input_root = canonical_input_root()

    repo = WORK_ROOT / "repository"
    run("git", "clone", "--branch", "main", REPOSITORY, str(repo))
    run("git", "-C", str(repo), "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, "HEAD")
    run("git", "-C", str(repo), "rev-parse", "HEAD")
    run(sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))

    config = yaml.safe_load((repo / "config_runpod_smoke.yaml").read_text(encoding="utf-8"))
    config["trash_root"] = str(input_root)
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
