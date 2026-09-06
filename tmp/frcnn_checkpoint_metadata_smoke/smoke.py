"""Non-canonical two-process exact-resume smoke for Faster R-CNN."""
from __future__ import annotations

import datetime as dt
import atexit
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

COMMIT = "4078fc18c1a20798cd7a6a8a7a7d533411568e50"
REPOSITORY = "https://github.com/Wie8Ieee/marine-.git"
ROOT = Path("/kaggle/working") / f"frcnn_exact_resume_smoke_{COMMIT[:8]}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
DIAGNOSTICS = Path("/kaggle/working/smoke_diagnostics") / ROOT.name


def marker(stage: str, state: str, **details: object) -> None:
    """Append a flushed stage marker before/after every launcher boundary."""
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "state": state, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **details}
    with (DIAGNOSTICS / "stage_markers.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, default=str) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def unhandled(exc_type, exc, tb):
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    (DIAGNOSTICS / "unhandled_exception.json").write_text(json.dumps({"error_type": exc_type.__name__, "error_message": str(exc)}, indent=2) + "\n", encoding="utf-8")
    marker("launcher", "UNHANDLED_EXCEPTION")
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = unhandled


def run(*args: str, **kwargs) -> None:
    marker("launcher_command", "START", command=list(args))
    with (DIAGNOSTICS / "launcher_stdout.log").open("a", encoding="utf-8") as stdout, (DIAGNOSTICS / "launcher_stderr.log").open("a", encoding="utf-8") as stderr:
        completed = subprocess.run(args, check=False, stdout=stdout, stderr=stderr, text=True, **kwargs)
    marker("launcher_command", "END", return_code=completed.returncode)
    completed.check_returncode()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_root() -> Path:
    roots = [p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one mounted Trash dataset, found {roots}")
    return roots[0]


def config_for(base: dict, out_dir: Path, resume: bool, session_id: str, stop_after: int, checkpoint: Path | None = None) -> dict:
    cfg = json.loads(json.dumps(base))
    cfg.update({"canonical": False, "trash_root": str(data_root()), "river_root": None, "out_dir": str(out_dir)})
    cfg["training"].update({"resume": resume, "workers": 0})
    cfg["run"].update({"quick_debug": True, "resume_smoke_test": True, "evaluate": False, "train_yolo": False, "train_frcnn": True, "train_ssd": False})
    cfg["resume_smoke"] = {"stage1_smoke_epochs": 1, "stage2_smoke_total_epochs": 2, "session_a_stop_after_stage2_epoch": 1, "session_b_resume_until_stage2_epoch": 2, "workers": 0, "classification": "SMOKE_DEBUG_ONLY"}
    cfg["session_control"] = {"session_id": session_id, "stop_after_stage2_epoch": stop_after}
    if checkpoint:
        cfg["session_control"].update({"resume_checkpoint": str(checkpoint), "resume_checkpoint_sha256": sha256(checkpoint)})
    return cfg


def write_config(name: str, cfg: dict) -> Path:
    path = ROOT / f"{name}_runtime_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def execute(repo: Path, config: Path, env: dict[str, str], name: str, diagnostics_root: Path | None = None) -> None:
    print(f"START_{name}", flush=True)
    command = [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(config)]
    if diagnostics_root is None:
        run(*command, cwd=repo, env=env)
        return
    sys.path.insert(0, str(repo / "tools"))
    from smoke_process_diagnostics import run_logged_process, write_stage_marker
    mirror_root = DIAGNOSTICS
    write_stage_marker(mirror_root, "process_b", "STARTING", command=command)
    try:
        run_logged_process(command, cwd=repo, env=env, output_root=diagnostics_root, mirror_root=mirror_root, label="process_b")
    except Exception as exc:
        write_stage_marker(mirror_root, "process_b", "FAILED", error_type=type(exc).__name__, error_message=str(exc))
        raise
    write_stage_marker(mirror_root, "process_b", "COMPLETE")


def summary(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sched, opt = ckpt["scheduler"], ckpt["optimizer"]
    return {"epochs": [int(r["epoch"]) for r in ckpt["training_history"]], "lrs": [float(r["lr"]) for r in ckpt["training_history"]], "best_epoch": int(ckpt["best_epoch"]), "best_map": float(ckpt["best_map"]), "completed_epoch": int(ckpt["completed_epoch"]), "next_epoch": int(ckpt["next_epoch"]), "optimizer_groups": opt["param_groups"], "scheduler": {k: sched.get(k) for k in ("T_max", "last_epoch", "_step_count", "_last_lr")}, "checkpoint_sha256": sha256(path)}


ROOT.mkdir(parents=True, exist_ok=False)
marker("launcher", "START")
repo = ROOT / "repository"
marker("reference", "START")
run("git", "clone", REPOSITORY, str(repo))
run("git", "-C", str(repo), "fetch", "origin")
run("git", "-C", str(repo), "checkout", "--detach", COMMIT)
actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
print(actual, flush=True)
if actual != COMMIT:
    raise RuntimeError("BLOCKED — WRONG GIT COMMIT")
run(sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))
base = yaml.safe_load((repo / "config_runpod_frcnn_seed42.yaml").read_text(encoding="utf-8"))
experiment_id = ROOT.name
environment = ROOT / "environment.json"
environment.write_text(json.dumps({"smoke": True, "classification": "SMOKE_DEBUG_ONLY", "canonical": False, "experiment_id": experiment_id, "commit": actual, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"}, indent=2), encoding="utf-8")
env = dict(os.environ, CANONICAL_GIT_COMMIT=actual, CANONICAL_EXPERIMENT_ID=experiment_id, PYTHONDONTWRITEBYTECODE="1")

# One uninterrupted reference and a true fresh-process A -> B resume path.
ref_out, resume_out = ROOT / "reference_output", ROOT / "resume_output"
reference = write_config("reference", config_for(base, ref_out, False, "reference_uninterrupted", 2))
process_a = write_config("process_a", config_for(base, resume_out, False, "session_a_clean", 1))
execute(repo, reference, env, "REFERENCE")
marker("reference", "END")
marker("process_a", "START")
execute(repo, process_a, env, "PROCESS_A")
marker("process_a", "END")
print("PROCESS_A_SUBPROCESS_EXITED", flush=True)
sys.path.insert(0, str(repo))
from marine_3model_experiment import validate_session_a_contract
status_a = validate_session_a_contract(resume_out, expected_next_stage2_epoch=2)
marker("process_a_contract", "PASS")
last_a = Path(status_a["last_checkpoint"])
print("PROCESS_A_ARTIFACTS_ACCEPTED", flush=True)
print("READY_TO_START_PROCESS_B", flush=True)
process_b = write_config("process_b", config_for(base, resume_out, True, "session_b_resume", 2, last_a))
marker("process_b_configuration", "READY")
execute(repo, process_b, env, "PROCESS_B", diagnostics_root=resume_out)
marker("process_b", "END")

marker("reference_comparison", "START")
ref = summary(ref_out / "runs" / "seed_42" / "torchvision" / "frcnn" / "last.pt")
resumed = summary(resume_out / "runs" / "seed_42" / "torchvision" / "frcnn" / "last.pt")
checks = {"history_continuity": resumed["epochs"] == [1, 2, 3], "no_missing_or_duplicate_epochs": resumed["epochs"] == [1, 2, 3], "dataloader_epoch_seed_continuity": [42 + e for e in resumed["epochs"]] == [43, 44, 45], "learning_rate_continuity": resumed["lrs"] == ref["lrs"], "optimizer_progression": resumed["optimizer_groups"] == ref["optimizer_groups"], "scheduler_progression": resumed["scheduler"] == ref["scheduler"], "best_epoch_matches_reference": resumed["best_epoch"] == ref["best_epoch"], "best_metric_matches_reference": resumed["best_map"] == ref["best_map"], "next_epoch_correct": resumed["next_epoch"] == 4}
comparison = {"classification": "SMOKE_DEBUG_ONLY", "canonical": False, "reference": ref, "resumed": resumed, "checks": checks, "model_checkpoint_file_sha_equal": resumed["checkpoint_sha256"] == ref["checkpoint_sha256"], "note": "Checkpoint-byte equality is recorded but not required; CUDA operations may be nondeterministic.", "status": "RESUME_SMOKE_COMPARISON_PASS" if all(checks.values()) else "RESUME_SMOKE_COMPARISON_FAIL"}
(resume_out / "resume_smoke_comparison.json").write_text(json.dumps(comparison, indent=2, default=str) + "\n", encoding="utf-8")
if comparison["status"] != "RESUME_SMOKE_COMPARISON_PASS":
    raise RuntimeError("Exact-resume structural comparison failed")
marker("reference_comparison", "PASS")
marker("smoke_verifier", "START")
run(sys.executable, str(repo / "tools/verify_smoke_artifacts.py"), "--out-dir", str(resume_out), "--config", str(process_b), "--environment", str(environment), cwd=repo)
marker("smoke_verifier", "PASS")
(ROOT / "SMOKE_ONLY.json").write_text(json.dumps({"classification": "SMOKE_DEBUG_ONLY", "canonical": False, "research_eligibility": "NOT_ELIGIBLE_FOR_RESEARCH_RESULTS", "commit": actual, "experiment_id": experiment_id}, indent=2) + "\n", encoding="utf-8")
