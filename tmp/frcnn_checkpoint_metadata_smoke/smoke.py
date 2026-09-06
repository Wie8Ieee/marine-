"""Version 7: failure-durable, non-canonical Faster exact-resume smoke."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path
import torch
import yaml

# Replaced with the committed Version 7 SHA immediately before the Kaggle push.
COMMIT = "PINNED_COMMIT_REPLACED_BEFORE_LAUNCH"
REPOSITORY = "https://github.com/Wie8Ieee/marine-.git"
WORKING = Path("/kaggle/working")
ROOT = WORKING / "frcnn_smoke_v8_export"
RUNTIME = WORKING / "frcnn_smoke_v8_runtime"
LAST_STAGE = "NOT_STARTED"
RESULT = {"smoke_status": "FAIL", "last_verified_stage": LAST_STAGE, "failed_stage": None, "exception_type": None, "message": None, "return_code": None}

def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str); stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)

def marker(name: str, **details: object) -> None:
    global LAST_STAGE
    payload = {"stage": name, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **details}
    with (ROOT / "stage_markers.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, default=str) + "\n"); stream.flush(); os.fsync(stream.fileno())
    LAST_STAGE = name

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def safe_command(command: list[str]) -> list[str]:
    return ["<redacted>" if any(key in part.lower() for key in ("token", "password", "api_key", "secret")) else part for part in command]

class ProcessFailure(RuntimeError):
    def __init__(self, stage: str, return_code: int) -> None:
        super().__init__(f"{stage} subprocess returned {return_code}")
        self.stage, self.return_code = stage, return_code


def execute(stage: str, command: list[str], cwd: Path, env: dict[str, str], directory: Path | None = None) -> None:
    """Stream every subprocess log to disk and raise only to main's safe handler."""
    directory = directory or ROOT / stage; directory.mkdir(parents=True, exist_ok=True)
    marker(f"{stage.upper()}_STARTED"); atomic_json(directory / "command.json", {"command": safe_command(command)})
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, text=True)
        code = process.wait(); stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
    execution = {"stage": stage, "command": safe_command(command), "pid": process.pid, "return_code": code, "start_timestamp_utc": started, "end_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic_json(directory / "execution.json", execution)
    if code:
        atomic_json(directory / "failure.json", {**execution, "message": f"subprocess returned {code}"})
        raise ProcessFailure(stage, code)
    marker(f"{stage.upper()}_COMPLETED", return_code=code)

def data_root() -> Path:
    roots = [p for p in Path("/kaggle/input").rglob("*") if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()]
    if len(roots) != 1: raise RuntimeError(f"Expected one mounted Trash dataset, found {roots}")
    return roots[0]

def config_for(base: dict, out_dir: Path, resume: bool, session_id: str, stop_after: int, checkpoint: Path | None = None) -> dict:
    cfg = json.loads(json.dumps(base)); cfg.update({"canonical": False, "trash_root": str(data_root()), "river_root": None, "out_dir": str(out_dir)})
    cfg["training"].update({"resume": resume, "workers": 0})
    cfg["run"].update({"quick_debug": True, "resume_smoke_test": True, "evaluate": False, "train_yolo": False, "train_frcnn": True, "train_ssd": False})
    cfg["resume_smoke"] = {"stage1_smoke_epochs": 1, "stage2_smoke_total_epochs": 2, "session_a_stop_after_stage2_epoch": 1, "session_b_resume_until_stage2_epoch": 2, "workers": 0, "classification": "SMOKE_DEBUG_ONLY"}
    cfg["session_control"] = {"session_id": session_id, "stop_after_stage2_epoch": stop_after}
    if checkpoint: cfg["session_control"].update({"resume_checkpoint": str(checkpoint), "resume_checkpoint_sha256": sha256(checkpoint)})
    return cfg

def write_config(name: str, config: dict) -> Path:
    path = ROOT / "configs" / f"{name}_runtime_config.yaml"
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False); stream.flush(); os.fsync(stream.fileno())
    return path

def checkpoint_summary(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {"epochs": [int(row["epoch"]) for row in ckpt["training_history"]], "lrs": [float(row["lr"]) for row in ckpt["training_history"]], "best_epoch": int(ckpt["best_epoch"]), "best_map": float(ckpt["best_map"]), "next_epoch": int(ckpt["next_epoch"]), "optimizer": ckpt["optimizer"]["param_groups"], "scheduler": {key: ckpt["scheduler"].get(key) for key in ("T_max", "last_epoch", "_step_count", "_last_lr")}, "sha256": sha256(path)}

def fail(stage: str, exc: BaseException, return_code: int | None = None) -> None:
    if isinstance(exc, ProcessFailure):
        stage, return_code = exc.stage, exc.return_code
    RESULT.update({"smoke_status": "FAIL", "last_verified_stage": LAST_STAGE, "failed_stage": stage, "exception_type": type(exc).__name__, "message": str(exc), "return_code": return_code})
    atomic_json(ROOT / "unhandled_exception.json", {"stage": stage, "error_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}); marker("SMOKE_FAIL", failed_stage=stage, error_type=type(exc).__name__)

def bundle() -> None:
    try:
        with zipfile.ZipFile(WORKING / "frcnn_smoke_v8_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for item in ROOT.rglob("*"):
                if item.is_file(): archive.write(item, item.relative_to(WORKING))
        marker("BUNDLE_CREATED")
    except Exception as exc:
        atomic_json(ROOT / "bundle_failure.json", {"error_type": type(exc).__name__, "message": str(exc)})

def main() -> bool:
    global LAST_STAGE
    for directory in (ROOT, ROOT / "reference", ROOT / "process_a", ROOT / "process_b", ROOT / "configs", ROOT / "diagnostics", RUNTIME): directory.mkdir(parents=True, exist_ok=True)
    marker("NOTEBOOK_STARTED"); marker("EXPORT_ROOT_CREATED"); repo = RUNTIME / "checkout"
    try:
        launcher_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        if repo.exists(): raise RuntimeError(f"CHECKOUT_CONTRACT_FAILED — checkout already exists: {repo}")
        execute("repository_clone", ["git", "clone", REPOSITORY, str(repo)], ROOT, launcher_env, ROOT / "diagnostics" / "repository_clone")
        execute("repository_fetch", ["git", "-C", str(repo), "fetch", "origin"], ROOT, launcher_env, ROOT / "diagnostics" / "repository_fetch")
        execute("repository_checkout", ["git", "-C", str(repo), "checkout", "--detach", COMMIT], ROOT, launcher_env, ROOT / "diagnostics" / "repository_checkout")
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT: raise RuntimeError("BLOCKED — WRONG GIT COMMIT")
        marker("REPOSITORY_CLONED"); marker("COMMIT_VERIFIED", commit=actual)
        execute("dependencies", [sys.executable, "-u", "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], repo, launcher_env)
        base = yaml.safe_load((repo / "config_runpod_frcnn_seed42.yaml").read_text(encoding="utf-8")); experiment_id = f"frcnn_exact_resume_smoke_v8_{actual[:8]}"
        environment = ROOT / "environment.json"; atomic_json(environment, {"smoke": True, "classification": "SMOKE_DEBUG_ONLY", "canonical": False, "experiment_id": experiment_id, "commit": actual, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"})
        env = dict(launcher_env, CANONICAL_GIT_COMMIT=actual, CANONICAL_EXPERIMENT_ID=experiment_id)
        ref_out, resume_out = ROOT / "reference", ROOT / "process_a"
        reference = write_config("reference", config_for(base, ref_out, False, "reference_uninterrupted", 2)); process_a = write_config("process_a", config_for(base, resume_out, False, "session_a_clean", 1)); marker("CONFIGS_CREATED")
        execute("reference", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(reference)], repo, env); marker("REFERENCE_COMPLETED")
        execute("process_a", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(process_a)], repo, env); marker("PROCESS_A_COMPLETED")
        marker("PROCESS_A_CONTRACT_STARTED"); sys.path.insert(0, str(repo)); from marine_3model_experiment import validate_session_a_contract
        status = validate_session_a_contract(resume_out, expected_next_stage2_epoch=2); last = Path(status["last_checkpoint"])
        if not last.is_file() or not last.stat().st_size or sha256(last) != status["last_checkpoint_sha256"]: raise RuntimeError("PROCESS_A_CONTRACT_FAIL")
        checkpoint_summary(last); marker("PROCESS_A_CONTRACT_PASSED")
        process_b = write_config("process_b", config_for(base, ROOT / "process_b", True, "session_b_resume", 2, last)); marker("PROCESS_B_CONFIG_CREATED")
        execute("process_b", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(process_b)], repo, env); marker("PROCESS_B_COMPLETED")
        marker("COMPARISON_STARTED"); reference_ckpt = checkpoint_summary(ref_out / "runs" / "seed_42" / "torchvision" / "frcnn" / "last.pt"); resumed_ckpt = checkpoint_summary(ROOT / "process_b" / "runs" / "seed_42" / "torchvision" / "frcnn" / "last.pt")
        checks = {"history": resumed_ckpt["epochs"] == [1, 2, 3], "lr": resumed_ckpt["lrs"] == reference_ckpt["lrs"], "optimizer": resumed_ckpt["optimizer"] == reference_ckpt["optimizer"], "scheduler": resumed_ckpt["scheduler"] == reference_ckpt["scheduler"], "best_epoch": resumed_ckpt["best_epoch"] == reference_ckpt["best_epoch"], "next_epoch": resumed_ckpt["next_epoch"] == 4}
        atomic_json(ROOT / "diagnostics" / "resume_smoke_comparison.json", {"checks": checks, "reference": reference_ckpt, "resumed": resumed_ckpt})
        if not all(checks.values()): raise RuntimeError("Exact-resume structural comparison failed")
        marker("COMPARISON_PASSED"); marker("VERIFIER_STARTED")
        execute("verifier", [sys.executable, "-u", str(repo / "tools/verify_smoke_artifacts.py"), "--out-dir", str(ROOT / "process_b"), "--config", str(process_b), "--environment", str(environment)], repo, env)
        atomic_json(ROOT / "smoke_artifact_manifest.json", {"classification": "SMOKE_DEBUG_ONLY", "canonical": False, "commit": actual, "experiment_id": experiment_id}); marker("VERIFIER_PASSED")
        RESULT.update({"smoke_status": "PASS", "last_verified_stage": "VERIFIER_PASSED", "failed_stage": None, "exception_type": None, "message": None, "return_code": None}); marker("SMOKE_PASS"); return True
    except Exception as exc:
        fail(LAST_STAGE, exc); return False
    finally:
        atomic_json(ROOT / "smoke_result.json", RESULT); shutil.copy2(ROOT / "smoke_result.json", WORKING / "smoke_result.json"); shutil.copy2(ROOT / "stage_markers.jsonl", WORKING / "stage_markers.jsonl"); bundle()

if __name__ == "__main__": main()  # Never re-raise: Kaggle must export SMOKE_FAIL normally.
