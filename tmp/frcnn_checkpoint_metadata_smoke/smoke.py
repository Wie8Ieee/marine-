"""Version 12: one non-canonical probe for Faster's first numerical divergence."""
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

# Replaced with the committed Version 12 SHA when packaging for Kaggle.
COMMIT = "PINNED_COMMIT_REPLACED_BEFORE_LAUNCH"
REPOSITORY = "https://github.com/Wie8Ieee/marine-.git"
WORKING = Path("/kaggle/working")
ROOT = WORKING / "frcnn_smoke_v12_diagnostic_export"
RUNTIME = WORKING / "frcnn_smoke_v12_diagnostic_runtime"
DIAGNOSTICS = ROOT / "diagnostics"
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
REFERENCE_ONE_OUTPUT = ARTIFACTS / "reference_one"
REFERENCE_TWO_OUTPUT = ARTIFACTS / "reference_two"
RESUME_OUTPUT = ARTIFACTS / "resume_run"
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
    path = CONFIGS / f"{name}_runtime_config.yaml"
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
        with zipfile.ZipFile(WORKING / "frcnn_smoke_v12_diagnostic_bundle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for item in ROOT.rglob("*"):
                if item.is_file(): archive.write(item, item.relative_to(WORKING))
        marker("BUNDLE_CREATED")
    except Exception as exc:
        atomic_json(ROOT / "bundle_failure.json", {"error_type": type(exc).__name__, "message": str(exc)})

def main() -> bool:
    global LAST_STAGE
    # Only parents are created by the launcher. Training owns its clean output directories.
    for directory in (ROOT, DIAGNOSTICS, CONFIGS, ARTIFACTS, RUNTIME): directory.mkdir(parents=True, exist_ok=True)
    marker("NOTEBOOK_STARTED"); marker("EXPORT_ROOT_CREATED"); repo = RUNTIME / "checkout"
    try:
        launcher_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
        if repo.exists(): raise RuntimeError(f"CHECKOUT_CONTRACT_FAILED — checkout already exists: {repo}")
        execute("repository_clone", ["git", "clone", REPOSITORY, str(repo)], ROOT, launcher_env, DIAGNOSTICS / "repository_clone")
        execute("repository_fetch", ["git", "-C", str(repo), "fetch", "origin"], ROOT, launcher_env, DIAGNOSTICS / "repository_fetch")
        execute("repository_checkout", ["git", "-C", str(repo), "checkout", "--detach", COMMIT], ROOT, launcher_env, DIAGNOSTICS / "repository_checkout")
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT: raise RuntimeError("BLOCKED — WRONG GIT COMMIT")
        marker("REPOSITORY_CLONED"); marker("COMMIT_VERIFIED", commit=actual)
        execute("dependencies", [sys.executable, "-u", "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], repo, launcher_env)
        base = yaml.safe_load((repo / "config_runpod_frcnn_seed42.yaml").read_text(encoding="utf-8")); experiment_id = f"frcnn_divergence_diagnostic_v12_{actual[:8]}"
        environment = ROOT / "environment.json"; atomic_json(environment, {"smoke": True, "classification": "SMOKE_DEBUG_ONLY", "canonical": False, "diagnostic_only": True, "experiment_id": experiment_id, "commit": actual, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"})
        env = dict(launcher_env, CANONICAL_GIT_COMMIT=actual, CANONICAL_EXPERIMENT_ID=experiment_id)
        ref_one, ref_two, resume_out = REFERENCE_ONE_OUTPUT, REFERENCE_TWO_OUTPUT, RESUME_OUTPUT
        for ref_out in (ref_one, ref_two):
            if ref_out.exists(): raise RuntimeError(f"REFERENCE_OUTPUT_CONTRACT_FAILED — output already exists: {ref_out}")
        if resume_out.exists(): raise RuntimeError(f"PROCESS_A_OUTPUT_CONTRACT_FAILED — output already exists: {resume_out}")
        reference_one = write_config("reference_one", config_for(base, ref_one, False, "reference_one_uninterrupted", 2))
        reference_two = write_config("reference_two", config_for(base, ref_two, False, "reference_two_uninterrupted", 2))
        process_a = write_config("process_a", config_for(base, resume_out, False, "session_a_clean", 1)); marker("CONFIGS_CREATED")
        env_r1 = dict(env, FRCNN_DIVERGENCE_TRACE=str(DIAGNOSTICS / "reference_one" / "training_trace.jsonl"))
        env_r2 = dict(env, FRCNN_DIVERGENCE_TRACE=str(DIAGNOSTICS / "reference_two" / "training_trace.jsonl"))
        env_a = dict(env, FRCNN_DIVERGENCE_TRACE=str(DIAGNOSTICS / "process_a" / "training_trace.jsonl"))
        execute("reference_one", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(reference_one)], repo, env_r1, DIAGNOSTICS / "reference_one"); marker("REFERENCE_ONE_COMPLETED")
        execute("reference_two", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(reference_two)], repo, env_r2, DIAGNOSTICS / "reference_two"); marker("REFERENCE_TWO_COMPLETED")
        execute("process_a", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(process_a)], repo, env_a, DIAGNOSTICS / "process_a"); marker("PROCESS_A_COMPLETED")
        marker("PROCESS_A_CONTRACT_STARTED"); sys.path.insert(0, str(repo)); from marine_3model_experiment import validate_session_a_contract
        status = validate_session_a_contract(resume_out, expected_next_stage2_epoch=2); last = Path(status["last_checkpoint"])
        if not last.is_file() or not last.stat().st_size or sha256(last) != status["last_checkpoint_sha256"]: raise RuntimeError("PROCESS_A_CONTRACT_FAIL")
        checkpoint_summary(last); marker("PROCESS_A_CONTRACT_PASSED")
        if not resume_out.exists(): raise RuntimeError("PROCESS_B_OUTPUT_CONTRACT_FAILED — Process A output missing")
        process_b = write_config("process_b", config_for(base, resume_out, True, "session_b_resume", 2, last)); marker("PROCESS_B_CONFIG_CREATED")
        env_b = dict(env, FRCNN_DIVERGENCE_TRACE=str(DIAGNOSTICS / "process_b" / "training_trace.jsonl"))
        execute("process_b", [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(process_b)], repo, env_b, DIAGNOSTICS / "process_b"); marker("PROCESS_B_COMPLETED")
        marker("DIVERGENCE_CLASSIFICATION_STARTED")
        diagnosis = ROOT / "divergence_diagnosis.json"
        execute("classifier", [
            sys.executable, "-u", str(repo / "tools/classify_frcnn_divergence.py"),
            "--reference-one", str(DIAGNOSTICS / "reference_one" / "training_trace.jsonl"),
            "--reference-two", str(DIAGNOSTICS / "reference_two" / "training_trace.jsonl"),
            "--process-a", str(DIAGNOSTICS / "process_a" / "training_trace.jsonl"),
            "--process-b", str(DIAGNOSTICS / "process_b" / "training_trace.jsonl"),
            "--output", str(diagnosis),
        ], repo, env, DIAGNOSTICS / "classifier")
        result = json.loads(diagnosis.read_text(encoding="utf-8"))
        marker("DIAGNOSIS_COMPLETE", classification=result["classification"], first_stage=result["first_divergent_stage"], first_epoch=result["first_divergent_epoch"])
        RESULT.update({"smoke_status": "DIAGNOSIS_COMPLETE", "last_verified_stage": "DIAGNOSIS_COMPLETE", "failed_stage": None, "exception_type": None, "message": None, "return_code": 0}); return True
    except Exception as exc:
        fail(LAST_STAGE, exc); return False
    finally:
        atomic_json(ROOT / "smoke_result.json", RESULT); shutil.copy2(ROOT / "smoke_result.json", WORKING / "smoke_result.json"); shutil.copy2(ROOT / "stage_markers.jsonl", WORKING / "stage_markers.jsonl"); bundle()

if __name__ == "__main__": main()  # Never re-raise: Kaggle must export SMOKE_FAIL normally.
