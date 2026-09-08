"""Run one real trainer, preserve diagnostics, then enforce its artifact contract."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.runpod_release import atomic_json, load_and_validate_config, source_commit  # noqa: E402


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def reserve(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Duplicate launch blocked by state file: {path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def monitor_gpu(stop: threading.Event, samples: list[int]) -> None:
    while not stop.wait(1):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            samples.append(int(result.stdout.splitlines()[0].strip()))
        except Exception:
            pass


def stream_process(command: list[str], cwd: Path, log: Path, env: dict) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            assert process.stdout
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                stream.write(line)
                stream.flush()
            return process.wait()
        finally:
            if process.stdout:
                process.stdout.close()


def run(model: str, config: Path, repo: Path, commit: str, mode: str,
        environment: Path, control: Path) -> dict:
    preflight = mode == "preflight"
    cfg = load_and_validate_config(
        config, model, allow_preflight=preflight, allow_output_override=True,
    )
    head = source_commit(repo)
    if head != commit:
        raise RuntimeError(f"Repository pin mismatch: {head} != {commit}")
    output = Path(cfg["out_dir"])
    state = control / f"{output.parent.name}_{model}.json"
    if output.exists():
        raise RuntimeError(f"Clean trainer output must not exist: {output}")
    record = {
        "schema_version": 1, "status": "RUNNING", "model": model,
        "mode": mode, "source_commit": commit, "config": str(config),
        "output": str(output), "started_utc": utc(), "return_code": None,
        "failed_stage": "TRAINING",
    }
    reserve(state, record)
    log = control / "logs" / f"{output.parent.name}_{model}.log"
    execution = control / "executions" / f"{output.parent.name}_{model}.json"
    stop, samples = threading.Event(), []
    thread = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
    thread.start()
    primary_error: Exception | None = None
    try:
        env = os.environ.copy()
        env["CANONICAL_GIT_COMMIT"] = commit
        env["CANONICAL_EXPERIMENT_ID"] = f"{output.parent.name}_{model}"
        command = [sys.executable, "-u", str(repo / "marine_3model_experiment.py"), "--config", str(config)]
        code = stream_process(command, repo, log, env)
        record["return_code"] = code
        if code:
            raise RuntimeError(f"Trainer subprocess exited with return code {code}")
        record["failed_stage"] = "ARTIFACT_VERIFICATION"
        if not environment.is_file():
            raise RuntimeError(f"Environment record missing: {environment}")
        (output / "environment.txt").write_bytes(environment.read_bytes())
        record.update({"ended_utc": utc(), "peak_gpu_memory_mib": max(samples) if samples else None})
        atomic_json(execution, record)
        command = [
            sys.executable, "-u", str(repo / "tools/verify_training_artifacts.py"),
            "--model", model, "--out-dir", str(output), "--config", str(config),
            "--expected-commit", commit, "--execution-record", str(execution),
        ]
        if preflight:
            command.append("--preflight")
        subprocess.run(command, cwd=repo, check=True)
        record.update({
            "status": "VERIFIED_COMPLETE", "failed_stage": None,
            "artifact_manifest": str(output / "training_artifact_manifest.json"),
        })
    except Exception as exc:
        primary_error = exc
        record.update({
            "status": "FAILED", "exception_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "ended_utc": utc(),
        })
    finally:
        stop.set()
        thread.join(timeout=3)
        record["peak_gpu_memory_mib"] = max(samples) if samples else record.get("peak_gpu_memory_mib")
        if not execution.exists():
            atomic_json(execution, record)
        atomic_json(state, record)
    if primary_error:
        raise primary_error
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("yolo", "frcnn", "ssd"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--commit", required=True)
    parser.add_argument("--mode", choices=("preflight", "official"), required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--control", type=Path, default=Path("/workspace/persistent/marine_control"))
    args = parser.parse_args()
    run(args.model, args.config.resolve(), args.repo.resolve(), args.commit, args.mode,
        args.environment.resolve(), args.control.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
