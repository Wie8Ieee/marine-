"""Failure-durable helpers for the non-canonical Faster exact-resume smoke."""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import traceback
import zipfile
from pathlib import Path
from typing import Callable, Sequence


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


class SmokeExport:
    """Owns an export tree and never raises a stage failure to Kaggle."""

    def __init__(self, root: Path, working_root: Path) -> None:
        self.root, self.working_root = root, working_root
        self.last_stage = "NOT_STARTED"
        self.result: dict = {"smoke_status": "FAIL", "last_verified_stage": self.last_stage,
                             "failed_stage": None, "exception_type": None, "message": None, "return_code": None}
        for directory in (root, root / "reference", root / "process_a", root / "process_b", root / "configs", root / "diagnostics"):
            directory.mkdir(parents=True, exist_ok=True)
        self.marker("EXPORT_ROOT_CREATED")

    def marker(self, stage: str, **details: object) -> None:
        payload = {"stage": stage, "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **details}
        target = self.root / "stage_markers.jsonl"
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, default=str) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        self.last_stage = stage

    def failure(self, stage: str, exc: BaseException, return_code: int | None = None) -> None:
        self.marker("SMOKE_FAIL", failed_stage=stage, error_type=type(exc).__name__)
        self.result.update({"smoke_status": "FAIL", "last_verified_stage": self.last_stage,
                            "failed_stage": stage, "exception_type": type(exc).__name__,
                            "message": str(exc), "return_code": return_code})
        atomic_json(self.root / "unhandled_exception.json", {"stage": stage, "error_type": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc()})

    def run_process(self, stage: str, command: Sequence[str], cwd: Path, env: dict[str, str]) -> int:
        directory = self.root / stage
        safe_command = ["<redacted>" if any(word in str(item).lower() for word in ("token", "password", "api_key", "secret")) else str(item) for item in command]
        atomic_json(directory / "command.json", {"command": safe_command})
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
            child = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, text=True)
            code = child.wait()
            stdout.flush(); os.fsync(stdout.fileno()); stderr.flush(); os.fsync(stderr.fileno())
        execution = {"stage": stage, "pid": child.pid, "return_code": code, "start_timestamp_utc": started,
                     "end_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "command": safe_command}
        atomic_json(directory / "execution.json", execution)
        if code:
            atomic_json(directory / "failure.json", {**execution, "message": f"subprocess returned {code}"})
            raise RuntimeError(f"{stage} subprocess returned {code}")
        return code

    def finalize(self, passed: bool = False) -> None:
        if passed:
            self.marker("SMOKE_PASS")
            self.result.update({"smoke_status": "PASS", "last_verified_stage": self.last_stage,
                                "failed_stage": None, "exception_type": None, "message": None, "return_code": None})
        atomic_json(self.root / "smoke_result.json", self.result)
        for name in ("smoke_result.json", "stage_markers.jsonl"):
            shutil.copy2(self.root / name, self.working_root / name)
        bundle = self.working_root / "frcnn_smoke_v7_bundle.zip"
        try:
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                for item in self.root.rglob("*"):
                    if item.is_file(): archive.write(item, item.relative_to(self.root.parent))
            self.marker("BUNDLE_CREATED")
        except Exception as exc:  # preserve the primary failure regardless of zip failure
            atomic_json(self.root / "bundle_failure.json", {"error_type": type(exc).__name__, "message": str(exc)})
