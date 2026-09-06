"""Durable, secret-safe subprocess diagnostics for non-canonical smoke launchers."""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import traceback
from pathlib import Path
from typing import Mapping, Sequence


_SECRET = re.compile(r"(?i)(token|secret|password|api[_-]?key)(=|:)[^\s]+")


def _safe(value: str) -> str:
    return _SECRET.sub(r"\1\2[REDACTED]", value)


def run_logged_process(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str], output_root: Path, label: str,
) -> int:
    """Run one process, retaining diagnostics even if it exits unsuccessfully."""
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / f"{label}_stdout.log"
    stderr_path = output_root / f"{label}_stderr.log"
    execution_path = output_root / f"{label}_execution.json"
    failure_path = output_root / f"{label}_failure.json"
    started = dt.datetime.now(dt.timezone.utc)
    return_code = None
    error = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=cwd, env=dict(env), stdout=stdout, stderr=stderr, check=False, text=True)
        return_code = completed.returncode
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(command))
        return return_code
    except Exception as exc:
        error = exc
        failure_path.write_text(json.dumps({
            "label": label, "error_type": type(exc).__name__, "error_message": _safe(str(exc)),
            "traceback": _safe(traceback.format_exc()), "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path), "return_code": return_code,
        }, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        ended = dt.datetime.now(dt.timezone.utc)
        execution_path.write_text(json.dumps({
            "label": label, "command": [_safe(str(item)) for item in command],
            "return_code": return_code, "started_utc": started.isoformat(), "ended_utc": ended.isoformat(),
            "stdout_log": str(stdout_path), "stderr_log": str(stderr_path),
            "failure_log": str(failure_path) if error else None,
        }, indent=2) + "\n", encoding="utf-8")
