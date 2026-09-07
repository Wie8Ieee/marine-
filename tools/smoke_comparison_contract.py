"""One fail-closed comparison contract for non-canonical exact-resume smoke."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

COMPARISON_FILENAME = "resume_smoke_comparison.json"
SCHEMA_VERSION = 1
PASS_STATUS = "RESUME_SMOKE_COMPARISON_PASS"
FAIL_STATUS = "FAIL"
NUMERICAL_POLICY = {"mode": "exact", "rtol": 0.0, "atol": 0.0}
STATE_FIELDS = {
    "model_state": ("model",),
    "optimizer_state": ("optimizer",),
    "scheduler_state": ("scheduler",),
    "scaler_state": ("scaler",),
    "rng_state": ("python_random_state", "numpy_random_state",
                  "torch_cpu_rng_state", "torch_cuda_rng_states", "rng_format_version"),
    "dataloader_state": ("dataloader_generator_state",),
    "sampler_state": ("sampler_state",),
    "training_history": ("training_history",),
    "best_metric": ("best_map", "val_metrics"),
    "provenance": ("checkpoint_identity", "training_config_sha256", "dataset_sha256",
                   "split_sha256", "git_commit", "seed", "experiment_id", "architecture"),
    "stage": ("stage", "stage_epoch"),
}
REQUIRED_CHECKS = (
    "history", "lr", "optimizer", "scheduler", "best_epoch", "next_epoch",
    *STATE_FIELDS,
)


def comparison_path(output_root: Path) -> Path:
    # This pure path function must never create the clean training directory.
    return Path(output_root) / COMPARISON_FILENAME


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def exact_equal(left, right) -> bool:
    """Compare values, not just state-dict structure; never serialize RNG values."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)):
            return False
        if left.dtype != right.dtype or left.shape != right.shape:
            return False
        return bool(torch.isfinite(left).all() and torch.isfinite(right).all()
                    and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
                and left.dtype == right.dtype and left.shape == right.shape
                and bool(np.isfinite(left).all() and np.isfinite(right).all()
                         and np.array_equal(left, right)))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(exact_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(exact_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (float, np.floating)):
        return bool(math.isfinite(left) and math.isfinite(right) and left == right)
    if isinstance(left, (int, str, bool, bytes, np.integer)) or left is None:
        return bool(left == right)
    return False


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path)}


def build_comparison(reference_path: Path, resumed_path: Path, total_epochs: int = 3) -> dict:
    if type(total_epochs) is not int or total_epochs < 1:
        raise RuntimeError("Invalid smoke epoch plan")
    for path in (reference_path, resumed_path):
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError("Missing or empty comparison checkpoint")
    # Only locally generated, provenance-checked pipeline checkpoints are in scope.
    ref = torch.load(reference_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    history_a, history_b = ref.get("training_history", []), resumed.get("training_history", [])
    epochs = list(range(1, total_epochs + 1))
    checks = {
        "history": ([row.get("epoch") for row in history_a] == epochs
                    and [row.get("epoch") for row in history_b] == epochs),
        "lr": (bool(history_a and history_b)
               and all("lr" in row for row in history_a + history_b)
               and exact_equal([row.get("lr") for row in history_a],
                               [row.get("lr") for row in history_b])),
        # Preserve every existing structural check and add full numerical states below.
        "optimizer": ("optimizer" in ref and "optimizer" in resumed
                      and exact_equal(ref["optimizer"].get("param_groups"),
                                      resumed["optimizer"].get("param_groups"))),
        "scheduler": ("scheduler" in ref and "scheduler" in resumed
                      and exact_equal(ref["scheduler"], resumed["scheduler"])),
        "best_epoch": ("best_epoch" in ref and "best_epoch" in resumed
                       and ref["best_epoch"] == resumed["best_epoch"]
                       and 1 <= ref["best_epoch"] <= total_epochs),
        "next_epoch": all(cp.get("next_epoch") == total_epochs + 1
                          and cp.get("completed_epoch") == cp.get("epoch") == total_epochs
                          for cp in (ref, resumed)),
    }
    details = {}
    for name, fields in STATE_FIELDS.items():
        missing = [field for field in fields if field not in ref or field not in resumed]
        differing = [field for field in fields if field not in missing
                     and not exact_equal(ref[field], resumed[field])]
        checks[name] = not missing and not differing
        details[name] = {"missing_fields": missing, "differing_fields": differing}
    for name in ("model_state", "optimizer_state"):
        field = STATE_FIELDS[name][0]
        if not ref.get(field) or not resumed.get(field):
            checks[name] = False
            details[name]["empty_state"] = True
    if "model" in ref and "model" in resumed:
        shared = ref["model"].keys() & resumed["model"].keys()
        details["model_state"]["differing_tensor_count"] = sum(
            not exact_equal(ref["model"][key], resumed["model"][key]) for key in shared)
    # Ignore only per-session operational identity (session_id/control hash/cfg paths).
    # No tensor tolerances are introduced: existing equality checks remain exact.
    report = {
        "schema_version": SCHEMA_VERSION,
        "classification": "SMOKE_DEBUG_ONLY",
        "canonical": False,
        "status": PASS_STATUS if all(checks.values()) else FAIL_STATUS,
        "numerical_policy": dict(NUMERICAL_POLICY),
        "expected_epochs": epochs,
        "reference": _record(reference_path),
        "resumed": _record(resumed_path),
        "checks": checks,
        "details": details,
    }
    validate_comparison(report, require_pass=False)
    return report


def validate_comparison(report: dict, *, require_pass: bool = True) -> None:
    prefix = "SMOKE_COMPARISON_CONTRACT_FAILED — "
    required = {"schema_version", "classification", "canonical", "status", "numerical_policy",
                "expected_epochs", "reference", "resumed", "checks", "details"}
    if not isinstance(report, dict) or not required <= report.keys():
        raise RuntimeError(prefix + "required schema fields missing")
    if type(report["schema_version"]) is not int or report["schema_version"] != SCHEMA_VERSION:
        raise RuntimeError(prefix + "unsupported schema version")
    if report["classification"] != "SMOKE_DEBUG_ONLY" or report["canonical"] is not False:
        raise RuntimeError(prefix + "invalid experiment classification")
    if report["numerical_policy"] != NUMERICAL_POLICY:
        raise RuntimeError(prefix + "numerical tolerance changed")
    epochs = report["expected_epochs"]
    if not isinstance(epochs, list) or not epochs or any(type(e) is not int for e in epochs):
        raise RuntimeError(prefix + "invalid epoch plan")
    if epochs != list(range(1, len(epochs) + 1)):
        raise RuntimeError(prefix + "non-contiguous epoch plan")
    checks = report["checks"]
    if not isinstance(checks, dict) or not set(REQUIRED_CHECKS) <= checks.keys():
        raise RuntimeError(prefix + "required checks missing")
    if any(type(value) is not bool for value in checks.values()):
        raise RuntimeError(prefix + "check results must be booleans")
    if not isinstance(report["details"], dict):
        raise RuntimeError(prefix + "invalid comparison details")
    for label in ("reference", "resumed"):
        record = report[label]
        if not isinstance(record, dict) or not {"path", "size_bytes", "sha256"} <= record.keys():
            raise RuntimeError(prefix + "checkpoint identity fields missing")
        if (not isinstance(record["path"], str) or not record["path"]
                or type(record["size_bytes"]) is not int or record["size_bytes"] <= 0
                or not isinstance(record["sha256"], str) or len(record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in record["sha256"])):
            raise RuntimeError(prefix + "invalid checkpoint identity")
    expected_status = PASS_STATUS if all(checks.values()) else FAIL_STATUS
    if report["status"] != expected_status:
        raise RuntimeError(prefix + "status contradicts comparison checks")
    if require_pass and expected_status != PASS_STATUS:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(prefix + "failed checks: " + failed)


def write_comparison(output_root: Path, report: dict) -> Path:
    validate_comparison(report, require_pass=False)
    if not Path(output_root).is_dir():
        raise RuntimeError("COMPARISON_OUTPUT_MISSING — Process A/B must create resume output")
    path = comparison_path(output_root)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def verify_comparison_artifact(output_root: Path, reference_output: Path, total_epochs: int) -> dict:
    path = comparison_path(output_root)
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError("SMOKE_COMPARISON_CONTRACT_FAILED — canonical comparison file missing")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("SMOKE_COMPARISON_CONTRACT_FAILED — malformed JSON") from exc
    validate_comparison(report)
    suffix = Path("runs/seed_42/torchvision/frcnn/last.pt")
    actual = build_comparison(reference_output / suffix, output_root / suffix, total_epochs)
    validate_comparison(actual)
    for field in ("checks", "expected_epochs", "numerical_policy"):
        if report[field] != actual[field]:
            raise RuntimeError("SMOKE_COMPARISON_CONTRACT_FAILED — stale or altered results")
    for label in ("reference", "resumed"):
        for field in ("sha256", "size_bytes"):
            if report[label][field] != actual[label][field]:
                raise RuntimeError("SMOKE_COMPARISON_CONTRACT_FAILED — checkpoint binding mismatch")
    return report
