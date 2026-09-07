"""Hash-only probes for the non-canonical Faster exact-resume diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _feed(digest: "hashlib._Hash", value: Any) -> None:
    digest.update(type(value).__name__.encode("utf-8") + b"\0")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            _feed(digest, key)
            _feed(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode("ascii"))
        for item in value:
            _feed(digest, item)
    elif isinstance(value, (bytes, bytearray)):
        digest.update(bytes(value))
    elif value is None or isinstance(value, (str, int, float, bool, np.generic)):
        digest.update(repr(value).encode("utf-8"))
    else:
        digest.update(repr(value).encode("utf-8"))


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _feed(digest, value)
    return digest.hexdigest()


def rng_hashes() -> dict[str, Any]:
    return {
        "python": state_sha256(random.getstate()),
        "numpy": state_sha256(np.random.get_state()),
        "torch_cpu": state_sha256(torch.get_rng_state()),
        "torch_cuda": [state_sha256(state) for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available() else [],
    }


def cuda_determinism() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32
        if torch.cuda.is_available() else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32
        if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "UNSET"),
    }


def training_state(model, optimizer=None, scheduler=None, scaler=None,
                   generator=None, history=None) -> dict[str, Any]:
    return {
        "model": state_sha256(model.state_dict()),
        "optimizer": state_sha256(optimizer.state_dict()) if optimizer is not None else None,
        "scheduler": state_sha256(scheduler.state_dict()) if scheduler is not None else None,
        "scaler": state_sha256(scaler.state_dict()) if scaler is not None else None,
        "generator": state_sha256(generator.get_state()) if generator is not None else None,
        "history": state_sha256(history) if history is not None else None,
        "rng": rng_hashes(),
    }


def checkpoint_state(checkpoint: dict) -> dict[str, Any]:
    return {
        "model": state_sha256(checkpoint.get("model")),
        "optimizer": state_sha256(checkpoint.get("optimizer")),
        "scheduler": state_sha256(checkpoint.get("scheduler")),
        "scaler": state_sha256(checkpoint.get("scaler")),
        "generator": state_sha256(checkpoint.get("dataloader_generator_state")),
        "history": state_sha256(checkpoint.get("training_history")),
        "rng": {
            "python": state_sha256(checkpoint.get("python_random_state")),
            "numpy": state_sha256(checkpoint.get("numpy_random_state")),
            "torch_cpu": state_sha256(checkpoint.get("torch_cpu_rng_state")),
            "torch_cuda": [state_sha256(state) for state in checkpoint.get("torch_cuda_rng_states", [])],
        },
        "completed_epoch": checkpoint.get("completed_epoch"),
        "next_epoch": checkpoint.get("next_epoch"),
    }


def batch_state(images: list[torch.Tensor], targets: list[dict]) -> dict[str, Any]:
    sample_ids = [Path(str(target.get("path", target.get("image_id", "unknown")))).stem
                  for target in targets]
    safe_targets = [{key: value for key, value in target.items() if key != "path"}
                    for target in targets]
    return {
        "sample_ids": sample_ids,
        "images_sha256": state_sha256(images),
        "targets_sha256": state_sha256(safe_targets),
    }


def gradient_sha256(model) -> str:
    return state_sha256({name: parameter.grad for name, parameter in model.named_parameters()
                         if parameter.grad is not None})


def append_event(path: str | Path | None, event: str, **payload: Any) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, **payload}
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
