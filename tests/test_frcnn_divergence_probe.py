from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.classify_frcnn_divergence import classify
from tools.frcnn_divergence_probe import append_event, state_sha256


def event(name, epoch=None, marker="same"):
    value = {"event": name}
    if epoch is not None:
        value["global_epoch"] = epoch
    if name == "INITIALIZED":
        value.update(cuda={"deterministic_algorithms": False}, loader={"num_workers": 0})
    elif name in {"BEFORE_ITERATOR", "AFTER_ITERATOR"}:
        value.update(stage="all", state={
            "model": marker, "optimizer": marker, "scheduler": marker,
            "scaler": marker, "generator": marker + name, "history": marker,
            "rng": {"python": marker, "numpy": marker, "torch_cpu": marker,
                    "torch_cuda": [marker]},
        })
    elif name == "FIRST_BATCH_PRE_STEP":
        value.update(stage="all", batch={"sample_ids": ["a"], "images_sha256": marker,
                                         "targets_sha256": marker}, total_loss=1.0,
                     gradients=marker, state={"model": marker})
    elif name == "FIRST_BATCH_POST_STEP":
        value.update(stage="all", state={"model": marker, "optimizer": marker})
    elif name == "EPOCH_BOUNDARY":
        value.update(stage="all", state={
            "model": marker, "optimizer": marker, "scheduler": marker,
            "scaler": marker, "generator": marker + "AFTER_ITERATOR", "history": marker,
            "rng": {"python": marker, "numpy": marker, "torch_cpu": marker,
                    "torch_cuda": [marker]},
        })
    return value


def reference_trace():
    values = [event("INITIALIZED")]
    for epoch in (1, 2, 3):
        values.extend(event(name, epoch) for name in (
            "BEFORE_ITERATOR", "AFTER_ITERATOR", "FIRST_BATCH_PRE_STEP",
            "FIRST_BATCH_POST_STEP", "EPOCH_BOUNDARY"))
    return values


def process_b_trace():
    boundary = event("EPOCH_BOUNDARY", 2)["state"]
    checkpoint = copy.deepcopy(boundary)
    checkpoint.update(completed_epoch=2, next_epoch=3)
    return [
        event("INITIALIZED"),
        {"event": "CHECKPOINT_LOADED", "checkpoint": checkpoint, "checkpoint_sha256": "a" * 64},
        {"event": "MODEL_AND_RNG_RESTORED", "model": boundary["model"], "rng": boundary["rng"]},
        {"event": "ALL_STATES_RESTORED", "stage": "all", "state": copy.deepcopy(boundary)},
        *[event(name, 3) for name in (
            "BEFORE_ITERATOR", "AFTER_ITERATOR", "FIRST_BATCH_PRE_STEP",
            "FIRST_BATCH_POST_STEP", "EPOCH_BOUNDARY")],
    ]


class DivergenceProbeTests(unittest.TestCase):
    def test_hash_is_stable_and_value_sensitive(self):
        value = {"tensor": torch.tensor([1.0, 2.0]), "shape": (2,)}
        self.assertEqual(state_sha256(value), state_sha256(copy.deepcopy(value)))
        changed = copy.deepcopy(value); changed["tensor"][0] += 1
        self.assertNotEqual(state_sha256(value), state_sha256(changed))
        self.assertEqual(state_sha256(torch.tensor(1.0)), state_sha256(torch.tensor(1.0)))

    def test_trace_contains_hashes_not_tensor_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            digest = state_sha256(torch.tensor([11, 22, 33]))
            append_event(path, "HASH_ONLY", state_sha256=digest)
            raw = path.read_text(encoding="utf-8")
            self.assertIn(digest, raw)
            self.assertNotIn("11, 22, 33", raw)
            self.assertEqual(json.loads(raw)["event"], "HASH_ONLY")

    def test_reference_to_reference_difference_has_priority(self):
        r1 = reference_trace(); r2 = copy.deepcopy(r1)
        next(item for item in r2 if item["event"] == "FIRST_BATCH_POST_STEP"
             and item["global_epoch"] == 2)["state"]["model"] = "different"
        result = classify(r1, r2, reference_trace()[:11], process_b_trace())
        self.assertEqual(result["classification"], "BASELINE CUDA/DATA NONDETERMINISM")
        self.assertEqual(result["first_divergent_stage"], "FIRST_BATCH_POST_STEP")
        self.assertEqual(result["first_divergent_epoch"], 2)

    def test_clean_synthetic_trace_reports_no_divergence(self):
        result = classify(reference_trace(), reference_trace(), reference_trace()[:11], process_b_trace())
        self.assertEqual(result["classification"], "NO DIVERGENCE DETECTED")
        self.assertTrue(result["process_a_to_process_b_restore"]["equal"])
        self.assertTrue(result["first_resumed_batch"]["equal"])


if __name__ == "__main__":
    unittest.main()
