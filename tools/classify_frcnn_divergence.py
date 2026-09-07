"""Classify the first exact-resume divergence from hash-only training traces."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


EVENT_ORDER = (
    "INITIALIZED", "BEFORE_ITERATOR", "AFTER_ITERATOR",
    "FIRST_BATCH_PRE_STEP", "FIRST_BATCH_POST_STEP", "EPOCH_BOUNDARY",
)


def read_trace(path: Path) -> list[dict]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed diagnostic trace {path.name}:{line_number}") from exc
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise RuntimeError(f"Invalid diagnostic event {path.name}:{line_number}")
        events.append(event)
    return events


def find(events: list[dict], name: str, epoch: int | None = None) -> dict:
    matches = [event for event in events if event["event"] == name
               and (epoch is None or event.get("global_epoch") == epoch)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} event for epoch={epoch}; found {len(matches)}")
    return matches[0]


def comparable(event: dict) -> dict:
    return {key: value for key, value in event.items() if key != "event"}


def first_trace_difference(left: list[dict], right: list[dict]) -> tuple[str | None, int | None, list[str]]:
    initial_left, initial_right = find(left, "INITIALIZED"), find(right, "INITIALIZED")
    if comparable(initial_left) != comparable(initial_right):
        keys = sorted(key for key in set(initial_left) | set(initial_right)
                      if key != "event" and initial_left.get(key) != initial_right.get(key))
        return "INITIALIZED", 0, keys
    for epoch in (1, 2, 3):
        for event_name in EVENT_ORDER[1:]:
            a = [event for event in left if event["event"] == event_name and event.get("global_epoch") == epoch]
            b = [event for event in right if event["event"] == event_name and event.get("global_epoch") == epoch]
            if len(a) != len(b):
                return event_name, epoch, ["event_count"]
            if a and comparable(a[0]) != comparable(b[0]):
                keys = sorted(key for key in set(a[0]) | set(b[0])
                              if key != "event" and a[0].get(key) != b[0].get(key))
                return event_name, epoch, keys
    return None, None, []


def restored_state_matches(a_boundary: dict, b_loaded: dict, b_ready: dict) -> tuple[bool, list[str]]:
    checkpoint = b_loaded["checkpoint"]
    a_state = a_boundary["state"]
    ready = b_ready["state"]
    fields = ("model", "optimizer", "scheduler", "scaler", "history", "rng")
    mismatches = [field for field in fields
                  if a_state.get(field) != checkpoint.get(field) or checkpoint.get(field) != ready.get(field)]
    return not mismatches, mismatches


def classify(r1: list[dict], r2: list[dict], process_a: list[dict], process_b: list[dict]) -> dict:
    event, epoch, keys = first_trace_difference(r1, r2)
    r1_r2_equal = event is None
    r1_boundary = find(r1, "EPOCH_BOUNDARY", 2)
    a_boundary = find(process_a, "EPOCH_BOUNDARY", 2)
    boundary_equal = r1_boundary["state"] == a_boundary["state"]
    loaded = find(process_b, "CHECKPOINT_LOADED")
    restored = find(process_b, "ALL_STATES_RESTORED")
    restore_equal, restore_mismatches = restored_state_matches(a_boundary, loaded, restored)
    rng_restored = find(process_b, "MODEL_AND_RNG_RESTORED")["rng"]
    before_iterator = find(process_b, "BEFORE_ITERATOR", 3)
    after_iterator = find(process_b, "AFTER_ITERATOR", 3)
    rng_changed_before_iterator = rng_restored != before_iterator["state"]["rng"]
    r1_batch = find(r1, "FIRST_BATCH_PRE_STEP", 3)
    b_batch = find(process_b, "FIRST_BATCH_PRE_STEP", 3)
    batch_equal = r1_batch["batch"] == b_batch["batch"]
    pre_step_equal = (r1_batch["state"]["model"] == b_batch["state"]["model"]
                      and r1_batch["total_loss"] == b_batch["total_loss"])
    gradients_equal = r1_batch["gradients"] == b_batch["gradients"]
    r1_post = find(r1, "FIRST_BATCH_POST_STEP", 3)
    b_post = find(process_b, "FIRST_BATCH_POST_STEP", 3)
    post_step_equal = (r1_post["state"]["model"] == b_post["state"]["model"]
                       and r1_post["state"]["optimizer"] == b_post["state"]["optimizer"])

    if not r1_r2_equal:
        classification = "BASELINE CUDA/DATA NONDETERMINISM"
        first_stage, first_epoch = event, epoch
    elif not boundary_equal:
        classification = "PRE-RESUME INITIALIZATION OR DATA ORDER DIVERGENCE"
        first_stage, first_epoch = "EPOCH_BOUNDARY", 2
    elif not restore_equal:
        classification = "INCOMPLETE OR INCORRECT RESTORE"
        first_stage, first_epoch = "ALL_STATES_RESTORED", 3
    elif rng_changed_before_iterator:
        classification = "POST-RESTORE RNG CONSUMPTION"
        first_stage, first_epoch = "BEFORE_ITERATOR", 3
    elif not batch_equal:
        classification = "DATALOADER/SAMPLER/WORKER STATE ERROR"
        first_stage, first_epoch = "FIRST_BATCH_PRE_STEP", 3
    elif pre_step_equal and (not gradients_equal or not post_step_equal):
        classification = "NONDETERMINISTIC CUDA OPERATION"
        first_stage, first_epoch = "FIRST_BATCH_PRE_STEP", 3
    else:
        classification = "NO DIVERGENCE DETECTED"
        first_stage = first_epoch = None

    return {
        "status": "DIAGNOSIS_COMPLETE",
        "classification": classification,
        "first_divergent_stage": first_stage,
        "first_divergent_epoch": first_epoch,
        "first_divergent_batch": 1 if first_stage in {"FIRST_BATCH_PRE_STEP", "FIRST_BATCH_POST_STEP"} else None,
        "reference_to_reference": {"equal": r1_r2_equal, "differing_fields": keys},
        "reference_to_process_a_boundary": {"equal": boundary_equal},
        "process_a_to_process_b_restore": {"equal": restore_equal, "differing_fields": restore_mismatches},
        "first_resumed_batch": {
            "equal": batch_equal,
            "sample_ids_equal": r1_batch["batch"]["sample_ids"] == b_batch["batch"]["sample_ids"],
            "inputs_equal": r1_batch["batch"]["images_sha256"] == b_batch["batch"]["images_sha256"],
            "targets_equal": r1_batch["batch"]["targets_sha256"] == b_batch["batch"]["targets_sha256"],
            "pre_step_equal": pre_step_equal,
            "gradients_equal": gradients_equal,
            "post_step_equal": post_step_equal,
        },
        "rng_restore_order": {
            "restored_to_before_iterator_equal": not rng_changed_before_iterator,
            "before_to_after_iterator_global_rng_equal": before_iterator["state"]["rng"] == after_iterator["state"]["rng"],
            "epoch_generator_changed_by_iterator": before_iterator["state"]["generator"] != after_iterator["state"]["generator"],
        },
        "cuda_determinism": find(r1, "INITIALIZED")["cuda"],
        "loader": find(r1, "INITIALIZED")["loader"],
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("reference-one", "reference-two", "process-a", "process-b"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = classify(read_trace(args.reference_one), read_trace(args.reference_two),
                      read_trace(args.process_a), read_trace(args.process_b))
    atomic_json(args.output, result)
    print(result["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
