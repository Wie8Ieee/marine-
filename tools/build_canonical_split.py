"""Build and validate the canonical Trash-ICRA19 split artifacts.

This command is validation-only with respect to the dataset: it never edits,
moves, or deletes an image or label. It writes manifests under
``manifests/trash_icra19`` after every invariant has passed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "tmp/sequence_safe_build/prepared_datasets/trash_icra19_clean"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "manifests/trash_icra19"
SEQUENCE_PATTERN = re.compile(r"^(.+?)_frame[0-9]+")
CLASS_NAMES = ("plastic", "bio", "rov")
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def parse_label(path: Path) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter()
    objects = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise RuntimeError(f"Malformed label at {path}:{line_number}: expected 5 fields")
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError as exc:
            raise RuntimeError(f"Non-numeric label at {path}:{line_number}") from exc
        if class_id not in range(len(CLASS_NAMES)):
            raise RuntimeError(f"Invalid class {class_id} at {path}:{line_number}")
        if not all(0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise RuntimeError(f"Out-of-range box at {path}:{line_number}")
        counts[CLASS_NAMES[class_id]] += 1
        objects += 1
    return objects, counts


def build(data_root: Path, output_root: Path) -> dict:
    data_root = data_root.resolve()
    index_path = data_root / "dataset_index.csv"
    if not index_path.is_file():
        raise RuntimeError(f"Missing source index: {index_path}")

    source_rows = list(csv.DictReader(index_path.open(encoding="utf-8", newline="")))
    if len(source_rows) != 7684:
        raise RuntimeError(f"Expected 7,684 source rows, found {len(source_rows)}")

    rows: list[dict[str, object]] = []
    sequence_owners: dict[str, str] = {}
    sequence_frames: dict[tuple[str, str], int] = defaultdict(int)
    image_paths: set[Path] = set()
    label_paths: set[Path] = set()
    image_hash_owners: dict[str, Path] = {}
    duplicate_images: list[tuple[Path, Path]] = []

    for source in source_rows:
        split = source["split"]
        if split not in SPLITS:
            raise RuntimeError(f"Invalid split {split!r}")
        image = Path(source["image"]).resolve()
        label = Path(source["label"]).resolve()
        if not image.is_file():
            raise RuntimeError(f"Missing image: {image}")
        if not label.is_file():
            raise RuntimeError(f"Missing label: {label}")
        if image in image_paths:
            raise RuntimeError(f"Duplicate image path: {image}")
        if label in label_paths:
            raise RuntimeError(f"Duplicate label path: {label}")
        image_paths.add(image)
        label_paths.add(label)

        match = SEQUENCE_PATTERN.match(image.name)
        if not match:
            raise RuntimeError(f"Image does not match strict sequence regex: {image.name}")
        sequence = match.group(1)
        previous = sequence_owners.setdefault(sequence, split)
        if previous != split:
            raise RuntimeError(f"Sequence leakage: {sequence} is in {previous} and {split}")
        sequence_frames[(sequence, split)] += 1

        digest = sha256(image)
        if digest in image_hash_owners:
            duplicate_images.append((image_hash_owners[digest], image))
        else:
            image_hash_owners[digest] = image

        objects, class_counts = parse_label(label)
        indexed_objects = int(source["objects"])
        if objects != indexed_objects:
            raise RuntimeError(
                f"Object-count mismatch for {label}: index={indexed_objects}, actual={objects}"
            )
        rows.append(
            {
                "image_path": repo_relative(image),
                "label_path": repo_relative(label),
                "sequence_id": sequence,
                "split": split,
                "width": int(source["width"]),
                "height": int(source["height"]),
                "object_count": objects,
                **{f"{name}_count": class_counts[name] for name in CLASS_NAMES},
            }
        )

    if duplicate_images:
        sample = "; ".join(f"{a} == {b}" for a, b in duplicate_images[:5])
        raise RuntimeError(f"Found {len(duplicate_images)} duplicate image files: {sample}")
    if len(sequence_owners) != 175:
        raise RuntimeError(f"Expected 175 sequences, found {len(sequence_owners)}")

    rows.sort(key=lambda row: (SPLITS.index(str(row["split"])), str(row["image_path"])))
    summary: dict[str, object] = {
        "dataset": "Trash-ICRA19",
        "status": "canonical",
        "selection_reason": "strict sequence-level, frame-balanced 70/15/15 split with seed 42",
        "split_seed": 42,
        "split_mode": "sequence_70_15_15_frame_balanced",
        "sequence_regex": SEQUENCE_PATTERN.pattern,
        "classes": list(CLASS_NAMES),
        "images": len(rows),
        "labels": len(label_paths),
        "objects": sum(int(row["object_count"]) for row in rows),
        "sequences": len(sequence_owners),
        "sequence_overlap": 0,
        "exact_duplicate_images": 0,
        "splits": {},
        "legacy_splits": [
            {"images": {"train": 5138, "val": 1253, "test": 1293}, "reason": "sequence-count-balanced legacy split"},
            {"images": {"train": 5378, "val": 1153, "test": 1153}, "reason": "image-level stratified legacy split"},
        ],
    }
    for split in SPLITS:
        selected = [row for row in rows if row["split"] == split]
        summary["splits"][split] = {
            "images": len(selected),
            "labels": len(selected),
            "objects": sum(int(row["object_count"]) for row in selected),
            "sequences": len({str(row["sequence_id"]) for row in selected}),
            **{
                name: sum(int(row[f"{name}_count"]) for row in selected)
                for name in CLASS_NAMES
            },
        }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "canonical_split_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with (output_root / "sequence_split_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sequence", "split", "frames"])
        writer.writeheader()
        for (sequence, split), frames in sorted(sequence_frames.items(), key=lambda item: (SPLITS.index(item[0][1]), item[0][0])):
            writer.writerow({"sequence": sequence, "split": split, "frames": frames})

    (output_root / "canonical_split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    relative_data_root = "../../" + repo_relative(data_root)
    (output_root / "canonical_data.yaml").write_text(
        f"path: {relative_data_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 3\n"
        "names: [plastic, bio, rov]\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.data_root, args.output_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
