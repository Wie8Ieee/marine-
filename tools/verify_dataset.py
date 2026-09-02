"""Create or verify the canonical Trash-ICRA19 integrity baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "manifests/trash_icra19/canonical_split_manifest.csv"
CLASS_NAMES = ("plastic", "bio", "rov")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(path: Path) -> tuple[int, Counter[str], list[str]]:
    counts: Counter[str] = Counter()
    errors: list[str] = []
    objects = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            errors.append(f"{path}:{number}: expected 5 fields")
            continue
        try:
            class_id = int(fields[0])
            coords = [float(value) for value in fields[1:]]
        except ValueError:
            errors.append(f"{path}:{number}: non-numeric value")
            continue
        if class_id not in range(3):
            errors.append(f"{path}:{number}: invalid class {class_id}")
            continue
        if not all(0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
            errors.append(f"{path}:{number}: invalid normalized box")
            continue
        objects += 1
        counts[CLASS_NAMES[class_id]] += 1
    return objects, counts, errors


def inspect(manifest: Path, data_root: Path | None = None) -> tuple[dict, list[dict[str, str]], list[str]]:
    manifest = manifest.resolve()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    errors: list[str] = []
    file_rows: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    totals: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = {
        split: Counter() for split in ("train", "val", "test")
    }
    sequences: dict[str, str] = {}
    seen_images: set[str] = set()
    seen_labels: set[str] = set()

    for row in sorted(rows, key=lambda item: item["image_path"]):
        image_rel = Path(row["image_path"]).as_posix()
        label_rel = Path(row["label_path"]).as_posix()
        if data_root is None:
            image = REPO_ROOT / image_rel
            label = REPO_ROOT / label_rel
        else:
            image_parts = Path(image_rel).parts
            label_parts = Path(label_rel).parts
            image = data_root / Path(*image_parts[image_parts.index("images"):])
            label = data_root / Path(*label_parts[label_parts.index("labels"):])
        split = row["split"]
        sequence = row["sequence_id"]
        if image_rel in seen_images:
            errors.append(f"duplicate image path: {image_rel}")
        if label_rel in seen_labels:
            errors.append(f"duplicate label path: {label_rel}")
        seen_images.add(image_rel)
        seen_labels.add(label_rel)
        if not image.is_file():
            errors.append(f"missing image: {image_rel}")
            continue
        if not label.is_file():
            errors.append(f"missing label: {label_rel}")
            continue
        owner = sequences.setdefault(sequence, split)
        if owner != split:
            errors.append(f"sequence leakage: {sequence} in {owner} and {split}")

        image_hash = sha256(image)
        label_hash = sha256(label)
        pair_material = f"{image_rel}\0{image_hash}\0{label_rel}\0{label_hash}".encode()
        pair_hash = hashlib.sha256(pair_material).hexdigest()
        aggregate.update(pair_material + f"\0{split}\n".encode())
        objects, classes, label_errors = parse_label(label)
        errors.extend(label_errors)
        expected_objects = int(row["object_count"])
        if objects != expected_objects:
            errors.append(f"object count changed: {label_rel}: {objects} != {expected_objects}")
        for name in CLASS_NAMES:
            if classes[name] != int(row[f"{name}_count"]):
                errors.append(f"class count changed: {label_rel}: {name}")

        totals["images"] += 1
        totals["labels"] += 1
        totals["objects"] += objects
        totals["empty_labels"] += int(objects == 0)
        split_counts[split]["images"] += 1
        split_counts[split]["objects"] += objects
        for name in CLASS_NAMES:
            totals[name] += classes[name]
            split_counts[split][name] += classes[name]
        file_rows.append(
            {
                "image_path": image_rel,
                "image_sha256": image_hash,
                "label_path": label_rel,
                "label_sha256": label_hash,
                "pair_sha256": pair_hash,
                "split": split,
                "sequence_id": sequence,
            }
        )

    fingerprint = {
        "schema_version": 1,
        "hash_algorithm": "SHA-256",
        "path_policy": "repository-relative POSIX paths; timestamps excluded",
        "dataset": "Trash-ICRA19",
        "dataset_sha256": aggregate.hexdigest(),
        "split_manifest_sha256": sha256(manifest),
        "counts": dict(totals),
        "sequences": len(sequences),
        "sequence_overlap": sum(1 for error in errors if error.startswith("sequence leakage")),
        "malformed_or_out_of_range_boxes": len([error for error in errors if ":" in error and "label" not in error]),
        "splits": {name: dict(values) for name, values in split_counts.items()},
    }
    return fingerprint, file_rows, errors


def write_baseline(manifest: Path, fingerprint: dict, rows: list[dict[str, str]]) -> None:
    output = manifest.parent
    (output / "dataset_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "file_hashes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = fingerprint["counts"]
    lines = [
        "# Trash-ICRA19 Data Integrity Report",
        "",
        f"- Dataset SHA-256: `{fingerprint['dataset_sha256']}`",
        f"- Split manifest SHA-256: `{fingerprint['split_manifest_sha256']}`",
        f"- Images/labels: {counts['images']}/{counts['labels']}",
        f"- Objects: {counts['objects']}",
        f"- Classes: plastic={counts['plastic']}, bio={counts['bio']}, rov={counts['rov']}",
        f"- Sequences: {fingerprint['sequences']}",
        f"- Empty labels: {counts.get('empty_labels', 0)}",
        f"- Sequence overlap: {fingerprint['sequence_overlap']}",
        f"- Malformed/out-of-range boxes: {fingerprint['malformed_or_out_of_range_boxes']}",
        "- Missing images or labels: 0",
        "",
        "Verification command:",
        "",
        "```text",
        "python tools/verify_dataset.py --manifest manifests/trash_icra19/canonical_split_manifest.csv",
        "```",
    ]
    (output / "data_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Relocated dataset root containing images/ and labels/")
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    data_root = args.data_root.expanduser().resolve() if args.data_root else None
    fingerprint, rows, errors = inspect(manifest, data_root=data_root)
    if errors:
        print("Integrity errors:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    baseline_path = manifest.parent / "dataset_fingerprint.json"
    if args.write_baseline:
        write_baseline(manifest, fingerprint, rows)
        print(json.dumps(fingerprint, indent=2))
        return 0
    if not baseline_path.is_file():
        print(f"Missing baseline: {baseline_path}", file=sys.stderr)
        return 1
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    differences = [
        key for key in ("dataset_sha256", "split_manifest_sha256", "counts", "sequences", "splits")
        if fingerprint.get(key) != baseline.get(key)
    ]
    if differences:
        print(f"Fingerprint mismatch in: {', '.join(differences)}", file=sys.stderr)
        return 1
    print(f"VERIFIED dataset_sha256={fingerprint['dataset_sha256']}")
    print(f"VERIFIED split_manifest_sha256={fingerprint['split_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
