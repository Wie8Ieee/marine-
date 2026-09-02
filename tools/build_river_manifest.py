"""Validate the canonical River Floating Trash archive and write its registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / ".data_inspection/river-floating-trash-datasets.zip"
DEFAULT_OUTPUT = REPO_ROOT / "manifests/river"
CLASS_NAMES = (
    "bottle", "grass", "branch", "milk-box", "plastic-bag",
    "plastic-garbage", "ball", "leaf",
)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(archive: Path) -> tuple[list[dict[str, object]], dict, list[str]]:
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    aggregate = hashlib.sha256()
    image_hashes: set[str] = set()
    duplicate_members: dict[str, list[str]] = {}
    class_totals: Counter[str] = Counter()
    original_splits: Counter[str] = Counter()

    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        images = sorted(
            name for name in names
            if name.startswith("datasets/RFT/images/") and name.lower().endswith(".jpg")
        )
        for image_member in images:
            path = PurePosixPath(image_member)
            original_split = path.parts[-2]
            label_member = str(PurePosixPath("datasets/RFT/labels") / original_split / f"{path.stem}.txt")
            xml_member = str(PurePosixPath("datasets/RFT/xml") / f"{path.stem}.xml")
            if label_member not in names:
                errors.append(f"missing label: {label_member}")
                continue
            if xml_member not in names:
                errors.append(f"missing XML: {xml_member}")
                continue
            image_data = bundle.read(image_member)
            label_data = bundle.read(label_member)
            image_hash = digest_bytes(image_data)
            label_hash = digest_bytes(label_data)
            if image_hash in image_hashes:
                duplicate_members.setdefault(image_hash, []).append(image_member)
            else:
                duplicate_members[image_hash] = [image_member]
            image_hashes.add(image_hash)
            with Image.open(io.BytesIO(image_data)) as image:
                width, height = image.size
                image.verify()

            counts: Counter[str] = Counter()
            object_count = 0
            for line_number, raw in enumerate(label_data.decode("utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                fields = raw.split()
                if len(fields) != 5:
                    errors.append(f"malformed label: {label_member}:{line_number}")
                    continue
                try:
                    class_id = int(fields[0])
                    coords = [float(value) for value in fields[1:]]
                except ValueError:
                    errors.append(f"non-numeric label: {label_member}:{line_number}")
                    continue
                if class_id not in range(len(CLASS_NAMES)):
                    errors.append(f"invalid class: {label_member}:{line_number}")
                    continue
                if not all(0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"invalid box: {label_member}:{line_number}")
                    continue
                counts[CLASS_NAMES[class_id]] += 1
                class_totals[CLASS_NAMES[class_id]] += 1
                object_count += 1

            original_splits[original_split] += 1
            pair_hash = digest_bytes(
                f"{image_member}\0{image_hash}\0{label_member}\0{label_hash}".encode()
            )
            aggregate.update(
                f"{image_member}\0{image_hash}\0{label_member}\0{label_hash}\n".encode()
            )
            rows.append(
                {
                    "archive_path": archive.resolve().relative_to(REPO_ROOT).as_posix(),
                    "image_member": image_member,
                    "label_member": label_member,
                    "xml_member": xml_member,
                    "original_split": original_split,
                    "split": "external_test",
                    "width": width,
                    "height": height,
                    "object_count": object_count,
                    **{f"{name}_count": counts[name] for name in CLASS_NAMES},
                    "image_sha256": image_hash,
                    "label_sha256": label_hash,
                    "pair_sha256": pair_hash,
                }
            )

    if len(rows) != 2400:
        errors.append(f"expected 2,400 image/label pairs, found {len(rows)}")
    objects = sum(int(row["object_count"]) for row in rows)
    if objects != 3510:
        errors.append(f"expected 3,510 objects, found {objects}")
    trash_hashes_path = REPO_ROOT / "manifests/trash_icra19/file_hashes.csv"
    overlap = 0
    if trash_hashes_path.is_file():
        trash_hashes = {
            row["image_sha256"] for row in csv.DictReader(trash_hashes_path.open(encoding="utf-8", newline=""))
        }
        overlap = len(image_hashes & trash_hashes)
        if overlap:
            errors.append(f"River/Trash exact-image overlap: {overlap}")
    duplicate_groups = [members for members in duplicate_members.values() if len(members) > 1]
    duplicate_label_hashes: dict[str, set[str]] = {}
    for row in rows:
        duplicate_label_hashes.setdefault(str(row["image_sha256"]), set()).add(str(row["label_sha256"]))
    conflicting_hashes = {
        digest for digest, members in duplicate_members.items()
        if len(members) > 1 and len(duplicate_label_hashes[digest]) > 1
    }
    eligible_rows = [row for row in rows if str(row["image_sha256"]) not in conflicting_hashes]
    evaluation_aggregate = hashlib.sha256()
    for row in eligible_rows:
        evaluation_aggregate.update(
            f"{row['image_member']}\0{row['image_sha256']}\0{row['label_member']}\0{row['label_sha256']}\n".encode()
        )
    fingerprint = {
        "schema_version": 1,
        "dataset": "River Floating Trash",
        "role": "external_test_only",
        "status": "canonical_full_source_with_filtered_evaluation_subset",
        "hash_algorithm": "SHA-256",
        "path_policy": "archive-relative POSIX member names; timestamps excluded",
        "archive_path": archive.resolve().relative_to(REPO_ROOT).as_posix(),
        "archive_sha256": digest_file(archive),
        "dataset_sha256": aggregate.hexdigest(),
        "images": len(rows),
        "annotated_images": len(rows),
        "labels": len(rows),
        "objects": objects,
        "empty_labels": sum(int(row["object_count"] == 0) for row in rows),
        "classes": dict(class_totals),
        "original_splits": dict(original_splits),
        "trash_icra19_exact_image_overlap": overlap,
        "unique_image_contents": len(image_hashes),
        "duplicate_image_groups": len(duplicate_groups),
        "duplicate_extra_files": sum(len(group) - 1 for group in duplicate_groups),
        "duplicate_groups_with_different_labels": sum(
            len(duplicate_label_hashes[digest]) > 1
            for digest, members in duplicate_members.items() if len(members) > 1
        ),
        "duplicate_policy": "exclude every member of each conflicting exact-image group from official evaluation",
        "excluded_conflicting_images": len(rows) - len(eligible_rows),
        "evaluation_images": len(eligible_rows),
        "evaluation_objects": sum(int(row["object_count"]) for row in eligible_rows),
        "evaluation_dataset_sha256": evaluation_aggregate.hexdigest(),
        "malformed_or_out_of_range_boxes": len(errors),
    }
    return rows, fingerprint, errors


def write_outputs(output: Path, rows: list[dict[str, object]], fingerprint: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "river_canonical_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    label_hashes_by_image: dict[str, set[str]] = {}
    image_hash_counts = Counter(str(row["image_sha256"]) for row in rows)
    for row in rows:
        label_hashes_by_image.setdefault(str(row["image_sha256"]), set()).add(str(row["label_sha256"]))
    conflicting_hashes = {
        digest for digest, count in image_hash_counts.items()
        if count > 1 and len(label_hashes_by_image[digest]) > 1
    }
    eligible_rows = [row for row in rows if str(row["image_sha256"]) not in conflicting_hashes]
    with (output / "river_evaluation_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(eligible_rows)
    (output / "river_fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2) + "\n", encoding="utf-8"
    )
    mapping = {
        "source_classes": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        "target_classes": {"0": "trash"},
        "mapping": {str(index): 0 for index in range(len(CLASS_NAMES))},
        "policy": "River is external evaluation only; exclude all conflicting duplicate-image groups; collapse ground truth and predictions to trash; use one class-agnostic framework NMS",
    }
    (output / "river_class_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# River Floating Trash Integrity Report",
        "",
        "## Canonical source",
        "",
        f"- Archive: `{fingerprint['archive_path']}`",
        f"- Archive SHA-256: `{fingerprint['archive_sha256']}`",
        f"- Dataset SHA-256: `{fingerprint['dataset_sha256']}`",
        f"- Images/labels/objects: {fingerprint['images']}/{fingerprint['labels']}/{fingerprint['objects']}",
        f"- Original split counts: {fingerprint['original_splits']}",
        f"- Empty labels: {fingerprint['empty_labels']}",
        f"- Trash-ICRA19 exact-image overlap: {fingerprint['trash_icra19_exact_image_overlap']}",
        f"- Unique image contents: {fingerprint['unique_image_contents']}",
        f"- Duplicate image groups: {fingerprint['duplicate_image_groups']}",
        f"- Duplicate extra files: {fingerprint['duplicate_extra_files']}",
        f"- Duplicate groups with different label files: {fingerprint['duplicate_groups_with_different_labels']}",
        "- Malformed/out-of-range boxes: 0",
        f"- Official evaluation subset: {fingerprint['evaluation_images']} images/{fingerprint['evaluation_objects']} objects.",
        f"- Evaluation-subset SHA-256: `{fingerprint['evaluation_dataset_sha256']}`",
        f"- Excluded conflicting images: {fingerprint['excluded_conflicting_images']} (all members of conflicting groups).",
        "- Adoption status: canonical full source; official evaluation uses the filtered manifest.",
        "",
        "## Evaluation contract",
        "",
        "1. Validate the complete 2,400-image/3,510-object source, then evaluate only the conflict-free filtered manifest; never train or select checkpoints on River.",
        "2. Run the three-class Trash-ICRA19 detector unchanged.",
        "3. Collapse every predicted and ground-truth class to class 0 (`trash`).",
        "4. Apply class-agnostic NMS exactly once inside each detector framework (YOLO `agnostic_nms`; patched final detection NMS for TorchVision).",
        "5. Do not apply a second common NMS pass after framework prediction.",
        "6. Match at the documented confidence and IoU thresholds.",
        "",
        "## Legacy copies",
        "",
        "- The 64-image prepared copy is an invalid partial/debug extraction.",
        "- The 1,416-image prepared copy is incomplete and must not produce official results.",
    ]
    (output / "river_integrity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows, fingerprint, errors = inspect(args.archive.resolve())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    write_outputs(args.output.resolve(), rows, fingerprint)
    print(json.dumps(fingerprint, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
