from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import yaml
from PIL import Image

import marine_3model_experiment as experiment
import tools.runpod_model_runner as runner
from tools.runpod_release import sha256_file
from tools.verify_dataset import inspect, parse_label
from tools.verify_training_artifacts import ensure_no_evaluation


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "test-commit"


def canonical_fixture(root: Path) -> tuple[Path, Path, dict]:
    data = root / "data"
    manifest_dir = root / "manifest"
    manifest_dir.mkdir()
    rows = []
    for split, sequence in (("train", "seq01"), ("val", "seq02"), ("test", "seq03")):
        image = data / "images" / split / f"{sequence}_frame0001.jpg"
        label = data / "labels" / split / f"{sequence}_frame0001.txt"
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        Image.new("RGB", (16, 12), (10, 20, 30)).save(image)
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
        rows.append({
            "image_path": f"fixture/images/{split}/{image.name}",
            "label_path": f"fixture/labels/{split}/{label.name}",
            "sequence_id": sequence, "split": split, "width": "16", "height": "12",
            "object_count": "1", "plastic_count": "1", "bio_count": "0", "rov_count": "0",
        })
    manifest = manifest_dir / "canonical_split_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "images": 3,
        "splits": {name: {"images": 1} for name in ("train", "val", "test")},
    }
    (manifest_dir / "canonical_split_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    cfg = {
        "trash_root": str(data), "out_dir": str(root / "output"), "split_mode": "sequence_70_15_15",
        "seed": 42, "split_seed": 42, "class_names": ["plastic", "bio", "rov"],
        "leakage": {"sequence_regex": r"^(.+?)_frame[0-9]+", "strict_sequence_regex": True},
        "provenance": {"canonical_manifest": str(manifest), "split_manifest_sha256": digest},
        "run": {"evaluate": False, "train_yolo": False, "train_frcnn": False, "train_ssd": False},
    }
    return data, manifest, cfg


class CanonicalMembershipTests(unittest.TestCase):
    def test_manifest_is_the_only_split_source_and_materialization_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, cfg = canonical_fixture(root)
            splits, source = experiment.load_canonical_split_records(cfg, ROOT)
            self.assertEqual({name: len(rows) for name, rows in splits.items()}, {"train": 1, "val": 1, "test": 1})
            frame = experiment.materialize_yolo_dataset(splits, root / "materialized", cfg["class_names"], True)
            report = experiment.verify_materialized_split_membership(frame, splits, cfg, root / "audit.json")
            self.assertEqual(source["missing_images"], 0)
            self.assertEqual(report["images_moved_from_canonical_split"], 0)
            self.assertEqual(report["sequences_moved_from_canonical_split"], 0)
            self.assertEqual(report["sequence_overlap"], 0)

    def test_extra_image_and_label_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, manifest, _ = canonical_fixture(root)
            extra_image = data / "images/train/extra_frame0001.jpg"
            extra_label = data / "labels/train/extra_frame0001.txt"
            Image.new("RGB", (8, 8)).save(extra_image)
            extra_label.write_text("", encoding="utf-8")
            _, _, errors = inspect(manifest, data)
            self.assertTrue(any("extra image" in error for error in errors))
            self.assertTrue(any("extra label" in error for error in errors))

    def test_missing_image_and_label_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, manifest, _ = canonical_fixture(root)
            (data / "images/test/seq03_frame0001.jpg").unlink()
            (data / "labels/test/seq03_frame0001.txt").unlink()
            _, _, errors = inspect(manifest, data)
            self.assertTrue(any("missing image" in error for error in errors))
            self.assertTrue(any("missing label" in error for error in errors))


class ValidationContractTests(unittest.TestCase):
    def test_out_of_image_box_is_not_silently_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "bad.txt"
            label.write_text("0 0.95 0.5 0.2 0.2\n", encoding="utf-8")
            objects, _, errors, audit = parse_label(label)
            self.assertEqual(objects, 0)
            self.assertEqual(audit["boxes_requiring_clipping"], 1)
            self.assertTrue(any("requires clipping" in error for error in errors))
            with self.assertRaisesRegex(RuntimeError, "clipping-required"):
                experiment.read_yolo_label(label, 3, (640, 480))

    def test_full_decode_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, manifest, _ = canonical_fixture(root)
            (data / "images/val/seq02_frame0001.jpg").write_bytes(b"not-an-image")
            fingerprint, _, errors = inspect(manifest, data)
            self.assertEqual(fingerprint["full_image_decode_failures"], 1)
            self.assertTrue(any("image decode failure" in error for error in errors))

    def test_river_is_not_required_for_training_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"run": {"evaluate": False}, "river_root": None, "provenance": {}}
            self.assertFalse(experiment.validate_river_provenance_if_requested(cfg, Path(tmp)))

    def test_river_provenance_is_required_when_evaluation_is_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"run": {"evaluate": True}, "river_root": "/river", "provenance": {}}
            with self.assertRaisesRegex(RuntimeError, "River fingerprint"):
                experiment.validate_river_provenance_if_requested(cfg, Path(tmp))

    def test_training_only_run_writes_no_evaluation_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            cfg = {"seed": 42, "class_names": ["plastic", "bio", "rov"], "run": {
                "train_yolo": True, "train_frcnn": False, "train_ssd": False, "evaluate": False,
            }}
            paths = experiment.PreparedPaths(Path("unused"), None, Path("unused"), None, out / "summary.csv")
            with patch.object(experiment, "train_yolo_detector", return_value=(object(), {}, float("nan"), float("nan"))):
                experiment.run_single_seed(paths, out, cfg, torch.device("cpu"))
            ensure_no_evaluation(out)
            self.assertFalse((out / "results_overall_test.csv").exists())

    def test_yolo_stage2_uses_last_not_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage1 = Path(tmp)
            (stage1 / "weights").mkdir()
            (stage1 / "weights/best.pt").write_bytes(b"best")
            (stage1 / "weights/last.pt").write_bytes(b"last")
            self.assertEqual(experiment.yolo_stage2_initialization_checkpoint(stage1).name, "last.pt")

    def test_atomic_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            experiment.atomic_torch_save({"value": torch.tensor([1, 2])}, path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(".last.pt.tmp").exists())
            self.assertTrue(torch.equal(torch.load(path, weights_only=False)["value"], torch.tensor([1, 2])))


class ExecutionLifecycleTests(unittest.TestCase):
    def test_verified_state_and_manifest_bind_final_execution_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            cfg = yaml.safe_load((ROOT / "config_runpod_yolo_seed42.yaml").read_text(encoding="utf-8"))
            output = temp / "output"
            cfg["out_dir"] = str(output)
            config_path = temp / "config.yaml"
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            environment = temp / "environment.txt"
            environment.write_text("safe-environment", encoding="utf-8")
            control = temp / "control"

            def fake_training(*_args, **_kwargs):
                output.mkdir(parents=True)
                return 0

            def fake_verifier(*_args, **_kwargs):
                (output / "training_artifact_manifest.json").write_text(
                    json.dumps({"status": "VERIFIED_COMPLETE", "evidence": {}}), encoding="utf-8",
                )
                return types.SimpleNamespace(returncode=0)

            with patch.object(runner, "load_and_validate_config", return_value=cfg), \
                 patch.object(runner, "source_commit", return_value=COMMIT), \
                 patch.object(runner, "monitor_gpu", return_value=None), \
                 patch.object(runner, "stream_process", side_effect=fake_training), \
                 patch.object(runner.subprocess, "run", side_effect=fake_verifier):
                result = runner.run("yolo", config_path, ROOT, COMMIT, "official", environment, control)
            execution = control / "executions" / "tmp_yolo.json"
            # The state filename derives from output.parent.name (the temporary directory name).
            execution = next((control / "executions").glob("*_yolo.json"))
            record = json.loads(execution.read_text(encoding="utf-8"))
            manifest = json.loads((output / "training_artifact_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "VERIFIED_COMPLETE")
            self.assertEqual(record["status"], "VERIFIED_COMPLETE")
            self.assertEqual(manifest["evidence"]["execution"]["sha256"], sha256_file(execution))


if __name__ == "__main__":
    unittest.main()
