import tempfile
import unittest
import random
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from marine_3model_experiment import (
    YoloRecord, class_agnostic_detection_nms_once, exclude_conflicting_duplicate_groups,
    checkpoint_identity, config_sha256, capture_rng_state, grouped_split, make_loader, restore_rng_state, sequence_id, validate_no_split_leakage,
    validate_resume_checkpoint,
)


class SequenceSplitTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.cfg = {
            "trash_root": str(self.root),
            "leakage": {"sequence_regex": r"^(.+?)_frame[0-9]+", "strict_sequence_regex": True},
        }

    def record(self, name):
        return YoloRecord(self.root / name, None)

    @staticmethod
    def exact_checkpoint(cfg):
        generator = torch.Generator().manual_seed(123)
        identity = checkpoint_identity(cfg, "frcnn")
        return {
            "model": {}, "optimizer": {}, "scheduler": {}, "scaler": {},
            "epoch": 87, "completed_epoch": 87, "next_epoch": 88,
            "stage": "all", "stage_epoch": 77, "architecture": "frcnn",
            "cfg": {"provenance": cfg["provenance"].copy()}, "best_map": 0.4, "best_epoch": 80,
            "training_history": [{"epoch": 87}], "checkpoint_identity": identity,
            "training_config_sha256": config_sha256(cfg), "dataset_sha256": identity["dataset_sha256"],
            "split_sha256": identity["split_manifest_sha256"], "git_commit": identity["git_commit"],
            "seed": identity["seed"], "experiment_id": identity["experiment_id"],
            "dataloader_generator_state": generator.get_state(),
            "sampler_state": {"strategy": "epoch_seeded_generator", "seed": 129, "next_global_epoch": 88},
            **capture_rng_state(),
        }

    def test_prefix_is_sequence_not_class(self):
        self.assertEqual(sequence_id(self.record("obj0309_frame0000130.jpg"), self.cfg), "obj0309")
        self.assertEqual(sequence_id(self.record("bio0015_frame0000154.jpg"), self.cfg), "bio0015")

    def test_all_frames_stay_together(self):
        records = [self.record(f"obj{i:04d}_frame{frame:07d}.jpg") for i in range(12) for frame in range(i + 1)]
        splits = grouped_split(records, self.cfg, seed=42)
        validate_no_split_leakage(splits, self.cfg)
        owners = {}
        for split, items in splits.items():
            for item in items:
                seq = sequence_id(item, self.cfg)
                self.assertIn(seq, owners | {seq: split})
                owners.setdefault(seq, split)
                self.assertEqual(owners[seq], split)

    def test_unknown_filename_is_rejected(self):
        with self.assertRaises(RuntimeError):
            sequence_id(self.record("unstructured-name.jpg"), self.cfg)

    def test_training_loader_drops_singleton_final_batch(self):
        image_dir = self.root / "images" / "train"
        label_dir = self.root / "labels" / "train"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(17):
            Image.new("RGB", (32, 32)).save(image_dir / f"obj{index:04d}_frame0000001.jpg")
            (label_dir / f"obj{index:04d}_frame0000001.txt").write_text(
                "0 0.5 0.5 0.25 0.25\n", encoding="utf-8"
            )
        loader = make_loader(
            self.root, "train", 32, 3, batch=16, workers=0,
            shuffle=False, drop_last=True,
        )
        batches = list(loader)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0][0]), 16)

    def test_resume_requires_complete_training_state(self):
        with self.assertRaisesRegex(RuntimeError, "missing state"):
            validate_resume_checkpoint({"model": {}}, {"provenance": {}}, "frcnn")

    def test_resume_requires_matching_fingerprints(self):
        cfg = {"provenance": {"dataset_sha256": "dataset-a", "split_manifest_sha256": "split-a"}}
        checkpoint = self.exact_checkpoint(cfg)
        checkpoint["cfg"]["provenance"]["dataset_sha256"] = "dataset-b"
        with self.assertRaisesRegex(RuntimeError, "dataset_sha256"):
            validate_resume_checkpoint(checkpoint, cfg, "frcnn")

    def test_resume_accepts_exact_identity(self):
        provenance = {"dataset_sha256": "dataset-a", "split_manifest_sha256": "split-a"}
        cfg = {"provenance": provenance}
        checkpoint = self.exact_checkpoint(cfg)
        validate_resume_checkpoint(checkpoint, cfg, "frcnn")

    def test_epoch_seeded_sampler_order_matches_split_resume(self):
        def order(epoch):
            generator = torch.Generator().manual_seed(42 + epoch)
            return torch.randperm(31, generator=generator).tolist()
        uninterrupted = [order(epoch) for epoch in range(1, 5)]
        session_a = [order(epoch) for epoch in range(1, 3)]
        session_b = [order(epoch) for epoch in range(3, 5)]
        self.assertEqual(uninterrupted, session_a + session_b)

    def test_rng_capture_restore_replays_python_numpy_and_torch(self):
        random.seed(42)
        np_state = __import__("numpy")
        np_state.random.seed(42)
        torch.manual_seed(42)
        state = capture_rng_state()
        expected = (random.random(), np_state.random.rand(), torch.rand(1).item())
        restore_rng_state(state)
        actual = (random.random(), np_state.random.rand(), torch.rand(1).item())
        self.assertEqual(expected, actual)

    def test_training_config_hash_ignores_session_control(self):
        base = {"seed": 42, "training": {"resume": False}, "run": {"quick_debug": False}, "provenance": {}}
        first = {**base, "session_control": {"stop_after_stage2_epoch": 60, "session_id": "session_a_clean"}}
        second = {**base, "training": {"resume": True}, "session_control": {"stop_after_stage2_epoch": 100, "session_id": "session_b_resume"}}
        self.assertEqual(config_sha256(first), config_sha256(second))

    def test_river_conflicting_duplicate_group_is_fully_excluded(self):
        image_a = self.root / "a.png"
        image_b = self.root / "b.png"
        image_c = self.root / "c.png"
        Image.new("RGB", (8, 8), "red").save(image_a)
        image_b.write_bytes(image_a.read_bytes())
        Image.new("RGB", (8, 8), "blue").save(image_c)
        label_a = self.root / "a.txt"
        label_b = self.root / "b.txt"
        label_c = self.root / "c.txt"
        label_a.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        label_b.write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        label_c.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        kept = exclude_conflicting_duplicate_groups(
            [YoloRecord(image_a, label_a), YoloRecord(image_b, label_b), YoloRecord(image_c, label_c)],
            self.root / "excluded.csv",
        )
        self.assertEqual([record.image for record in kept], [image_c])

    def test_framework_class_agnostic_nms_is_single_pass(self):
        import torchvision.models.detection.roi_heads as roi_heads_module

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])
        original = roi_heads_module.box_ops.batched_nms
        self.assertEqual(len(original(boxes, scores, labels, 0.5)), 2)
        model = SimpleNamespace(roi_heads=SimpleNamespace(nms_thresh=0.7))
        with class_agnostic_detection_nms_once(model, 0.5):
            self.assertEqual(len(roi_heads_module.box_ops.batched_nms(boxes, scores, labels, 0.5)), 1)
            self.assertEqual(model.roi_heads.nms_thresh, 0.5)
        self.assertIs(roi_heads_module.box_ops.batched_nms, original)
        self.assertEqual(model.roi_heads.nms_thresh, 0.7)


if __name__ == "__main__":
    unittest.main()
