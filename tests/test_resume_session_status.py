import json
import tempfile
import unittest
from pathlib import Path

import torch

from marine_3model_experiment import (
    session_status_path, sha256_file, validate_session_a_contract, write_session_status,
)


class ResumeSessionStatusTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.last = self.root / "runs" / "seed_42" / "torchvision" / "frcnn" / "last.pt"
        self.last.parent.mkdir(parents=True)
        torch.save({"next_epoch": 3}, self.last)

    def status(self, **overrides):
        value = {
            "status": "SESSION_A_COMPLETE_READY_FOR_RESUME", "training_complete": False,
            "stage": "stage2", "completed_stage2_epoch": 1, "next_stage2_epoch": 2,
            "next_epoch": 3, "last_checkpoint": str(self.last.resolve()),
            "last_checkpoint_sha256": sha256_file(self.last),
        }
        value.update(overrides)
        return value

    def write(self, **overrides):
        path = self.root / "session_status.json"
        write_session_status(path, self.status(**overrides))
        return path

    def test_producer_and_consumer_share_output_root(self):
        self.assertEqual(session_status_path({"out_dir": str(self.root)}, self.root / "runs" / "seed_42"), self.root / "session_status.json")

    def test_atomic_status_write_and_controlled_stop_contract(self):
        path = self.write()
        self.assertTrue(path.is_file())
        self.assertFalse((self.root / ".session_status.json.tmp").exists())
        self.assertEqual(validate_session_a_contract(self.root, 2)["next_epoch"], 3)

    def test_final_and_failure_statuses_are_distinct(self):
        path = self.write(status="TRAINING_COMPLETE", training_complete=True)
        self.assertEqual(json.loads(path.read_text())["status"], "TRAINING_COMPLETE")
        path = self.write(status="FAILED", error_type="RuntimeError", error_message="boom")
        self.assertEqual(json.loads(path.read_text())["status"], "FAILED")

    def test_missing_status_is_a_clear_contract_error(self):
        with self.assertRaisesRegex(RuntimeError, "PROCESS_A_CONTRACT_FAILED — SESSION STATUS FILE MISSING"):
            validate_session_a_contract(self.root, 2)

    def test_malformed_status_is_a_clear_contract_error(self):
        (self.root / "session_status.json").write_text("{", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "SESSION STATUS JSON MALFORMED"):
            validate_session_a_contract(self.root, 2)

    def test_missing_checkpoint_blocks_process_b(self):
        self.write(last_checkpoint=str(self.root / "missing.pt"))
        with self.assertRaisesRegex(RuntimeError, "LAST CHECKPOINT MISSING OR EMPTY"):
            validate_session_a_contract(self.root, 2)

    def test_wrong_checkpoint_sha_blocks_process_b(self):
        self.write(last_checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(RuntimeError, "LAST CHECKPOINT SHA-256 MISMATCH"):
            validate_session_a_contract(self.root, 2)

    def test_next_epoch_must_match_checkpoint(self):
        self.write(next_epoch=4)
        with self.assertRaisesRegex(RuntimeError, "NEXT EPOCH INVALID"):
            validate_session_a_contract(self.root, 2)


if __name__ == "__main__":
    unittest.main()
