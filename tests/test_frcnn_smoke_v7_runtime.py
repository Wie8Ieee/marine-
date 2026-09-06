from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from tools.frcnn_smoke_v7_runtime import ProcessFailure, SmokeExport


class V7ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.work = Path(self.temp.name) / "working"
        self.work.mkdir(); self.export = SmokeExport(self.work / "frcnn_smoke_v7_export", self.work)
        self.env = dict(os.environ, PYTHONUNBUFFERED="1")

    def tearDown(self): self.temp.cleanup()

    def assert_failure(self, stage, command):
        with self.assertRaises(ProcessFailure) as caught: self.export.run_process(stage, command, self.work, self.env)
        self.export.failure(stage, caught.exception); self.export.finalize()
        result = json.loads((self.export.root / "smoke_result.json").read_text())
        self.assertEqual(result["smoke_status"], "FAIL")
        self.assertTrue((self.export.root / "stage_markers.jsonl").exists())
        self.assertTrue((self.export.root / stage / "stdout.log").exists())
        self.assertTrue((self.export.root / stage / "stderr.log").exists())
        self.assertTrue((self.work / "frcnn_smoke_v7_bundle.zip").exists())
        self.assertEqual(result["return_code"], 7)
    def test_failure_cases_preserve_export(self):
        for stage in ("reference", "process_a", "process_b", "comparison", "verifier"):
            with self.subTest(stage=stage):
                self.setUp(); self.assert_failure(stage, [sys.executable, "-u", "-c", "import sys; sys.exit(7)"]); self.tearDown()

    def test_import_and_contract_failures_preserve_result(self):
        for stage in ("import", "process_a_contract", "checkpoint_empty", "checkpoint_sha_mismatch"):
            with self.subTest(stage=stage):
                self.setUp(); self.export.failure(stage, RuntimeError(stage)); self.export.finalize()
                self.assertEqual(json.loads((self.export.root / "smoke_result.json").read_text())["failed_stage"], stage)
                self.tearDown()

    def test_success_creates_result_and_bundle(self):
        self.export.marker("VERIFIER_PASSED"); self.export.finalize(passed=True)
        self.assertEqual(json.loads((self.export.root / "smoke_result.json").read_text())["smoke_status"], "PASS")
        self.assertTrue((self.work / "frcnn_smoke_v7_bundle.zip").exists())

    def test_v8_checkout_is_separate_from_clone_diagnostics(self):
        source = Path("tmp/frcnn_checkpoint_metadata_smoke/smoke.py").read_text(encoding="utf-8")
        self.assertIn('RUNTIME / "checkout"', source)
        self.assertIn('ROOT / "diagnostics" / "repository_clone"', source)
        self.assertIn('if repo.exists()', source)
        self.assertNotIn("shutil.rmtree", source)
