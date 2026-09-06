import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from tools.smoke_process_diagnostics import run_logged_process, write_stage_marker


class SmokeProcessDiagnosticsTests(unittest.TestCase):
    def test_failed_process_preserves_stdout_stderr_and_failure_record(self):
        root = Path(tempfile.mkdtemp())
        command = [sys.executable, "-u", "-c", "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr); raise RuntimeError('expected failure')"]
        with self.assertRaises(Exception):
            run_logged_process(command, cwd=root, env=os.environ, output_root=root / "resume_output", mirror_root=root / "diagnostics", label="process_b")
        self.assertIn("stdout-ok", (root / "resume_output" / "process_b_stdout.log").read_text(encoding="utf-8"))
        self.assertIn("stderr-ok", (root / "resume_output" / "process_b_stderr.log").read_text(encoding="utf-8"))
        failure = json.loads((root / "diagnostics" / "process_b_failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["return_code"], 1)
        self.assertIn("CalledProcessError", failure["error_type"])
        execution = json.loads((root / "diagnostics" / "process_b_execution.json").read_text(encoding="utf-8"))
        self.assertEqual(execution["return_code"], 1)
        self.assertIn("-u", execution["command"])

    def test_stage_marker_is_atomic_and_persistent(self):
        root = Path(tempfile.mkdtemp())
        marker = write_stage_marker(root, "process_b", "STARTING", note="diagnostics initialized")
        self.assertEqual(json.loads(marker.read_text(encoding="utf-8"))["state"], "STARTING")
        self.assertFalse((root / ".process_b_stage.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
