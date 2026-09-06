import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from tools.smoke_process_diagnostics import run_logged_process


class SmokeProcessDiagnosticsTests(unittest.TestCase):
    def test_failed_process_preserves_stdout_stderr_and_failure_record(self):
        root = Path(tempfile.mkdtemp())
        command = [sys.executable, "-u", "-c", "import sys; print('stdout-ok'); print('stderr-ok', file=sys.stderr); raise RuntimeError('expected failure')"]
        with self.assertRaises(Exception):
            run_logged_process(command, cwd=root, env=os.environ, output_root=root, label="process_b")
        self.assertIn("stdout-ok", (root / "process_b_stdout.log").read_text(encoding="utf-8"))
        self.assertIn("stderr-ok", (root / "process_b_stderr.log").read_text(encoding="utf-8"))
        failure = json.loads((root / "process_b_failure.json").read_text(encoding="utf-8"))
        self.assertEqual(failure["return_code"], 1)
        self.assertIn("CalledProcessError", failure["error_type"])
        execution = json.loads((root / "process_b_execution.json").read_text(encoding="utf-8"))
        self.assertEqual(execution["return_code"], 1)
        self.assertIn("-u", execution["command"])


if __name__ == "__main__":
    unittest.main()
