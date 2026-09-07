from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

from marine_3model_experiment import capture_rng_state, validate_session_a_contract
from tools.smoke_comparison_contract import (
    COMPARISON_FILENAME, FAIL_STATUS, NUMERICAL_POLICY, PASS_STATUS, REQUIRED_CHECKS,
    SCHEMA_VERSION, build_comparison, comparison_path, validate_comparison,
    verify_comparison_artifact, write_comparison,
)

PROJECT = Path(__file__).resolve().parents[1]
SUFFIX = Path("runs/seed_42/torchvision/frcnn/last.pt")


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref, self.resumed = self.root / "reference", self.root / "resumed"
        self.state = {
            "model": {"weight": torch.tensor([0.25, -0.5])},
            "optimizer": {"param_groups": [{"lr": 0.0}],
                          "state": {0: {"momentum_buffer": torch.tensor([0.1, 0.2])}}},
            "scheduler": {"T_max": 2, "last_epoch": 2, "_last_lr": [0.0]},
            "scaler": {}, "best_map": 0.5, "val_metrics": {"map": 0.5},
            "best_epoch": 3, "epoch": 3, "completed_epoch": 3, "next_epoch": 4,
            "stage": "all", "stage_epoch": 2,
            "training_history": [{"epoch": e, "lr": lr, "loss": 0.25}
                                 for e, lr in zip([1, 2, 3], [0.0, 0.0005, 0.0])],
            "checkpoint_identity": {"seed": 42}, "training_config_sha256": "c" * 64,
            "dataset_sha256": "d" * 64, "split_sha256": "e" * 64,
            "git_commit": "a" * 40, "seed": 42, "experiment_id": "unit_smoke",
            "architecture": "frcnn", "dataloader_generator_state": torch.get_rng_state(),
            "sampler_state": {"next_epoch": 4}, **capture_rng_state(),
        }
        for out in (self.ref, self.resumed):
            (out / SUFFIX).parent.mkdir(parents=True)
            torch.save(self.state, out / SUFFIX)
        self.report = build_comparison(self.ref / SUFFIX, self.resumed / SUFFIX)

    def verify(self):
        return verify_comparison_artifact(self.resumed, self.ref, 3)

    def save_unvalidated(self, report):
        comparison_path(self.resumed).write_text(json.dumps(report), encoding="utf-8")

    def test_canonical_atomic_schema_and_pass(self):
        with mock.patch("tools.smoke_comparison_contract.os.replace", wraps=os.replace) as replace:
            path = write_comparison(self.resumed, self.report)
        self.assertEqual(path, self.resumed / COMPARISON_FILENAME)
        self.assertEqual(replace.call_count, 1)
        self.assertFalse(path.with_name("." + path.name + ".tmp").exists())
        self.assertEqual(self.verify()["status"], PASS_STATUS)
        self.assertEqual(self.report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(self.report["numerical_policy"], NUMERICAL_POLICY)
        self.assertTrue(set(REQUIRED_CHECKS) <= self.report["checks"].keys())

    def test_missing_status_rejected(self):
        report = copy.deepcopy(self.report); del report["status"]
        self.save_unvalidated(report)
        with self.assertRaisesRegex(RuntimeError, "schema fields missing"):
            self.verify()

    def test_fail_rejected_and_safely_persisted(self):
        report = copy.deepcopy(self.report)
        report["status"] = FAIL_STATUS; report["checks"]["model_state"] = False
        write_comparison(self.resumed, report)
        with self.assertRaisesRegex(RuntimeError, "failed checks"):
            self.verify()

    def test_false_check_cannot_be_marked_pass(self):
        report = copy.deepcopy(self.report); report["checks"]["optimizer_state"] = False
        self.save_unvalidated(report)
        with self.assertRaisesRegex(RuntimeError, "status contradicts"):
            self.verify()

    def test_required_check_missing_and_non_boolean_rejected(self):
        for mutate in ("missing", "integer"):
            report = copy.deepcopy(self.report)
            if mutate == "missing": del report["checks"]["model_state"]
            else: report["checks"]["model_state"] = 1
            self.save_unvalidated(report)
            with self.assertRaises(RuntimeError): self.verify()

    def test_diagnostics_only_never_accepted(self):
        diagnostic = self.root / "diagnostics"; diagnostic.mkdir()
        (diagnostic / COMPARISON_FILENAME).write_text(json.dumps(self.report))
        with self.assertRaisesRegex(RuntimeError, "canonical comparison file missing"):
            self.verify()

    def test_clean_resume_directory_not_created(self):
        future = self.root / "not_created_by_process_a"
        comparison_path(future)
        self.assertFalse(future.exists())
        with self.assertRaisesRegex(RuntimeError, "COMPARISON_OUTPUT_MISSING"):
            write_comparison(future, self.report)
        self.assertFalse(future.exists())

    def test_tensor_value_difference_fails_even_when_structure_matches(self):
        changed = copy.deepcopy(self.state)
        changed["model"]["weight"][0] += 1e-5
        torch.save(changed, self.resumed / SUFFIX)
        report = build_comparison(self.ref / SUFFIX, self.resumed / SUFFIX)
        self.assertEqual(report["status"], FAIL_STATUS)
        self.assertFalse(report["checks"]["model_state"])
        self.assertEqual(report["details"]["model_state"]["differing_tensor_count"], 1)

    def test_momentum_difference_fails_with_identical_param_groups(self):
        changed = copy.deepcopy(self.state)
        changed["optimizer"]["state"][0]["momentum_buffer"][0] += 1
        torch.save(changed, self.resumed / SUFFIX)
        report = build_comparison(self.ref / SUFFIX, self.resumed / SUFFIX)
        self.assertTrue(report["checks"]["optimizer"])
        self.assertFalse(report["checks"]["optimizer_state"])
        self.assertEqual(report["status"], FAIL_STATUS)

    def test_metric_history_rng_scaler_and_sampler_differences_rejected(self):
        mutations = {
            "best_metric": lambda cp: cp.update(best_map=0.51),
            "training_history": lambda cp: cp["training_history"][2].update(loss=0.3),
            "rng_state": lambda cp: cp["torch_cpu_rng_state"].fill_(42),
            "scaler_state": lambda cp: cp.update(scaler={"scale": 128}),
            "sampler_state": lambda cp: cp["sampler_state"].update(next_epoch=2),
        }
        for check, mutate in mutations.items():
            with self.subTest(check=check):
                changed = copy.deepcopy(self.state); mutate(changed)
                torch.save(changed, self.resumed / SUFFIX)
                report = build_comparison(self.ref / SUFFIX, self.resumed / SUFFIX)
                self.assertFalse(report["checks"][check])
                self.assertEqual(report["status"], FAIL_STATUS)
                # Report contains field names only, never serialized random generator values.
                self.assertNotIn("tensor(", json.dumps(report))
                self.assertNotIn("python_random_state\": [", json.dumps(report))

    def test_verifier_recomputes_and_rejects_forged_true_flags(self):
        write_comparison(self.resumed, self.report)
        changed = copy.deepcopy(self.state); changed["model"]["weight"][0] += 1
        torch.save(changed, self.resumed / SUFFIX)
        with self.assertRaisesRegex(RuntimeError, "failed checks"):
            self.verify()

    def test_checkpoint_sha_binding_and_tolerance_cannot_be_relaxed(self):
        for mutation in ("sha", "tolerance"):
            report = copy.deepcopy(self.report)
            if mutation == "sha": report["resumed"]["sha256"] = "0" * 64
            else: report["numerical_policy"]["atol"] = 0.1
            self.save_unvalidated(report)
            with self.assertRaises(RuntimeError): self.verify()


class LocalSmokeIntegrationTests(unittest.TestCase):
    def test_reference_process_a_process_b_comparison_and_real_verifier(self):
        spec = importlib.util.spec_from_file_location(
            "v11_launcher_under_test", PROJECT / "tmp/frcnn_checkpoint_metadata_smoke/smoke.py")
        launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            export = work / "export"; diagnostics = export / "diagnostics"
            artifacts = export / "artifacts"; configs = export / "configs"
            for parent in (export, diagnostics, artifacts, configs): parent.mkdir(exist_ok=True)
            reference, resumed = artifacts / "reference_run", artifacts / "resume_run"
            base = yaml.safe_load((PROJECT / "config_runpod_frcnn_seed42.yaml").read_text())
            fixture = PROJECT / "tests/fixtures/smoke_resume_child.py"
            env = dict(os.environ, CANONICAL_GIT_COMMIT="a" * 40,
                       CANONICAL_EXPERIMENT_ID="SMOKE_UNIT_FIXTURE", PYTHONUNBUFFERED="1")
            with mock.patch.multiple(launcher, ROOT=export, WORKING=work, CONFIGS=configs), \
                 mock.patch.object(launcher, "data_root", return_value=work / "fixture_data"):
                for phase, target, resume, stop in (
                    ("reference", reference, False, 2),
                    ("process_a", resumed, False, 1),
                    ("process_b", resumed, True, 2),
                ):
                    if resume:
                        status = validate_session_a_contract(resumed, 2)
                        self.assertEqual(status["next_epoch"], 3)
                        last = Path(status["last_checkpoint"])
                    else:
                        self.assertFalse(target.exists()); last = None
                    cfg = launcher.config_for(base, target, resume, phase, stop, last)
                    config = launcher.write_config(phase, cfg)
                    launcher.execute(phase, [sys.executable, "-B", "-u", str(fixture),
                                             "--config", str(config)], PROJECT, env, diagnostics / phase)
                    self.assertTrue((diagnostics / phase / "stdout.log").is_file())
                    self.assertTrue((diagnostics / phase / "stderr.log").is_file())
                    self.assertFalse((target / "stdout.log").exists())
                report = build_comparison(reference / SUFFIX, resumed / SUFFIX)
                self.assertEqual(report["status"], PASS_STATUS)
                write_comparison(resumed, report)
                environment = export / "environment.json"
                launcher.atomic_json(environment, {"classification": "SMOKE_DEBUG_ONLY"})
                launcher.execute("verifier", [
                    sys.executable, "-B", "-u", str(PROJECT / "tools/verify_smoke_artifacts.py"),
                    "--out-dir", str(resumed), "--reference-out-dir", str(reference),
                    "--config", str(config), "--environment", str(environment),
                ], PROJECT, env, diagnostics / "verifier")
                manifest = json.loads((resumed / "smoke_artifact_manifest.json").read_text())
                self.assertEqual(manifest["status"], "SMOKE_VERIFIED_FOR_PIPELINE_ONLY")
                self.assertIn("RESUME_NEXT_EPOCH=3",
                              (diagnostics / "process_b/stdout.log").read_text())
                self.assertEqual(json.loads((diagnostics / "verifier/execution.json").read_text())["return_code"], 0)
                launcher.bundle()
                self.assertTrue((work / "frcnn_smoke_v12_diagnostic_bundle.zip").is_file())
