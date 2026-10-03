"""E02 preflight/runner safety: E01 immutability, run isolation, Test lock, manifest.

No training, no CUDA and no dataset rebuild: the gates are exercised through the
pure configuration/inspection functions plus temporary synthetic artifacts.
"""

import csv
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("e02_runner_test", ROOT / "scripts/22_e02_ordinal_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from ior_yolo.trainers.ordinal import OrdinalRunSettings  # noqa: E402
from ior_yolo.utils.tensorboard_logger import TENSORBOARD_DIRNAME, tensorboard_roundtrip  # noqa: E402


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(runner.CONFIG.read_text(encoding="utf-8"))

    def test_config_identity_and_frozen_inheritance(self):
        loaded = runner.load_config()
        self.assertEqual(loaded["experiment_id"], "e02_yolo11n_seg_ordinal")
        self.assertEqual((loaded["task"], loaded["model"], loaded["pretrained"]), ("segment", "yolo11n-seg.pt", True))
        self.assertEqual(loaded["training_seed"], 0)
        self.assertEqual(loaded["split_seed_64"], 13436313853456744620)

    def test_execution_environment_is_identical_to_e01(self):
        e01 = yaml.safe_load(runner.E01_CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(self.config["execution_environment"], e01["execution_environment"])
        self.assertEqual(runner.assert_train_parity.__name__, "assert_train_parity")

    def test_train_block_parity_is_enforced(self):
        e01 = yaml.safe_load(runner.E01_CONFIG.read_text(encoding="utf-8"))
        runner.assert_train_parity(self.config["train"], e01["train"])
        for key, value in (("epochs", 101), ("imgsz", 320), ("batch", 4), ("seed", 1), ("optimizer", "SGD")):
            tampered = {**self.config["train"], key: value}
            with self.assertRaises(ValueError):
                runner.assert_train_parity(tampered, e01["train"])
        with self.assertRaises(ValueError):
            runner.assert_train_parity({**self.config["train"], "augmentation": {"mosaic": 0.5}}, e01["train"])

    def test_ordinal_block_feeds_the_trainer_settings(self):
        mapping = runner.ordinal_config_from_config(self.config)
        settings = OrdinalRunSettings.from_mapping(mapping)
        self.assertEqual(settings.lambda_ord, 0.5)
        self.assertEqual(settings.stage_order, ("immature apple", "semi-mature apple", "mature apple"))
        self.assertTrue(settings.log_batch_scalars)

    def test_ordinal_block_tampering_is_rejected(self):
        for key, value in (("lambda_ord", 1.0), ("encoding", [0, 2, 1]), ("semantics", "unordered"),
                           ("lambda_selection", "tuned_on_val"),
                           ("class_order", ["mature apple", "semi-mature apple", "immature apple"]),
                           ("loss", "cross_entropy"),
                           ("cumulative_targets", {"immature": [0.0, 0.0], "semi-mature": [0.0, 1.0],
                                                   "mature": [1.0, 1.0]})):
            tampered = yaml.safe_load(yaml.safe_dump(self.config))
            tampered["ordinal"][key] = value
            with self.assertRaises(ValueError):
                runner.ordinal_config_from_config(tampered)

    def test_lambda_is_fixed_and_recorded(self):
        self.assertEqual(self.config["ordinal"]["lambda_ord"], 0.5)
        self.assertEqual(self.config["ordinal"]["lambda_selection"], "predeclared_before_training")
        self.assertEqual(self.config["run"]["final_test"], runner.FINAL_TEST_POLICY)


class E01ImmutabilityTest(unittest.TestCase):
    def test_frozen_e01_artifacts_are_unchanged(self):
        report = runner.assert_e01_immutable(runner.load_config())
        self.assertTrue(report["ok"])
        self.assertEqual(report["status"], "verified")
        self.assertGreaterEqual(report["files_checked"], 9)
        self.assertIn("scripts/17_analyze_e01_results.py", report["verified"])
        self.assertIn("configs/experiments/e01_yolo11n_seg.yaml", report["verified"])

    def test_tampering_and_missing_files_are_reported(self):
        with tempfile.TemporaryDirectory(prefix="e02-e01-immutability-") as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            (root / "configs/trained.yaml").write_text("changed\n", encoding="utf-8")
            config = {"e01_reference": {"files_sha256": {
                "configs/trained.yaml": runner.sha(Path(__file__)),
                "configs/absent.yaml": "0" * 64}}}
            report = runner.e01_immutability_report(config, repo_root=root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["status"], "E01_MUTATED")
        self.assertEqual(report["missing"], ["configs/absent.yaml"])
        self.assertEqual([item["path"] for item in report["mismatched"]], ["configs/trained.yaml"])
        with self.assertRaises(RuntimeError):
            runner.assert_e01_immutable(config, repo_root=root)


class IsolationAndSplitLockTest(unittest.TestCase):
    def test_e02_run_directory_is_isolated_from_e01(self):
        report = runner.assert_run_isolation()
        self.assertTrue(report["isolated"])
        self.assertIn("e02_yolo11n_seg_ordinal", report["run_dir"])
        self.assertIn("e01_yolo11n_seg", report["e01_run_dir"])
        self.assertNotIn("e01", Path(report["run_dir"]).name)

    def test_overlapping_or_misnamed_run_directories_are_refused(self):
        e01 = runner.E01_RUN_DIR
        for candidate in (e01, e01 / "seed_0", e01.parent, runner.RUNS_ROOT / "e02_yolo11n_seg_ordinal" / "seed_1",
                          runner.RUNS_ROOT / "e01_yolo11n_seg/seed_0"):
            with self.assertRaises(RuntimeError):
                runner.assert_run_isolation(candidate, e01)

    def test_entrypoint_exposes_no_test_command(self):
        self.assertNotIn("test", runner.COMMANDS)
        self.assertEqual(runner.COMMANDS, ("preflight", "smoke", "tensorboard-smoke", "train", "val"))
        self.assertEqual(runner.EVALUATION_SPLIT, "val")
        self.assertNotIn("split", inspect.signature(runner.evaluate).parameters)
        self.assertNotIn("final_test", inspect.signature(runner.evaluate).parameters)
        self.assertNotIn('"--final-test"', inspect.getsource(runner))
        self.assertIn("Final Test", inspect.getsource(runner.evaluate))
        self.assertIn("val", runner.evaluate.__doc__)

    def test_derived_class_order_must_stay_ascending(self):
        with tempfile.TemporaryDirectory(prefix="e02-derived-") as temporary:
            root = Path(temporary)
            names = {0: "immature apple", 1: "semi-mature apple", 2: "mature apple"}
            (root / "dataset.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
            report = runner.verify_derived_classes(root)
            self.assertEqual(report["nc"], 3)
            self.assertEqual(report["ascending_encoding"], [0, 1, 2])
            swapped = {0: "immature apple", 1: "mature apple", 2: "semi-mature apple"}
            (root / "dataset.yaml").write_text(yaml.safe_dump({"names": swapped}), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.verify_derived_classes(root)
            (root / "dataset.yaml").unlink()
            with self.assertRaises(FileNotFoundError):
                runner.verify_derived_classes(root)

    def test_real_derived_dataset_matches_the_frozen_classes(self):
        if not (runner.DATASET / "dataset.yaml").is_file():
            self.skipTest("derived E01 dataset not present locally")
        report = runner.verify_derived_classes(runner.DATASET)
        self.assertEqual(list(report["names"].values()), list(runner.dataset.CLASSES))


class ManifestAndArgumentTest(unittest.TestCase):
    def test_manifest_template_declares_the_e02_fields(self):
        template = yaml.safe_load(runner.TEMPLATE.read_text(encoding="utf-8"))
        for key in ("ordinal", "ordinal_diagnostics", "ordinal_loss_first_epoch", "ordinal_loss_last_train_epoch",
                    "tensorboard", "tensorboard_event_files", "e01_reference", "e01_immutability",
                    "train_e01_parity", "loss_names", "final_test"):
            self.assertIn(key, template)
        self.assertEqual(template["experiment_id"], "e02_yolo11n_seg_ordinal")
        self.assertEqual(template["schema_version"], "e02-run-manifest-v1")
        self.assertEqual(template["final_test"]["status"], runner.FINAL_TEST_POLICY)
        self.assertIsNone(template["test_metrics_mask"])

    def test_manifest_round_trip_is_yaml_safe(self):
        class LibraryVersion(str):
            pass

        with tempfile.TemporaryDirectory(prefix="e02-manifest-") as temporary:
            run_dir = Path(temporary) / "seed_0"
            run_dir.mkdir()
            with patch.object(runner, "RUN_DIR", run_dir):
                runner.write_manifest({"run_id": "e02_seed0_test", "pytorch": LibraryVersion("2.14.1+cpu"),
                                       "ordinal": {"lambda_ord": 0.5}, "test_metrics_mask": None})
                loaded = runner.read_manifest()
        self.assertEqual(loaded["pytorch"], "2.14.1+cpu")
        self.assertIs(type(loaded["pytorch"]), str)
        self.assertEqual(loaded["ordinal"]["lambda_ord"], 0.5)

    def test_smoke_arguments_are_disposable_and_formal_arguments_are_frozen(self):
        config = runner.load_config()
        smoke = runner.smoke_train_args(config, "data.yaml", batch=8, device="cpu")
        self.assertEqual((smoke["epochs"], smoke["fraction"], smoke["workers"]), (1, 0.02, 0))
        self.assertEqual((smoke["imgsz"], smoke["batch"], smoke["device"]), (640, 8, "cpu"))
        self.assertFalse(smoke["save"] or smoke["plots"] or smoke["val"])
        self.assertEqual(smoke["hsv_h"], config["train"]["augmentation"]["hsv_h"])
        with tempfile.TemporaryDirectory(prefix="e02-args-") as temporary:
            run_dir = Path(temporary) / "e02_yolo11n_seg_ordinal/seed_0"
            with patch.object(runner, "RUN_DIR", run_dir):
                formal = runner.resolve_train_args(config, "data.yaml", batch_override=None)
                fallback = runner.resolve_train_args(config, "data.yaml", batch_override=4)
        self.assertEqual((formal["epochs"], formal["imgsz"], formal["batch"], formal["device"]), (100, 640, 8, 0))
        self.assertEqual(formal["seed"], 0)
        self.assertEqual(formal["name"], "seed_0")
        self.assertTrue(formal["pretrained"])
        self.assertEqual(fallback["batch"], 4)


REQUIRED_COLUMNS = ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss", "train/ordinal_loss",
                    "val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss", "val/ordinal_loss",
                    "metrics/precision(B)", "metrics/mAP50(B)", "metrics/precision(M)", "metrics/mAP50-95(M)",
                    "lr/pg0")


def write_results_csv(run_dir, *, ordinal=0.01, columns=REQUIRED_COLUMNS, epochs=2):
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for _ in range(epochs):
            row = [1.0 for _ in columns]
            for position, name in enumerate(columns):
                if name.endswith("ordinal_loss"):
                    row[position] = ordinal
            writer.writerow(row)


class TrainingArtifactContractTest(unittest.TestCase):
    def test_complete_artifacts_pass_the_dashboard_gate(self):
        with tempfile.TemporaryDirectory(prefix="e02-artifacts-") as temporary:
            run_dir = Path(temporary) / "seed_0"
            write_results_csv(run_dir)
            tensorboard_roundtrip(run_dir / TENSORBOARD_DIRNAME)
            report = runner.verify_training_artifacts(run_dir)
        self.assertEqual(report["ordinal_loss_first_epoch"], 0.01)
        self.assertEqual(report["ordinal_loss_last_epoch"], 0.01)
        self.assertEqual(report["tensorboard"]["missing_tags"], [])
        self.assertEqual(sorted(report["tensorboard"]["event_files"]), report["tensorboard"]["event_files"])

    def test_missing_loss_columns_are_refused(self):
        with tempfile.TemporaryDirectory(prefix="e02-artifacts-columns-") as temporary:
            run_dir = Path(temporary) / "seed_0"
            write_results_csv(run_dir, columns=tuple(name for name in REQUIRED_COLUMNS
                                                     if name != "train/ordinal_loss"))
            tensorboard_roundtrip(run_dir / TENSORBOARD_DIRNAME)
            with self.assertRaises(RuntimeError):
                runner.verify_training_artifacts(run_dir)

    def test_negative_ordinal_loss_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="e02-artifacts-ordinal-") as temporary:
            run_dir = Path(temporary) / "seed_0"
            write_results_csv(run_dir, ordinal=-1.0)
            tensorboard_roundtrip(run_dir / TENSORBOARD_DIRNAME)
            with self.assertRaises(RuntimeError):
                runner.verify_training_artifacts(run_dir)

    def test_training_without_a_dashboard_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="e02-artifacts-tb-") as temporary:
            run_dir = Path(temporary) / "seed_0"
            write_results_csv(run_dir)
            with self.assertRaises(RuntimeError):
                runner.verify_training_artifacts(run_dir)
            (run_dir / TENSORBOARD_DIRNAME).mkdir()
            tensorboard_roundtrip(run_dir / TENSORBOARD_DIRNAME, tags=("epoch",))
            with self.assertRaises(RuntimeError):
                runner.verify_training_artifacts(run_dir)

    def test_missing_results_csv_is_refused(self):
        with tempfile.TemporaryDirectory(prefix="e02-artifacts-empty-") as temporary:
            with self.assertRaises(FileNotFoundError):
                runner.verify_training_artifacts(Path(temporary) / "seed_0")

    def test_tensorboard_smoke_writes_the_required_contract_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="e02-tb-smoke-") as temporary:
            output = Path(temporary) / "tb_smoke"
            result = runner.tensorboard_smoke(output_dir=output)
            self.assertIn("TensorBoard smoke passed", result["status"])
            self.assertEqual(result["report"]["missing_tags"], [])
            self.assertTrue((output / TENSORBOARD_DIRNAME).is_dir())
            self.assertIn("train/ordinal_loss", result["required_tags"])
            with self.assertRaises(FileExistsError):
                runner.tensorboard_smoke(output_dir=output)

    def test_tensorboard_gate_requires_the_e02_run_directory(self):
        with self.assertRaises(RuntimeError):
            runner.tensorboard_gate(Path(tempfile.gettempdir()) / "somewhere-else")


if __name__ == "__main__":
    unittest.main()
