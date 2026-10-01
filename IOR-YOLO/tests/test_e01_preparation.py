"""Low-cost E01 frozen-input, conversion, format and run-schema checks."""

import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e01_dataset_test", ROOT / "scripts/13_build_e01_ultralytics_dataset.py")
e01 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e01)
spec = importlib.util.spec_from_file_location("e01_run_test", ROOT / "scripts/15_e01_run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def tree_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


class E01PreparationTest(unittest.TestCase):
    def test_polygon_format_and_rejection(self):
        region = {"region_attributes": {"name": "immature apple"},
                  "shape_attributes": {"all_points_x": [0, 10, 0], "all_points_y": [0, 0, 10]}}
        rows, counts = e01.region_rows([region], 10, 10, u01=False)
        self.assertEqual(rows, ["0 0.0000000000 0.0000000000 1.0000000000 0.0000000000 0.0000000000 1.0000000000"])
        self.assertEqual(counts["immature apple"], 1)
        region["shape_attributes"]["all_points_x"][1] = 11
        with self.assertRaises(ValueError):
            e01.region_rows([region], 10, 10, u01=False)

    def test_class_mapping_and_manifest_schema(self):
        self.assertEqual(e01.CLASSES, ("immature apple", "semi-mature apple", "mature apple"))
        self.assertIn("0: immature apple", e01.DATASET_YAML)
        self.assertIn("1: semi-mature apple", e01.DATASET_YAML)
        self.assertIn("2: mature apple", e01.DATASET_YAML)
        config = runner.load_config()
        template = yaml.safe_load(runner.TEMPLATE.read_text(encoding="utf-8"))
        required = ("run_id", "experiment_id", "git_commit", "timestamp_start_utc", "hostname",
                    "os", "platform", "execution_platform", "cloud_provider", "cloud_session_type",
                    "container_image_git_commit", "container_image_build_date", "python", "pytorch",
                    "python_executable", "ultralytics", "numpy", "opencv", "opencv_python",
                    "cuda_runtime", "gpu", "gpu_memory_bytes", "gpu_count", "visible_gpu_count",
                    "gpu_inventory", "selected_formal_device", "dataset_doi", "dataset_zip_sha256",
                    "raw_source_kind", "raw_source_path", "raw_identity_status",
                    "raw_identity_evidence_sha256", "files_sha256_manifest_sha256", "frozen_pool_sha256",
                    "frozen_split_sha256", "protocol_yaml_sha256", "model", "pretrained_weights_sha256",
                    "task", "training_seed", "imgsz", "epochs", "batch", "optimizer",
                    "learning_rate_initial", "weight_decay", "scheduler", "augmentation",
                    "best_checkpoint", "last_checkpoint", "validation_metrics_box", "validation_metrics_mask",
                    "test_metrics_box", "test_metrics_mask", "results_directory", "output_root", "notes")
        self.assertFalse(set(required) - set(template))
        self.assertEqual(config["training_seed"], 0)
        self.assertNotEqual(config["training_seed"], config["split_seed_64"])
        self.assertEqual(config["model"], "yolo11n-seg.pt")
        execution = config["execution_environment"]
        self.assertEqual(execution["formal_platform"], "Kaggle")
        self.assertEqual(execution["formal_device"], "cuda:0")
        self.assertTrue(execution["single_gpu_only"])
        self.assertEqual(execution["verified_kaggle_runtime"]["torch"], "2.10.0+cu128")
        self.assertEqual((config["task"], config["model"], config["training_seed"]),
                         ("segment", "yolo11n-seg.pt", 0))
        self.assertEqual((config["train"]["imgsz"], config["train"]["epochs"],
                          config["train"]["batch"], config["train"]["optimizer"]),
                         (640, 100, 8, "auto"))
        smoke_args = runner.smoke_train_args(config, "dataset.yaml", 8)
        self.assertEqual(smoke_args["device"], 0)
        self.assertNotIsInstance(smoke_args["device"], (list, tuple))

    def test_source_resolution_precedence_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            explicit = root / "explicit"
            configured = root / "configured"
            kaggle = root / "kaggle"
            local = root / "dataset.zip"
            for path in (explicit, configured, kaggle):
                path.mkdir()
            local.write_bytes(b"zip-placeholder")
            self.assertEqual(runner.resolve_source(
                explicit, expected_platform="kaggle",
                environ={"E01_D2_SOURCE": str(configured)},
                kaggle_default=kaggle, local_default=local), explicit.resolve())
            self.assertEqual(runner.resolve_source(
                None, expected_platform="kaggle",
                environ={"E01_D2_SOURCE": str(configured)},
                kaggle_default=kaggle, local_default=local), configured.resolve())
            with self.assertRaisesRegex(FileNotFoundError, "selected by E01_D2_SOURCE"):
                runner.resolve_source(
                    None, expected_platform="kaggle",
                    environ={"E01_D2_SOURCE": str(root / "missing-configured")},
                    kaggle_default=kaggle, local_default=local)
            self.assertEqual(runner.resolve_source(
                None, expected_platform="kaggle", environ={},
                kaggle_default=kaggle, local_default=local), kaggle.resolve())
            kaggle.rmdir()
            self.assertEqual(runner.resolve_source(
                None, expected_platform="kaggle", environ={},
                kaggle_default=kaggle, local_default=local), local.resolve())
            local.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "D2 source unresolved"):
                runner.resolve_source(None, expected_platform="kaggle", environ={},
                                      kaggle_default=kaggle, local_default=local)

    def test_real_conversion_and_deterministic_rebuild(self):
        if not e01.DEFAULT_ARCHIVE.is_file():
            self.skipTest("audited raw ZIP absent")
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            first = e01.build(e01.DEFAULT_ARCHIVE, e01.DEFAULT_MANIFESTS, e01.DEFAULT_PROTOCOL, a)
            second = e01.build(e01.DEFAULT_ARCHIVE, e01.DEFAULT_MANIFESTS, e01.DEFAULT_PROTOCOL, b)
            self.assertEqual(first["statistics"], second["statistics"])
            self.assertEqual(tree_hashes(a), tree_hashes(b))
            self.assertEqual(first["u01_derived_targets"], 3)
            self.assertEqual(sum(x["images"] for x in first["statistics"].values()), 1099)
            self.assertEqual(e01.validate_only(e01.DEFAULT_ARCHIVE, e01.DEFAULT_MANIFESTS,
                                                e01.DEFAULT_PROTOCOL, a)["status"], "validated")
            split = e01.read_csv(e01.DEFAULT_MANIFESTS / "d2_split_frozen.csv")
            u01 = [r for r in split if (r["official_split"], r["filename"]) == ("test", "IMG_54350.jpg")]
            self.assertEqual(len(u01), 1)
            label = a / "labels" / u01[0]["final_split"] / "IMG_54350.txt"
            lines = label.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(all(line.split()[0] in {"0", "1", "2"} for line in lines))
            for part, expected in e01.EXPECTED.items():
                self.assertEqual(first["statistics"][part]["images"], expected[0])
                self.assertEqual(len(list((a / "images" / part).glob("*.jpg"))), expected[0])
                self.assertEqual(len(list((a / "labels" / part).glob("*.txt"))), expected[0])
            self.assertFalse(any("_resize" in p.name or "_brightness" in p.name or "_noise" in p.name
                                 for p in (a / "images").rglob("*.jpg")))


if __name__ == "__main__":
    unittest.main()
