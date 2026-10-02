"""Engineering workflow safety checks; no model training or Test evaluation."""

import csv
import hashlib
import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/21_e01_local_engineering.py"
spec = importlib.util.spec_from_file_location("e01_local_engineering_test", SCRIPT)
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class E01LocalEngineeringTests(unittest.TestCase):
    def test_device_selection_is_explicit_and_single_gpu_only(self):
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            cuda=SimpleNamespace(is_available=lambda: True),
        )
        self.assertEqual(local.check_device("cpu", fake_torch), "cpu")
        self.assertEqual(local.check_device("mps", fake_torch), "mps")
        self.assertEqual(local.check_device("cuda:0", fake_torch), "cuda:0")
        with self.assertRaises(ValueError):
            local.check_device("cuda:1", fake_torch)
        with self.assertRaises(ValueError):
            local.check_device("0,1", fake_torch)
        execution = local.formal_runner.load_config()["execution_environment"]
        self.assertEqual(execution["formal_device"], "cuda:0")
        self.assertIs(execution["single_gpu_only"], True)

    def test_platform_paths_resolve_for_mac_windows_and_kaggle(self):
        self.assertTrue(local.is_absolute_for_platform("/Users/test/d2.zip", "Darwin"))
        self.assertTrue(local.is_absolute_for_platform("/kaggle/input/d2/dataset.zip", "Linux"))
        self.assertTrue(local.is_absolute_for_platform(r"C:\datasets\d2.zip", "Windows"))
        self.assertFalse(local.is_absolute_for_platform(r"datasets\d2.zip", "Windows"))
        with self.assertRaises(ValueError):
            local.is_absolute_for_platform("/data/d2.zip", "FreeBSD")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(local.resolve_path("data/d2.zip", base=root), (root / "data/d2.zip").resolve())

    def test_deterministic_subset_excludes_test_and_records_source_ids(self):
        rows = [
            {"final_split": split, "filename": f"{split}_{index}.jpg"}
            for split, count in (("train", 8), ("val", 4), ("test", 9))
            for index in range(count)
        ]
        first = local.select_engineering_rows(rows, 4, 2)
        second = local.select_engineering_rows(list(reversed(rows)), 4, 2)
        self.assertEqual(first, second)
        self.assertEqual([len(first[key]) for key in ("train", "val")], [4, 2])
        self.assertTrue(all(row["final_split"] in ("train", "val")
                            for selected in first.values() for row in selected))
        self.assertFalse(any("test_" in row["filename"]
                             for selected in first.values() for row in selected))

    def test_engineering_dataset_cannot_contain_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dataset.yaml").write_text(
                yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test"}),
                encoding="utf-8")
            manifest = {"test_access": False}
            with self.assertRaises(PermissionError):
                local.verify_subset(root, manifest)

    def test_formal_protocol_is_unchanged_and_uses_official_initialization(self):
        tracked = (
            ROOT / "configs/data/d2_frozen_protocol.yaml",
            ROOT / "configs/experiments/e01_yolo11n_seg.yaml",
            ROOT / "data/manifests/d2_experiment_pool_frozen.csv",
            ROOT / "data/manifests/d2_split_frozen.csv",
            ROOT / "data/manifests/d2_exclusions_frozen.csv",
        )
        before = {path: file_hash(path) for path in tracked}
        formal = local.verified_formal_config()
        self.assertEqual(local.formal_runner.formal_model_name(formal), "yolo11n-seg.pt")
        self.assertEqual(local.FORMAL_DEVICE, "cuda:0")
        self.assertEqual(formal["task"], "segment")
        self.assertEqual(formal["train"]["imgsz"], 640)
        self.assertEqual(formal["train"]["epochs"], 100)
        self.assertEqual(formal["train"]["batch"], 8)
        self.assertEqual(formal["train"]["optimizer"], "auto")
        self.assertEqual(formal["training_seed"], 0)
        self.assertEqual(before, {path: file_hash(path) for path in tracked})
        invalid = dict(formal, model="/tmp/engineering/last.pt")
        with self.assertRaises(ValueError):
            local.formal_runner.formal_model_name(invalid)

    def test_local_subset_does_not_mutate_frozen_artifacts(self):
        frozen = (
            ROOT / "configs/data/d2_frozen_protocol.yaml",
            ROOT / "data/manifests/d2_experiment_pool_frozen.csv",
            ROOT / "data/manifests/d2_split_frozen.csv",
            ROOT / "data/manifests/d2_exclusions_frozen.csv",
        )
        before = {path: file_hash(path) for path in frozen}
        rows = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            derived = root / "derived"
            manifests = root / "manifests"
            manifests.mkdir()
            names = {0: "immature apple", 1: "semi-mature apple", 2: "mature apple"}
            (derived / "dataset.yaml").parent.mkdir(parents=True)
            (derived / "dataset.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")
            for split, count in (("train", 4), ("val", 2), ("test", 3)):
                for index in range(count):
                    name = f"{split}_{index}.jpg"
                    rows.append({"final_split": split, "filename": name,
                                 "source_group_id": f"group-{split}-{index}"})
                    (derived / "images" / split).mkdir(parents=True, exist_ok=True)
                    (derived / "labels" / split).mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 8)).save(derived / "images" / split / name, format="JPEG")
                    (derived / "labels" / split / f"{Path(name).stem}.txt").write_text(
                        "0 0.1 0.1 0.9 0.1 0.5 0.9\n", encoding="utf-8")
            split_path = manifests / "d2_split_frozen.csv"
            with split_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("final_split", "filename", "source_group_id"))
                writer.writeheader()
                writer.writerows(rows)
            engineering_data = root / "engineering"
            with patch.object(local, "ENGINEERING_DATA", engineering_data), \
                    patch.object(local, "validate_derived_train_val", return_value={"train": {}, "val": {}}):
                result = local.build_subset(derived, manifests, "local-smoke")
                subset_manifest = yaml.safe_load(
                    (Path(result["subset"]) / "engineering_subset_manifest.yaml").read_text(encoding="utf-8"))
                self.assertFalse(subset_manifest["test_access"])
                self.assertEqual(set(subset_manifest["source_image_ids"]), {"train", "val"})
                self.assertNotIn("test", yaml.safe_load(
                    (Path(result["subset"]) / "dataset.yaml").read_text(encoding="utf-8")))
                self.assertEqual(len(subset_manifest["source_image_ids"]["train"]), 4)
                self.assertEqual(len(subset_manifest["source_image_ids"]["val"]), 2)
        self.assertEqual(before, {path: file_hash(path) for path in frozen})


    def test_run_manifest_serialization_accepts_library_version_strings(self):
        """Regression: torch.__version__ is a TorchVersion str subclass yaml.safe_dump rejects."""

        class LibraryVersion(str):
            """Stands in for torch.torch_version.TorchVersion."""

        manifest = {
            "run_id": "local-smoke_regression", "mode": "local-smoke", "device": "cpu",
            "environment": {"torch": LibraryVersion("2.14.1+cpu"), "cuda_available": False,
                            "ultralytics": "8.3.220"},
            "engineering_overrides": {"imgsz": 64, "epochs": 1, "batch": 1, "workers": 0,
                                      "validation_during_training": False, "amp": False},
            "source_image_ids": {"train": ["340.jpg"], "val": ["2280.jpg"]},
            "validation": None, "prediction_export": None, "analysis": None,
            "last_checkpoint_sha256": "a" * 64,
        }
        with self.assertRaises(yaml.representer.RepresenterError):
            yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)
        with tempfile.TemporaryDirectory() as temp:
            path = local.write_run_manifest(Path(temp), manifest)
            written = yaml.safe_load(path.read_text(encoding="utf-8"))
        expected = {**manifest, "environment": {"torch": "2.14.1+cpu", "cuda_available": False,
                                                "ultralytics": "8.3.220"}}
        self.assertEqual(written, expected)
        self.assertIs(type(written["environment"]["torch"]), str)
        source = inspect.getsource(local.run_engineering)
        self.assertIn("write_run_manifest(run_dir, manifest)", source)
        self.assertNotIn("yaml.safe_dump(manifest", source)

    def test_yaml_safe_reduces_values_and_rejects_unknown_objects(self):
        class LibraryVersion(str):
            pass

        safe = local.yaml_safe({"text": LibraryVersion("x"), "int": 1, "float": 1.5, "bool": True,
                                "none": None, "tuple": (1, "a"), "nested": {"list": [LibraryVersion("y")]}})
        self.assertEqual(safe, {"text": "x", "int": 1, "float": 1.5, "bool": True, "none": None,
                                "tuple": [1, "a"], "nested": {"list": ["y"]}})
        for key, expected_type in (("text", str), ("int", int), ("float", float), ("bool", bool)):
            self.assertIs(type(safe[key]), expected_type)
        for bad in (Path("run_manifest.yaml"), object(), {1, 2}):
            with self.assertRaises(TypeError):
                local.yaml_safe({"bad": bad})
        # Builtin dict keys are preserved so namespace-style mappings still round-trip.
        self.assertEqual(local.yaml_safe({0: "immature apple"}), {0: "immature apple"})

    def test_engineering_prediction_export_pins_validation_thresholds(self):
        """Regression: the engineering export must not inherit predict's conf=0.25 default."""

        class FakeModel:
            """Records predict kwargs and yields empty results for every image."""

            def __init__(self):
                self.calls = []

            def predict(self, **kwargs):
                self.calls.append(kwargs)
                source = Path(kwargs["source"])
                return iter(SimpleNamespace(path=str(image), boxes=[], masks=None, orig_shape=(8, 8))
                            for image in sorted(source.glob("*.jpg")))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            val_images = root / "subset" / "images" / "val"
            val_images.mkdir(parents=True)
            for name in ("a.jpg", "b.jpg"):
                Image.new("RGB", (8, 8)).save(val_images / name, format="JPEG")
            for name in ("run", "rejected_conf", "rejected_iou"):
                (root / name).mkdir()
            model = FakeModel()
            info = local.export_engineering_predictions(
                model, root / "subset", root / "run", "cpu", 128,
                conf=local.ENGINEERING_PREDICTION_CONF, iou=local.ENGINEERING_PREDICTION_IOU)
            self.assertEqual({key: info[key] for key in ("conf", "iou", "imgsz", "images", "predictions")},
                             {"conf": 0.001, "iou": 0.7, "imgsz": 128, "images": 2, "predictions": 0})
            self.assertEqual(Path(info["path"]).name, "predictions_val.json")
            self.assertTrue(Path(info["path"]).is_file())
            # The thresholds must reach ultralytics predict itself, not be implied.
            self.assertEqual(model.calls[0]["conf"], 0.001)
            self.assertEqual(model.calls[0]["iou"], 0.7)
            self.assertEqual(model.calls[0]["imgsz"], 128)
            self.assertEqual(set(model.calls[0]), {"source", "stream", "device", "imgsz", "conf",
                                                   "iou", "save", "verbose"})
            with self.assertRaises(ValueError):
                local.export_engineering_predictions(model, root / "subset", root / "rejected_conf", "cpu",
                                                     128, conf=None, iou=0.7)
            with self.assertRaises(ValueError):
                local.export_engineering_predictions(model, root / "subset", root / "rejected_iou", "cpu",
                                                     128, conf=0.001, iou=None)
            self.assertEqual(len(model.calls), 1)

    def test_engineering_prediction_export_declares_thresholds_explicitly(self):
        """Thresholds must be explicit, recorded, and equal to the pinned val values."""
        self.assertEqual(local.ENGINEERING_PREDICTION_CONF, 0.001)
        self.assertEqual(local.ENGINEERING_PREDICTION_IOU, 0.7)
        # ultralytics 8.3.220 Model.predict defaults to conf=0.25 (its validator uses
        # 0.001), so the engineering export must stay below that library default.
        self.assertLess(local.ENGINEERING_PREDICTION_CONF, 0.25)
        parameters = inspect.signature(local.export_engineering_predictions).parameters
        for name in ("conf", "iou"):
            self.assertIs(parameters[name].default, inspect.Parameter.empty)
            self.assertIs(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        export_source = inspect.getsource(local.export_engineering_predictions)
        self.assertIn("conf=conf", export_source)
        self.assertIn("iou=iou", export_source)
        self.assertNotIn("conf=0.25", export_source)
        run_source = inspect.getsource(local.run_engineering)
        self.assertIn("conf=ENGINEERING_PREDICTION_CONF", run_source)
        self.assertIn("iou=ENGINEERING_PREDICTION_IOU", run_source)
        self.assertIn('"prediction_export": prediction_info', run_source)
        self.assertIn('"prediction_export": prediction_info',
                      inspect.getsource(local.engineering_analysis))
        formal = (ROOT / "scripts/18_export_e01_predictions.py").read_text(encoding="utf-8")
        self.assertIn(f"conf={local.ENGINEERING_PREDICTION_CONF!r}", formal)
        self.assertIn(f"iou={local.ENGINEERING_PREDICTION_IOU!r}", formal)

    def test_formal_run_manifest_writer_is_yaml_safe(self):
        class LibraryVersion(str):
            pass

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "seed_0"
            run_dir.mkdir()
            with patch.object(local.formal_runner, "RUN_DIR", run_dir):
                local.formal_runner.write_manifest({"run_id": "e01_seed0_regression",
                                                    "pytorch": LibraryVersion("2.14.1+cpu")})
            written = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
        self.assertEqual(written["pytorch"], "2.14.1+cpu")
        self.assertIs(type(written["pytorch"]), str)


if __name__ == "__main__":
    unittest.main()
