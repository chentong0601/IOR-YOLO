"""Engineering workflow safety checks; no model training or Test evaluation."""

import csv
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/19_e01_local_engineering.py"
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
        self.assertEqual(local.formal_runner.FORMAL_DEVICE, "cuda:0")

    def test_platform_paths_resolve_for_mac_windows_and_kaggle(self):
        self.assertTrue(local.is_absolute_for_platform("/Users/test/d2.zip", "Darwin"))
        self.assertTrue(local.is_absolute_for_platform(r"C:\datasets\d2.zip", "Windows"))
        self.assertFalse(local.is_absolute_for_platform(r"datasets\d2.zip", "Windows"))
        self.assertTrue(local.is_absolute_for_platform("/kaggle/input/d2/dataset.zip", "Linux"))
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


if __name__ == "__main__":
    unittest.main()
