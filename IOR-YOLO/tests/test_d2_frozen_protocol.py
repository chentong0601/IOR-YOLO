"""Regression checks on the frozen D2 manifest generator and U01 conversion."""

import csv
import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/12_freeze_d2_protocol.py"
spec = importlib.util.spec_from_file_location("d2_frozen_protocol", SCRIPT)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)
ARCHIVE = ROOT / "data/raw/multistage_apple_v4/dataset-20260508.zip"
MANIFESTS = ROOT / "data/manifests"


class FrozenProtocolTest(unittest.TestCase):
    def test_derived_u01_removes_only_unknown_region(self):
        if not ARCHIVE.is_file():
            self.skipTest("raw ZIP unavailable")
        with zipfile.ZipFile(ARCHIVE) as archive:
            annotations = freeze.candidate.read_original_annotations(archive)
        raw = annotations[("test", "IMG_54350.jpg")]
        derived, dropped = freeze.derived_regions(raw, "test", "IMG_54350.jpg")
        self.assertEqual(len(raw), 4)
        self.assertEqual(len(derived), 3)
        self.assertEqual(derived, raw[:3])
        self.assertEqual(dropped, "3")
        self.assertEqual(len(freeze.candidate.valid_labels(derived)), 3)
        with self.assertRaises(ValueError):
            freeze.derived_regions(raw[:3], "test", "IMG_54350.jpg")

    def test_two_isolated_rebuilds_and_published_manifests(self):
        if not ARCHIVE.is_file():
            self.skipTest("raw ZIP unavailable")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = freeze.build(ARCHIVE, MANIFESTS, Path(a))
            second = freeze.build(ARCHIVE, MANIFESTS, Path(b))
            self.assertEqual(first, second)
            for name, digest in first["artifact_sha256"].items():
                self.assertEqual((Path(a) / name).read_bytes(), (Path(b) / name).read_bytes())
                published = (ROOT / "configs/data" if name.endswith(".yaml") else MANIFESTS) / name
                self.assertEqual(freeze.candidate.digest_file(published), digest)
                self.assertNotIn(b"\r", published.read_bytes())
            self.assertEqual(first["seed64"], 13436313853456744620)
            self.assertEqual(first["counts"]["train"]["images"], 769)
            self.assertEqual(first["counts"]["val"]["images"], 165)
            self.assertEqual(first["counts"]["test"]["images"], 165)
            self.assertEqual(sum(first["exclusions"].values()), 1713)
            with (MANIFESTS / freeze.NAMES[1]).open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1099)
            u01 = [r for r in rows if r["official_split"] == "test" and r["filename"] == "IMG_54350.jpg"]
            self.assertEqual(len(u01), 1)
            self.assertEqual(u01[0]["derived_target_count"], "3")
            self.assertEqual(u01[0]["dropped_raw_region_indices"], "3")
            self.assertNotIn("ignore", u01[0]["annotation_policy"].replace("no ignore", ""))
            self.assertTrue(all(r["variant"] == "original" and r["augmentation_type"] == "none"
                                for r in rows))
            self.assertTrue(all(n == 0 for n in first["relation_crossings"].values()))


if __name__ == "__main__":
    unittest.main()
