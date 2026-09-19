"""Synthetic ZIP checks for full-decode and manifest generation."""

from __future__ import annotations

import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is needed for full decode")
class D2IntegrityTest(unittest.TestCase):
    def test_decode_hash_and_cross_split_relations_without_raw_writes(self) -> None:
        from PIL import Image, ImageEnhance

        image = Image.new("RGB", (16, 16))
        for y in range(16):
            for x in range(16):
                image.putpixel((x, y), (x * 10, y * 10, (x + y) * 5))

        def encoded(source: Image.Image, size: tuple[int, int] | None = None) -> bytes:
            output = io.BytesIO()
            (source.resize(size) if size else source).save(output, format="JPEG", quality=95)
            return output.getvalue()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            manifest_dir = root / "manifests"
            manifest_dir.mkdir()
            records = {
                "train": ("apple.jpg", image),
                "val": ("apple_brightness.jpg", ImageEnhance.Brightness(image).enhance(1.1)),
                "test": ("other.jpg", image),
            }
            with zipfile.ZipFile(archive, "w") as output:
                for split, (name, source) in records.items():
                    for variant in ("", "_resize"):
                        folder = split + variant
                        output.writestr(f"dataset/{folder}/{name}",
                                        encoded(source, (24, 24) if variant else None))
                        annotation = {"record": {"filename": name, "regions": [{
                            "shape_attributes": {"name": "polyline",
                                                 "all_points_x": [0, 10, 0],
                                                 "all_points_y": [0, 0, 10]},
                            "region_attributes": {"name": "immature apple"}}]}}
                        output.writestr(f"dataset/{folder}.json", json.dumps(annotation))
            original_bytes = archive.read_bytes()
            result = subprocess.run([
                sys.executable, str(PROJECT / "scripts" / "05_audit_d2_integrity.py"),
                "--archive", str(archive), "--manifest-dir", str(manifest_dir)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["decode_status_counts"], {"decoded": 6})
            self.assertEqual(summary["cross_split_named_families"], 1)
            self.assertEqual(summary["family_status_counts"]["Supported"], 1)
            self.assertEqual(len(summary["exact_hash_groups_cross_split"]), 1)
            with (manifest_dir / "files_sha256.csv").open(newline="", encoding="utf-8") as stream:
                files = list(csv.DictReader(stream))
            with (manifest_dir / "duplicate_report.csv").open(newline="", encoding="utf-8") as stream:
                relations = list(csv.DictReader(stream))
            self.assertEqual(len(files), 6)
            self.assertTrue(all(row["sha256"] and row["decode_status"] == "decoded" for row in files))
            self.assertTrue(any(row["relation_type"] == "exact_duplicate" and
                                row["split_a"] != row["split_b"] for row in relations))
            for name in ("files_sha256.csv", "duplicate_report.csv", "class_counts.csv"):
                raw = (manifest_dir / name).read_bytes()
                self.assertNotIn(b"\r\n", raw, name)
                self.assertTrue(raw.endswith(b"\n"), name)
            self.assertEqual(archive.read_bytes(), original_bytes)


if __name__ == "__main__":
    unittest.main()
