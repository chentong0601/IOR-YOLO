"""Small synthetic checks for read-only acquisition and duplicate reports."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF))


def tiny_png() -> bytes:
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\x00" + b"\xff\x00\x00" * 2 + b"\x00" + b"\x00\xff\x00" * 2)
    return header + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", pixels) + png_chunk(b"IEND", b"")


class DatasetAuditToolsTest(unittest.TestCase):
    def run_script(self, script: str, *args: str) -> dict:
        result = subprocess.run([sys.executable, str(PROJECT / "scripts" / script), *args],
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def test_archive_and_tree_report_without_modifying_raw_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.png"
            image.write_bytes(tiny_png())
            record = {"sample": {"filename": "sample.png", "regions": [
                {"shape_attributes": {"all_points_x": [0, 1, 0],
                                      "all_points_y": [0, 0, 1]},
                 "region_attributes": {"stage": "immature"}}]}}
            (root / "annotations.json").write_text(json.dumps(record), encoding="utf-8")
            archive = root / "original.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.write(image, image.name)
            original = archive.read_bytes()

            report = self.run_script("01_inspect_dataset.py", "--archive", str(archive),
                                     "--root", str(root))
            self.assertEqual(report["archive"]["archive_format"], "zip")
            self.assertEqual(report["archive"]["members"], 1)
            self.assertEqual(report["tree"]["image_files"], 1)
            self.assertEqual(report["tree"]["annotation_records"], 1)
            self.assertEqual(report["tree"]["instances"], 1)
            self.assertEqual(report["tree"]["class_values"], {"stage=immature": 1})
            self.assertEqual(report["tree"]["invalid_polygons"], [])
            self.assertEqual(archive.read_bytes(), original)

    def test_exact_duplicate_is_only_a_candidate_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "sample.png").write_bytes(tiny_png())
            (root / "b" / "sample_resized.png").write_bytes(tiny_png())
            report = self.run_script("03_check_leakage.py", "--root", str(root))
            self.assertEqual(report["image_files"], 2)
            self.assertEqual(len(report["exact_duplicate_groups"]), 1)
            self.assertEqual(report["split_status"], "not created or validated")


if __name__ == "__main__":
    unittest.main()
