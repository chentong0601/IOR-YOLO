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


def jpeg_header_with_size(width: int, height: int) -> bytes:
    # Header-only fixture: the ZIP audit reads SOF dimensions, never full pixel data.
    return (b"\xff\xd8" + b"\xff\xe0\x00\x04AB" + b"\xff\xc0\x00\x0b\x08"
            + height.to_bytes(2, "big") + width.to_bytes(2, "big")
            + b"\x01\x01\x11\x00" + b"\xff\xd9")


class DatasetAuditToolsTest(unittest.TestCase):
    def run_script(self, script: str, *args: str) -> dict:
        result = subprocess.run([sys.executable, str(PROJECT / "scripts" / script), *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
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

    def test_zip_structure_only_counts_file_members_and_labels_hints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("dataset/", b"")
                output.writestr("dataset/original/sample.jpg", b"example")
                output.writestr("dataset/640x480/sample.jpg", b"example")
                output.writestr("dataset/augmented/sample_gamma.png", b"example")
                output.writestr("dataset/annotations/via.json", b"{}")
                output.writestr("dataset/README.md", b"example")
                output.writestr("dataset/classes.txt", b"example")
                output.writestr("__MACOSX/._sample.jpg", b"sidecar")
            before = archive.read_bytes()
            report = self.run_script("01_inspect_dataset.py", "--archive", str(archive),
                                     "--structure-only")["archive"]
            structure = report["structure"]
            self.assertIsNone(report["sha256"])
            self.assertEqual(report["members"], 8)
            self.assertEqual(structure["file_members"], 7)
            self.assertEqual(structure["explicit_directory_members"], 1)
            self.assertEqual(structure["image_file_members_by_extension"], 3)
            self.assertEqual(structure["annotation_candidate_file_members_by_name_or_extension"], 1)
            self.assertEqual(structure["augmentation_name_hints"]["count"], 1)
            self.assertEqual(structure["top_level_directory_names"], ["__MACOSX", "dataset"])
            self.assertEqual(structure["resolution_name_hints"]["count"], 1)
            self.assertEqual(structure["repeated_image_basename_hints"]["basename_count"], 1)
            self.assertEqual(structure["readme_metadata_class_name_hints"]["count"], 2)
            self.assertEqual(archive.read_bytes(), before)

    def test_d2_zip_audit_pairs_records_and_flags_cross_split_named_derivative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "fixture.zip"
            originals = {"train": "fruit.png", "val": "fruit_brightness.png", "test": "other.png"}
            with zipfile.ZipFile(archive, "w") as output:
                for group, name in originals.items():
                    for folder in (group, group + "_resize"):
                        output.writestr(f"dataset/{folder}/{name}", tiny_png())
                        record = {"file": {"filename": name, "regions": [{
                            "shape_attributes": {"all_points_x": [0, 1, 0],
                                                 "all_points_y": [0, 0, 1]},
                            "region_attributes": {} if group == "test" else
                            {"name": "immature apple"}}]}}
                        output.writestr(f"dataset/{folder}.json", json.dumps(record))
            before = archive.read_bytes()
            result = self.run_script("04_audit_d2_zip.py", "--archive", str(archive))
            self.assertEqual(result["splits"]["train"]["image_file_members"], 1)
            self.assertEqual(result["splits"]["train"]["total_regions_instances"], 1)
            self.assertEqual(result["splits"]["train"]["class_instance_counts"],
                             {"immature apple": 1})
            self.assertEqual(result["splits"]["test"]["missing_class_name_regions"]["count"], 1)
            self.assertEqual(result["splits"]["train"]["image_header_dimensions"], {"2x2": 1})
            self.assertEqual(result["original_to_resize"]["train"]["annotation_comparison"][
                             "coordinate_pairs_with_residual_over_1_5px"], 0)
            self.assertEqual(result["original_cross_split_filename_checks"]["train_vs_val"][
                             "augmentation_suffix_family_overlap_candidates"]["count"], 1)
            self.assertEqual(archive.read_bytes(), before)

    def test_d2_zip_audit_reads_jpeg_sof_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "jpeg-fixture.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for group in ("train", "val", "test"):
                    for folder in (group, group + "_resize"):
                        output.writestr(f"dataset/{folder}/sample.jpg", jpeg_header_with_size(369, 277))
                        output.writestr(f"dataset/{folder}.json", json.dumps({
                            "sample": {"filename": "sample.jpg", "regions": []}}))
            result = self.run_script("04_audit_d2_zip.py", "--archive", str(archive))
            self.assertEqual(result["splits"]["train"]["image_header_dimensions"], {"369x277": 1})
            self.assertEqual(result["splits"]["train"]["image_header_errors"]["count"], 0)


if __name__ == "__main__":
    unittest.main()
