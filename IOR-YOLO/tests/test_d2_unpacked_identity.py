"""Low-cost byte-identity tests for automatically unpacked D2 input."""

import csv
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "d2_unpacked_identity_test", ROOT / "scripts/19_verify_d2_unpacked.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class D2UnpackedIdentityTest(unittest.TestCase):
    def test_exact_images_and_json_are_required(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            raw, evidence = base / "dataset-20260508", base / "evidence"
            (raw / "train").mkdir(parents=True)
            (raw / "train_resize").mkdir()
            evidence.mkdir()
            originals = {"train/a.jpg": b"original", "train_resize/a.jpg": b"resize"}
            for name, content in originals.items():
                (raw / name).write_bytes(content)
            annotations = {"train.json": b'{"a": 1}\n',
                           "train_resize.json": b'{"a": 2}\n'}
            for name, content in annotations.items():
                (raw / name).write_bytes(content)
            manifest = evidence / "files_sha256.csv"
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=("split", "variant", "filename", "bytes", "sha256"),
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerow({"split": "train", "variant": "original", "filename": "a.jpg",
                                 "bytes": len(originals["train/a.jpg"]), "sha256": sha(originals["train/a.jpg"])})
                writer.writerow({"split": "train", "variant": "resize", "filename": "a.jpg",
                                 "bytes": len(originals["train_resize/a.jpg"]),
                                 "sha256": sha(originals["train_resize/a.jpg"])})
            identity_file = evidence / "identity.json"
            identity_file.write_text(json.dumps({
                "schema_version": "d2-unpacked-identity-v1",
                "source_zip_sha256": "frozen-zip-provenance",
                "files_sha256_manifest_sha256": identity.digest(manifest),
                "expected_image_files": 2,
                "json_files": {name: {"bytes": len(content), "sha256": sha(content)}
                               for name, content in annotations.items()},
            }), encoding="utf-8")
            result = identity.verify(raw, manifest, identity_file)
            self.assertEqual(result["status"], "VERIFIED")
            self.assertEqual(result["image_files"], 2)
            (raw / "train/a.jpg").write_bytes(b"changed!")
            with self.assertRaisesRegex(ValueError, "image bytes differ"):
                identity.verify(raw, manifest, identity_file)


if __name__ == "__main__":
    unittest.main()
