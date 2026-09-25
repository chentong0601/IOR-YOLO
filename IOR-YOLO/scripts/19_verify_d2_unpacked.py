"""Verify an unpacked D2 v4 directory against tracked byte-level evidence.

This never creates a ZIP and never treats file counts as identity evidence.
It compares all 2,812 JPEG bytes plus all six annotation JSON bytes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path("/kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508")
DEFAULT_FILES = ROOT / "data/manifests/files_sha256.csv"
DEFAULT_IDENTITY = ROOT / "configs/data/d2_unpacked_identity.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def relative_image_path(row: dict[str, str]) -> Path:
    folder = row["split"] + ("_resize" if row["variant"] == "resize" else "")
    return Path(folder) / row["filename"]


def verify(root: Path, files_manifest: Path = DEFAULT_FILES,
           identity_path: Path = DEFAULT_IDENTITY) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"unpacked D2 root not found: {root}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity["schema_version"] != "d2-unpacked-identity-v1":
        raise ValueError("unexpected unpacked identity schema")
    if digest(files_manifest) != identity["files_sha256_manifest_sha256"]:
        raise ValueError("tracked image-hash evidence changed")
    rows = manifest_rows(files_manifest)
    if len(rows) != identity["expected_image_files"]:
        raise ValueError("image-hash evidence cardinality changed")

    expected = {relative_image_path(row): row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("duplicate image path in tracked hash evidence")
    actual = {path.relative_to(root) for path in root.rglob("*")
              if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg")}
    missing, extra = sorted(expected.keys() - actual), sorted(actual - expected.keys())
    if missing or extra:
        raise ValueError(f"unpacked image set differs: missing={len(missing)}, extra={len(extra)}")

    counts = Counter()
    for relative, row in expected.items():
        path = root / relative
        if path.stat().st_size != int(row["bytes"]) or digest(path) != row["sha256"]:
            raise ValueError(f"unpacked image bytes differ: {relative.as_posix()}")
        counts[f"{row['split']}:{row['variant']}"] += 1

    expected_json = identity["json_files"]
    actual_json = {path.relative_to(root).as_posix() for path in root.glob("*.json") if path.is_file()}
    if actual_json != set(expected_json):
        raise ValueError(f"annotation JSON set differs: {sorted(actual_json)}")
    for name, evidence in expected_json.items():
        path = root / name
        if path.stat().st_size != evidence["bytes"] or digest(path) != evidence["sha256"]:
            raise ValueError(f"annotation JSON bytes differ: {name}")

    return {
        "status": "VERIFIED",
        "identity_scope": "all 2812 JPEG files and all six annotation JSON files",
        "source_kind": "unpacked_directory",
        "source_root": str(root),
        "source_zip_sha256_provenance": identity["source_zip_sha256"],
        "files_sha256_manifest_sha256": identity["files_sha256_manifest_sha256"],
        "identity_evidence_sha256": digest(identity_path),
        "image_files": len(rows),
        "json_files": len(expected_json),
        "counts": dict(sorted(counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--files-manifest", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.files_manifest, args.identity), indent=2))


if __name__ == "__main__":
    main()
