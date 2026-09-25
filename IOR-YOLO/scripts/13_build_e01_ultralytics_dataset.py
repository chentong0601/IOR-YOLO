"""Rebuild/validate E01 YOLO segmentation data from immutable D2 artifacts.

No split selection, augmentation, resizing or raw annotation mutation occurs.
The derived output is disposable and ignored by Git. Requires Pillow.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data/raw/multistage_apple_v4/dataset-20260508.zip"
DEFAULT_MANIFESTS = ROOT / "data/manifests"
DEFAULT_PROTOCOL = ROOT / "configs/data/d2_frozen_protocol.yaml"
DEFAULT_OUTPUT = ROOT / "data/processed/d2_e01_ultralytics"
DEFAULT_IDENTITY = ROOT / "configs/data/d2_unpacked_identity.json"
SPLITS = ("train", "val", "test")
CLASSES = ("immature apple", "semi-mature apple", "mature apple")
NAMES = ("d2_experiment_pool_frozen.csv", "d2_split_frozen.csv", "d2_exclusions_frozen.csv")
EXPECTED = {"train": (769, 646, 297, 512), "val": (165, 143, 65, 115),
            "test": (165, 126, 63, 105)}
DATASET_YAML = """# Generated from D2 frozen protocol; dataset root is this file's directory.
train: images/train
val: images/val
test: images/test
names:
  0: immature apple
  1: semi-mature apple
  2: mature apple
"""

spec = importlib.util.spec_from_file_location("d2_unpacked_identity", Path(__file__).with_name("19_verify_d2_unpacked.py"))
unpacked_identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(unpacked_identity)


class DirectorySource:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def namelist(self) -> list[str]:
        return [f"{self.root.name}/{path.relative_to(self.root).as_posix()}"
                for path in self.root.rglob("*") if path.is_file()]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, member: str) -> bytes:
        prefix = f"{self.root.name}/"
        if not member.startswith(prefix):
            raise ValueError(f"raw member prefix differs from directory root: {member}")
        path = (self.root / member.removeprefix(prefix)).resolve()
        if self.root not in path.parents:
            raise ValueError(f"unsafe raw member: {member}")
        return path.read_bytes()


def open_source(path: Path):
    return DirectorySource(path) if path.is_dir() else zipfile.ZipFile(path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_protocol(path: Path) -> dict:
    """Frozen YAML is a flat mapping of JSON-compatible values, by design."""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, value = line.split(": ", 1)
        result[key] = json.loads(value)
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def frozen_inputs(source: Path, manifests: Path, protocol_path: Path,
                  identity_path: Path = DEFAULT_IDENTITY) -> tuple[dict, list[dict], list[dict], dict]:
    protocol = read_protocol(protocol_path)
    if protocol["protocol_version"] != "d2-v4-stage2b5-v1" or protocol["split_ratio"] != "70/15/15":
        raise ValueError("unexpected frozen protocol/version/ratio")
    if source.is_dir():
        identity = unpacked_identity.verify(source, manifests / "files_sha256.csv", identity_path)
    elif source.name == protocol["zip_filename"] and sha(source) == protocol["zip_sha256"]:
        identity = {"status": "VERIFIED", "source_kind": "original_zip",
                    "source_zip_sha256_provenance": protocol["zip_sha256"]}
    else:
        raise ValueError("raw source does not match frozen ZIP or unpacked byte-level evidence")
    if sha(protocol_path) != "09db366f059ac933a68a05ec2c8de40a4df1600b859819ca1507d3cf4d719870":
        raise ValueError("frozen protocol YAML changed; E01 requires explicit version review")
    for name in NAMES:
        if sha(manifests / name) != protocol["frozen_csv_sha256"][name]:
            raise ValueError(f"frozen manifest changed: {name}")
    pool, split, excluded = (read_csv(manifests / name) for name in NAMES)
    if len(pool) != 1099 or len(split) != 1099 or len(excluded) != 1713:
        raise ValueError("frozen manifest cardinality changed")
    key = lambda r: (r["official_split"], r["filename"], r["variant"])
    if {key(r): r for r in pool} != {key(r): r for r in split}:
        raise ValueError("pool and split manifests disagree")
    if set(map(key, pool)) & set(map(key, excluded)):
        raise ValueError("frozen exclusion overlaps pool")
    if any(r["include"] != "true" or r["variant"] != "original" or
           r["augmentation_type"] != "none" or r["representation_type"] != "plain_base" or
           r["final_split"] not in SPLITS for r in split):
        raise ValueError("forbidden representation/assignment in frozen pool")
    if any(r["final_split"] != "excluded" for r in excluded):
        raise ValueError("excluded image given a split")
    groups, guards = defaultdict(set), defaultdict(set)
    for r in split:
        groups[r["source_group_id"]].add(r["final_split"])
        guards[r["split_guard_cluster_id"]].add(r["final_split"])
    if any(len(x) > 1 for x in (*groups.values(), *guards.values())):
        raise ValueError("source group or guard cluster crosses split")
    return protocol, split, excluded, identity


def region_rows(regions: list[dict], width: int, height: int, *, u01: bool) -> tuple[list[str], Counter]:
    if u01:
        if len(regions) != 4 or regions[3].get("region_attributes", {}).get("name") in CLASSES:
            raise ValueError("U01 raw annotation changed")
        regions = regions[:3]
    lines = []
    counts = Counter()
    for region in regions:
        label = region.get("region_attributes", {}).get("name")
        if label not in CLASSES:
            raise ValueError(f"unknown class: {label!r}")
        shape = region.get("shape_attributes", {})
        xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
        if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys) or len(xs) < 3:
            raise ValueError("polygon needs >=3 paired points")
        points = []
        for x, y in zip(xs, ys):
            if not isinstance(x, (float, int)) or not isinstance(y, (float, int)) or not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("non-finite polygon coordinate")
            nx, ny = x / width, y / height
            if not (0 <= nx <= 1 and 0 <= ny <= 1):
                raise ValueError(f"polygon coordinate outside image: {x}, {y} / {width}, {height}")
            points.append((nx, ny))
        if len(set(points)) < 3 or abs(sum(points[i][0] * points[(i+1) % len(points)][1] -
                                       points[(i+1) % len(points)][0] * points[i][1]
                                       for i in range(len(points)))) <= 1e-12:
            raise ValueError("degenerate polygon")
        lines.append(" ".join([str(CLASSES.index(label)),
                               *(f"{v:.10f}" for point in points for v in point)]))
        counts[label] += 1
    return lines, counts


def annotations_in_source(source, split: list[dict]) -> dict[str, dict[str, list[dict]]]:
    members = set(source.namelist())
    json_members = {r["raw_annotation_zip_member"] for r in split}
    records = {}
    for member in json_members:
        if member not in members:
            raise ValueError(f"missing raw JSON: {member}")
        raw = json.loads(source.read(member))
        parsed = {}
        for record in raw.values():
            regions = record["regions"]
            if record["filename"] in parsed:
                raise ValueError("duplicate filename in raw JSON")
            parsed[record["filename"]] = list(regions.values()) if isinstance(regions, dict) else regions
        records[member] = parsed
    return records


def expected_counters(split: list[dict]) -> dict[str, Counter]:
    counters = {name: Counter() for name in SPLITS}
    for row in split:
        c = counters[row["final_split"]]
        c["images"] += 1
        for key in ("immature_instances", "semi_mature_instances", "mature_instances"):
            c[key] += int(row[key])
    for name, expected in EXPECTED.items():
        actual = tuple(counters[name][key] for key in ("images", "immature_instances", "semi_mature_instances", "mature_instances"))
        if actual != expected:
            raise ValueError(f"frozen {name} counts differ: {actual} != {expected}")
    return counters


def build(source_path: Path, manifests: Path, protocol_path: Path, output: Path, *, replace: bool = False,
          identity_path: Path = DEFAULT_IDENTITY) -> dict:
    protocol, split, excluded, identity = frozen_inputs(source_path, manifests, protocol_path, identity_path)
    expected_counters(split)
    if output.exists() and not replace:
        raise FileExistsError(f"derived output already exists: {output}; use --replace to rebuild")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e01-build-", dir=output.parent) as temp:
        staging = Path(temp) / "dataset"
        for part in SPLITS:
            (staging / "images" / part).mkdir(parents=True)
            (staging / "labels" / part).mkdir(parents=True)
        (staging / "dataset.yaml").write_text(DATASET_YAML, encoding="utf-8", newline="\n")
        statistics = {name: Counter() for name in SPLITS}
        image_keys = set()
        with open_source(source_path) as source:
            annotation_records = annotations_in_source(source, split)
            members = set(source.namelist())
            for row in split:
                member, part, name = row["raw_image_zip_member"], row["final_split"], row["filename"]
                if member not in members or not member.endswith("/" + name):
                    raise ValueError(f"missing/mismatched raw image: {member}")
                key = (part, name)
                if key in image_keys:
                    raise ValueError(f"filename collision in derived {part}: {name}")
                image_keys.add(key)
                raw = source.read(member)
                with Image.open(io.BytesIO(raw)) as image:
                    image.load()
                    if image.format != "JPEG":
                        raise ValueError(f"not JPEG: {member}")
                    width, height = image.size
                regions = annotation_records[row["raw_annotation_zip_member"]].get(name)
                if regions is None:
                    raise ValueError(f"image missing from raw JSON: {member}")
                u01 = (row["official_split"], name) == ("test", "IMG_54350.jpg")
                lines, counts = region_rows(regions, width, height, u01=u01)
                target_counts = tuple(int(row[k]) for k in ("immature_instances", "semi_mature_instances", "mature_instances"))
                if len(lines) != int(row["derived_target_count"]) or tuple(counts[x] for x in CLASSES) != target_counts:
                    raise ValueError(f"derived annotation differs from frozen manifest: {member}")
                if (row["dropped_raw_region_indices"] == "3") != u01:
                    raise ValueError("U01 drop index policy mismatch")
                (staging / "images" / part / name).write_bytes(raw)
                label = staging / "labels" / part / (Path(name).stem + ".txt")
                if label.exists():
                    raise ValueError(f"label-stem collision: {label.name}")
                label.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
                statistics[part]["images"] += 1
                statistics[part]["immature_instances"] += counts[CLASSES[0]]
                statistics[part]["semi_mature_instances"] += counts[CLASSES[1]]
                statistics[part]["mature_instances"] += counts[CLASSES[2]]
        for part in SPLITS:
            actual = tuple(statistics[part][key] for key in ("images", "immature_instances", "semi_mature_instances", "mature_instances"))
            if actual != EXPECTED[part]:
                raise ValueError(f"converted {part} counts differ: {actual}")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
    return {"status": "derived E01 segmentation dataset validated; no training run",
            "dataset_protocol_version": protocol["protocol_version"],
            "raw_zip_sha256": protocol["zip_sha256"],
            "raw_source_kind": identity["source_kind"],
            "raw_identity_status": identity["status"],
            "raw_identity_evidence_sha256": identity.get("identity_evidence_sha256"),
            "statistics": {part: dict(statistics[part]) for part in SPLITS},
            "excluded_representations": len(excluded), "derived_root": str(output),
            "u01_derived_targets": 3}


def validate_only(source_path: Path, manifests: Path, protocol_path: Path, output: Path,
                  identity_path: Path = DEFAULT_IDENTITY) -> dict:
    protocol, split, excluded, identity = frozen_inputs(source_path, manifests, protocol_path, identity_path)
    expected_counters(split)
    if (output / "dataset.yaml").read_text(encoding="utf-8") != DATASET_YAML:
        raise ValueError("derived dataset YAML differs")
    with open_source(source_path) as source:
        raw_annotations = annotations_in_source(source, split)
        for part in SPLITS:
            rows = [r for r in split if r["final_split"] == part]
            images = set((output / "images" / part).iterdir())
            labels = set((output / "labels" / part).iterdir())
            if len(images) != len(rows) or len(labels) != len(rows):
                raise ValueError(f"derived {part} image/label count mismatch")
            for row in rows:
                name = row["filename"]
                image = output / "images" / part / name
                label = output / "labels" / part / (Path(name).stem + ".txt")
                if image not in images or label not in labels:
                    raise ValueError(f"derived pair missing: {part}/{name}")
                raw = source.read(row["raw_image_zip_member"])
                if image.read_bytes() != raw:
                    raise ValueError(f"derived image changed: {part}/{name}")
                with Image.open(io.BytesIO(raw)) as im:
                    width, height = im.size
                regions = raw_annotations[row["raw_annotation_zip_member"]][name]
                expected_lines, _ = region_rows(regions, width, height,
                    u01=(row["official_split"], name) == ("test", "IMG_54350.jpg"))
                if label.read_text(encoding="utf-8") != "\n".join(expected_lines) + "\n":
                    raise ValueError(f"derived label changed: {part}/{name}")
    return {"status": "validated", "counts": EXPECTED,
            "protocol_version": protocol["protocol_version"], "exclusions": len(excluded),
            "raw_source_kind": identity["source_kind"],
            "raw_identity_status": identity["status"],
            "raw_identity_evidence_sha256": identity.get("identity_evidence_sha256")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--source", type=Path, help="original ZIP or verified unpacked dataset root")
    parser.add_argument("--archive", type=Path, help="deprecated alias for --source")
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    if args.source and args.archive:
        parser.error("use only one of --source or --archive")
    source = args.source or args.archive or DEFAULT_ARCHIVE
    result = (build(source, args.manifest_dir, args.protocol, args.output, replace=args.replace,
                    identity_path=args.identity)
              if args.command == "build" else
              validate_only(source, args.manifest_dir, args.protocol, args.output,
                            identity_path=args.identity))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
