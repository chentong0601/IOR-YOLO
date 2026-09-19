"""Read-only D2 package and extracted-tree inspection; no splitting or repair."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from collections import Counter
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ior_yolo.utils.io import image_size, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def archive_report(path: Path) -> dict:
    report = {"filename": path.name, "bytes": path.stat().st_size,
              "sha256": sha256_file(path), "archive_format": "unverified",
              "members": None, "unsafe_paths": []}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        report["archive_format"] = "zip"
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            names = archive.getnames()
        report["archive_format"] = "tar"
    else:
        return report
    report["members"] = len(names)
    report["unsafe_paths"] = [name for name in names if name.startswith("/") or ".." in Path(name).parts][:20]
    return report


def regions_of(record: dict) -> list:
    regions = record.get("regions", [])
    if isinstance(regions, dict):
        return list(regions.values())
    return regions if isinstance(regions, list) else []


def via_records(data: object) -> list[dict]:
    if not isinstance(data, dict):
        return []
    metadata = data.get("_via_img_metadata", data)
    if not isinstance(metadata, dict):
        return []
    return [value for value in metadata.values()
            if isinstance(value, dict) and "filename" in value and "regions" in value]


def polygon_area(xs: list[float], ys: list[float]) -> float:
    return abs(sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                   for i in range(len(xs)))) / 2


def inspect_tree(root: Path, verify_decode: bool = False) -> dict:
    images = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    annotations = [p for p in root.rglob("*.json") if p.is_file()]
    image_class = None
    if verify_decode:
        try:
            from PIL import Image
        except ImportError as exc:
            raise SystemExit("--verify-decode needs Pillow; no package was installed by this tool") from exc
        image_class = Image
    report = {"image_files": len(images), "annotation_json_files": len(annotations),
              "resolutions": {}, "unreadable_images": [], "annotation_records": 0,
              "instances": 0, "class_fields": {}, "class_values": {},
              "invalid_polygons": [], "empty_annotations": [],
              "full_decode_checked": verify_decode, "decode_failures": [],
              "orphan_annotation_filenames": [], "orphan_image_filenames": [],
              "unparseable_json": [], "limitations": [
                  "Image header/dimensions check is not full image decoding.",
                  "Filename matching is preliminary when multiple resolutions share names.",
                  "No sample grouping or final split is inferred."]}
    resolutions = Counter()
    dimensions_by_name: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for image in images:
        try:
            dimensions = image_size(image)
            dimensions_by_name[image.name].append(dimensions)
            resolutions[str(dimensions)] += 1
        except (OSError, ValueError) as exc:
            report["unreadable_images"].append({"path": str(image.relative_to(root)), "error": str(exc)})
        if image_class is not None:
            try:
                with image_class.open(image) as opened:
                    opened.load()
            except Exception as exc:
                report["decode_failures"].append({"path": str(image.relative_to(root)),
                                                  "error": str(exc)})
    image_names = Counter(p.name for p in images)
    annotation_names = Counter()
    field_counts = Counter()
    value_counts = Counter()
    for annotation in annotations:
        try:
            data = json.loads(annotation.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError, OSError) as exc:
            report["unparseable_json"].append({"path": str(annotation.relative_to(root)), "error": str(exc)})
            continue
        records = via_records(data)
        for record in records:
            filename = str(record["filename"])
            annotation_names[filename] += 1
            regions = regions_of(record)
            report["annotation_records"] += 1
            if not regions:
                report["empty_annotations"].append(filename)
            for index, region in enumerate(regions):
                report["instances"] += 1
                shape = region.get("shape_attributes", {}) if isinstance(region, dict) else {}
                xs, ys = shape.get("all_points_x"), shape.get("all_points_y")
                if not (isinstance(xs, list) and isinstance(ys, list)
                        and len(xs) == len(ys) and len(xs) >= 3
                        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                for v in xs + ys)):
                    report["invalid_polygons"].append({"filename": filename, "region": index,
                                                       "reason": "missing or malformed vertices"})
                elif polygon_area(xs, ys) <= 0:
                    report["invalid_polygons"].append({"filename": filename, "region": index,
                                                       "reason": "zero-area polygon"})
                elif any(x < 0 for x in xs) or any(y < 0 for y in ys):
                    report["invalid_polygons"].append({"filename": filename, "region": index,
                                                       "reason": "negative coordinate"})
                elif len(dimensions_by_name[filename]) == 1:
                    width, height = dimensions_by_name[filename][0]
                    if any(x >= width for x in xs) or any(y >= height for y in ys):
                        report["invalid_polygons"].append({"filename": filename, "region": index,
                                                           "reason": "coordinate outside image"})
                attrs = region.get("region_attributes", {}) if isinstance(region, dict) else {}
                if isinstance(attrs, dict):
                    for key, value in attrs.items():
                        field_counts[key] += 1
                        value_counts[f"{key}={value}"] += 1
    report["resolutions"] = dict(sorted(resolutions.items()))
    report["class_fields"] = dict(sorted(field_counts.items()))
    report["class_values"] = dict(sorted(value_counts.items()))
    report["orphan_annotation_filenames"] = sorted(set(annotation_names) - set(image_names))[:100]
    report["orphan_image_filenames"] = sorted(set(image_names) - set(annotation_names))[:100]
    report["duplicate_image_filenames"] = {name: count for name, count in image_names.items() if count > 1}
    report["ambiguous_polygon_bounds"] = sorted(name for name, sizes in dimensions_by_name.items()
                                                if len(sizes) > 1 and name in annotation_names)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Original downloaded archive; never modified")
    parser.add_argument("--root", type=Path, help="Extracted read-only tree")
    parser.add_argument("--verify-decode", action="store_true",
                        help="Fully decode each image with Pillow if available; optional slow pass")
    args = parser.parse_args()
    if not args.archive and not args.root:
        parser.error("provide --archive and/or --root")
    output = {}
    if args.archive:
        if not args.archive.is_file():
            parser.error(f"archive not found: {args.archive}")
        output["archive"] = archive_report(args.archive)
    if args.root:
        if not args.root.is_dir():
            parser.error(f"directory not found: {args.root}")
        output["tree"] = inspect_tree(args.root, args.verify_decode)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
