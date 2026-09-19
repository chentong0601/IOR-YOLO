"""Read-only D2 package and extracted-tree inspection; no splitting or repair."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import zipfile
from collections import Counter
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ior_yolo.utils.io import image_size, sha256_file

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ARCHIVE_IMAGE_SUFFIXES = IMAGE_SUFFIXES | {".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_SUFFIXES = {".json", ".xml"}
PREVIEW_LIMIT = 20


def zip_structure_report(infos: list[zipfile.ZipInfo]) -> dict:
    """Summarize central-directory names only; never read or extract a member body."""
    files = sorted(info.filename for info in infos if not info.is_dir())
    content_files = [name for name in files if not name.startswith("__MACOSX/")
                     and not name.rsplit("/", 1)[-1].startswith("._")]
    extensions = Counter(Path(name).suffix.lower() or "[no extension]" for name in files)
    images = [name for name in content_files if Path(name).suffix.lower() in ARCHIVE_IMAGE_SUFFIXES]
    annotation_candidates = [name for name in content_files
                             if Path(name).suffix.lower() in ANNOTATION_SUFFIXES
                             or (Path(name).suffix.lower() == ".txt" and
                                 re.search(r"(?:^|[/_\-.])(ann(?:otation)?s?|labels?)(?:[/_\-.]|$)",
                                           name, re.IGNORECASE))]
    top = Counter(name.split("/", 1)[0] for name in files)
    directories = set()
    for name in files:
        parts = name.split("/")
        directories.update("/".join(parts[:index]) for index in range(1, len(parts)))
    directories.update(info.filename.rstrip("/") for info in infos if info.is_dir())
    by_directory = Counter(name.rpartition("/")[0] or "[root]" for name in files)
    basename_paths: dict[str, list[str]] = defaultdict(list)
    for name in images:
        basename_paths[Path(name).name.lower()].append(name)
    repeated = {key: values for key, values in sorted(basename_paths.items()) if len(values) > 1}
    augmentation_pattern = re.compile(
        r"(?:^|[/_\-.])(?:aug(?:mented|mentation)?|brightness|hsv|gamma|noise|gaussian|"
        r"enhanced?|synthetic)(?:[/_\-.]|$)", re.IGNORECASE)
    resolution_pattern = re.compile(
        r"(?:\d{2,5}[x×]\d{2,5}|(?:^|[/_\-.])(?:resized?|resolution|small|large|"
        r"high[_-]?res|low[_-]?res)(?:[/_\-.]|$))", re.IGNORECASE)
    reference_pattern = re.compile(
        r"(?:^|[/_\-.])(?:readme|metadata|meta|classes?|license|dataset[_-]?info)"
        r"(?:[/_\-.]|$)", re.IGNORECASE)
    augmentation = sorted(name for name in content_files if augmentation_pattern.search(name))
    resolution = sorted(name for name in content_files if resolution_pattern.search(name))
    reference = sorted(name for name in content_files if reference_pattern.search(name))
    return {
        "explicit_directory_members": sum(info.is_dir() for info in infos),
        "file_members": len(files), "derived_directories": len(directories),
        "top_level_directory_names": sorted(name for name in directories if "/" not in name),
        "top_level_file_names_preview": [name for name in files if "/" not in name][:PREVIEW_LIMIT],
        "top_level_file_members": dict(sorted(top.items())),
        "directory_file_members_top_30": dict(sorted(by_directory.items(),
                                                     key=lambda item: (-item[1], item[0]))[:30]),
        "extensions": dict(sorted(extensions.items())),
        "image_file_members_by_extension": len(images),
        "image_extensions": dict(sorted(Counter(Path(name).suffix.lower() for name in images).items())),
        "annotation_candidate_file_members_by_name_or_extension": len(annotation_candidates),
        "annotation_candidate_paths_preview": annotation_candidates[:PREVIEW_LIMIT],
        "augmentation_name_hints": {"count": len(augmentation), "paths_preview": augmentation[:PREVIEW_LIMIT]},
        "resolution_name_hints": {"count": len(resolution), "paths_preview": resolution[:PREVIEW_LIMIT]},
        "repeated_image_basename_hints": {"basename_count": len(repeated),
                                          "groups_preview": dict(list(repeated.items())[:PREVIEW_LIMIT])},
        "readme_metadata_class_name_hints": {"count": len(reference),
                                             "paths_preview": reference[:PREVIEW_LIMIT]},
        "limitations": [
            "Counts are ZIP file members by extension/name, not distinct original images, VIA records, or fruit instances.",
            "No member body was read: annotation content, actual dimensions, corruption and completeness are unverified.",
            "Augmentation, resolution and repeated-basename findings are naming hints, not proven source relations.",
            "Metadata sidecars under __MACOSX/ or named ._* are excluded from image/annotation candidates."
        ]}


def archive_report(path: Path, structure_only: bool = False) -> dict:
    report = {"filename": path.name, "bytes": path.stat().st_size,
              "sha256": None if structure_only else sha256_file(path), "archive_format": "unverified",
              "members": None, "unsafe_paths": []}
    if structure_only and not zipfile.is_zipfile(path):
        raise ValueError("--structure-only supports ZIP packages only")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if structure_only:
                report["structure"] = zip_structure_report(infos)
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
    parser.add_argument("--structure-only", action="store_true",
                        help="ZIP central-directory summary only; skip SHA256 and member-body reads")
    args = parser.parse_args()
    if not args.archive and not args.root:
        parser.error("provide --archive and/or --root")
    if args.structure_only and (not args.archive or args.root or args.verify_decode):
        parser.error("--structure-only requires --archive and cannot be combined with --root/--verify-decode")
    output = {}
    if args.archive:
        if not args.archive.is_file():
            parser.error(f"archive not found: {args.archive}")
        try:
            output["archive"] = archive_report(args.archive, args.structure_only)
        except (ValueError, zipfile.BadZipFile) as exc:
            parser.error(str(exc))
    if args.root:
        if not args.root.is_dir():
            parser.error(f"directory not found: {args.root}")
        output["tree"] = inspect_tree(args.root, args.verify_decode)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
