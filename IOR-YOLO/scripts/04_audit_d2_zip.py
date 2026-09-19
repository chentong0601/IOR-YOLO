"""Read-only, low-cost D2 ZIP audit: names, VIA JSON and image headers only."""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


SPLITS = ("train", "val", "test")
GROUPS = SPLITS + tuple(s + "_resize" for s in SPLITS)
PREVIEW = 15
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DERIVATIVE_SUFFIX = re.compile(r"_(brightness|noise|gaussian|hsv|gamma)$", re.I)


def sample(values: set[str] | list[str]) -> list[str]:
    return sorted(values)[:PREVIEW]


def member_dimensions(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int]:
    """Read only an image header, without full image decode or extraction."""
    with archive.open(info) as stream:
        header = stream.read(24)
        if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) == 24:
            width = int.from_bytes(header[16:20], "big")
            height = int.from_bytes(header[20:24], "big")
            if width and height:
                return width, height
            raise ValueError("invalid PNG dimensions")
        if not header.startswith(b"\xff\xd8"):
            raise ValueError("not a supported PNG/JPEG image header")
        stream.seek(2)
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
               0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while True:
            marker = stream.read(1)
            if not marker:
                raise ValueError("JPEG dimensions not found")
            if marker != b"\xff":
                continue
            code = stream.read(1)
            while code == b"\xff":
                code = stream.read(1)
            if not code:
                raise ValueError("truncated JPEG marker")
            number = code[0]
            if number in {0xD8, 0xD9, 0x01} or 0xD0 <= number <= 0xD7:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                raise ValueError("truncated JPEG segment")
            length = int.from_bytes(length_bytes, "big")
            if length < 2:
                raise ValueError("invalid JPEG segment")
            if number in sof:
                payload = stream.read(5)
                if len(payload) != 5:
                    raise ValueError("truncated JPEG SOF")
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                if width and height:
                    return width, height
                raise ValueError("invalid JPEG dimensions")
            stream.read(length - 2)


def annotation_report(data: object, images: set[str], dimensions: dict[str, tuple[int, int]]) -> tuple[dict, dict]:
    if not isinstance(data, dict):
        raise ValueError("VIA JSON top level is not an object")
    records: dict[str, list] = {}
    duplicate_filenames: list[str] = []
    bad_records: list[str] = []
    malformed: list[str] = []
    boundary_vertices: list[str] = []
    missing_labels: list[str] = []
    empty: list[str] = []
    labels = Counter()
    shapes = Counter()
    instance_count = 0
    for key, record in data.items():
        if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
            bad_records.append(str(key))
            continue
        filename = record["filename"]
        if filename in records:
            duplicate_filenames.append(filename)
        regions = record.get("regions")
        if isinstance(regions, dict):
            regions = list(regions.values())
        if not isinstance(regions, list):
            bad_records.append(filename + " (regions not a list/dict)")
            continue
        records[filename] = regions
        if not regions:
            empty.append(filename)
        for index, region in enumerate(regions):
            instance_count += 1
            location = f"{filename}#{index}"
            if not isinstance(region, dict):
                malformed.append(location + " (region not an object)")
                continue
            attrs = region.get("region_attributes")
            label = attrs.get("name") if isinstance(attrs, dict) else None
            if not isinstance(label, str) or not label.strip():
                missing_labels.append(location)
                labels["[missing/invalid name]"] += 1
            else:
                labels[label] += 1
            shape = region.get("shape_attributes")
            shapes[shape.get("name", "[missing]") if isinstance(shape, dict) else "[invalid]"] += 1
            xs = shape.get("all_points_x") if isinstance(shape, dict) else None
            ys = shape.get("all_points_y") if isinstance(shape, dict) else None
            if not (isinstance(xs, list) and isinstance(ys, list)
                    and len(xs) == len(ys) and len(xs) >= 3
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs + ys)):
                malformed.append(location + " (invalid polygon coordinates)")
                continue
            area2 = sum(xs[i] * ys[(i + 1) % len(xs)] - xs[(i + 1) % len(xs)] * ys[i]
                        for i in range(len(xs)))
            if area2 == 0:
                malformed.append(location + " (zero-area polygon)")
            size = dimensions.get(filename)
            if size and (min(xs) < 0 or min(ys) < 0 or max(xs) > size[0] or max(ys) > size[1]):
                malformed.append(location + " (vertex more than one pixel beyond valid range)")
            elif size and (max(xs) == size[0] or max(ys) == size[1]):
                boundary_vertices.append(location)
    summary = {
        "json_top_level_records": len(data), "records_with_filename_and_regions": len(records),
        "unique_record_filenames": len(records), "total_regions_instances": instance_count,
        "class_instance_counts": dict(sorted(labels.items())), "shape_names": dict(sorted(shapes.items())),
        "empty_annotations": {"count": len(empty), "sample": sample(empty)},
        "malformed_region_findings": {"count": len(malformed), "sample": sample(malformed)},
        "boundary_at_image_width_or_height_regions": {
            "count": len(boundary_vertices), "sample": sample(boundary_vertices)},
        "missing_class_name_regions": {"count": len(missing_labels), "sample": sample(missing_labels)},
        "malformed_record_findings": {"count": len(bad_records), "sample": sample(bad_records)},
        "duplicate_record_filenames": {"count": len(duplicate_filenames), "sample": sample(duplicate_filenames)},
        "json_images_missing_in_zip": {"count": len(set(records) - images), "sample": sample(set(records) - images)},
        "zip_images_missing_in_json": {"count": len(images - set(records)), "sample": sample(images - set(records))},
    }
    return summary, records


def compare_annotations(original: dict[str, list], resized: dict[str, list],
                        original_dims: dict[str, tuple[int, int]],
                        resized_dims: dict[str, tuple[int, int]]) -> dict:
    names = set(original) & set(resized)
    result: dict = {
        "matching_record_filenames": len(names),
        "original_only_records": {"count": len(set(original) - set(resized)),
                                  "sample": sample(set(original) - set(resized))},
        "resize_only_records": {"count": len(set(resized) - set(original)),
                                "sample": sample(set(resized) - set(original))},
        "different_region_count": [], "different_class_sequence": [],
        "coordinate_comparisons": 0, "coordinate_pairs_with_residual_over_1_5px": 0,
        "coordinate_pairs_with_residual_over_2px": 0,
        "coordinate_residual_over_2px_examples": [],
        "max_coordinate_scaling_residual_px": 0.0, "noncomparable_geometry": []}
    for name in sorted(names):
        a, b = original[name], resized[name]
        if len(a) != len(b):
            result["different_region_count"].append(name)
            continue
        labels_a = [r.get("region_attributes", {}).get("name") if isinstance(r, dict) and
                    isinstance(r.get("region_attributes"), dict) else None for r in a]
        labels_b = [r.get("region_attributes", {}).get("name") if isinstance(r, dict) and
                    isinstance(r.get("region_attributes"), dict) else None for r in b]
        if labels_a != labels_b:
            result["different_class_sequence"].append(name)
        if name not in original_dims or name not in resized_dims:
            result["noncomparable_geometry"].append(name)
            continue
        xfactor = resized_dims[name][0] / original_dims[name][0]
        yfactor = resized_dims[name][1] / original_dims[name][1]
        for ra, rb in zip(a, b):
            sa = ra.get("shape_attributes", {}) if isinstance(ra, dict) else {}
            sb = rb.get("shape_attributes", {}) if isinstance(rb, dict) else {}
            if not isinstance(sa, dict) or not isinstance(sb, dict):
                result["noncomparable_geometry"].append(name)
                continue
            xa, ya = sa.get("all_points_x"), sa.get("all_points_y")
            xb, yb = sb.get("all_points_x"), sb.get("all_points_y")
            if not all(isinstance(v, list) for v in (xa, ya, xb, yb)) or not (len(xa) == len(xb)
                    and len(ya) == len(yb) and len(xa) == len(ya)):
                result["noncomparable_geometry"].append(name)
                continue
            try:
                residuals = [abs(x * xfactor - xx) for x, xx in zip(xa, xb)] + [
                    abs(y * yfactor - yy) for y, yy in zip(ya, yb)]
            except TypeError:
                result["noncomparable_geometry"].append(name)
                continue
            if residuals:
                result["coordinate_comparisons"] += 1
                highest = max(residuals)
                result["max_coordinate_scaling_residual_px"] = max(
                    result["max_coordinate_scaling_residual_px"], highest)
                if highest > 1.5:
                    result["coordinate_pairs_with_residual_over_1_5px"] += 1
                if highest > 2:
                    result["coordinate_pairs_with_residual_over_2px"] += 1
                    if len(result["coordinate_residual_over_2px_examples"]) < PREVIEW:
                        result["coordinate_residual_over_2px_examples"].append(
                            {"filename": name, "max_residual_px": round(highest, 4)})
    for key in ("different_region_count", "different_class_sequence", "noncomparable_geometry"):
        values = result[key]
        result[key] = {"count": len(values), "sample": sample(values)}
    result["max_coordinate_scaling_residual_px"] = round(result["max_coordinate_scaling_residual_px"], 4)
    return result


def source_stem(stem: str) -> str:
    return DERIVATIVE_SUFFIX.sub("", stem)


def audit_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        roots = {PurePosixPath(i.filename).parts[0] for i in infos if i.filename}
        if len(roots) != 1:
            raise ValueError(f"expected one top directory, got: {sorted(roots)}")
        root = roots.pop()
        by_split: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if (not info.is_dir() and len(parts) == 3 and parts[0] == root
                    and parts[1] in GROUPS
                    and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES):
                by_split[parts[1]].append(info)
        summaries: dict[str, dict] = {}
        filenames: dict[str, set[str]] = {}
        dimensions: dict[str, dict[str, tuple[int, int]]] = {}
        records: dict[str, dict[str, list]] = {}
        for group in GROUPS:
            group_infos = by_split[group]
            names = [PurePosixPath(info.filename).name for info in group_infos]
            filenames[group] = set(names)
            name_duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
            dims: dict[str, tuple[int, int]] = {}
            errors: list[str] = []
            for info in group_infos:
                name = PurePosixPath(info.filename).name
                try:
                    dims[name] = member_dimensions(archive, info)
                except (ValueError, OSError, EOFError, zipfile.BadZipFile) as exc:
                    errors.append(f"{name}: {exc}")
            dimensions[group] = dims
            counts = Counter(f"{w}x{h}" for w, h in dims.values())
            dimension_examples = defaultdict(list)
            for filename, (w, h) in sorted(dims.items()):
                if len(dimension_examples[f"{w}x{h}"]) < 5:
                    dimension_examples[f"{w}x{h}"].append(filename)
            json_name = f"{root}/{group}.json"
            data = json.loads(archive.read(json_name).decode("utf-8"))
            annotation, records[group] = annotation_report(data, filenames[group], dims)
            summaries[group] = {
                "image_file_members": len(group_infos), "distinct_filenames": len(filenames[group]),
                "duplicate_zip_filenames": {"count": len(name_duplicates), "sample": sample(name_duplicates)},
                "image_header_dimensions": dict(sorted(counts.items())),
                "dimension_filenames_preview": dict(sorted(dimension_examples.items())),
                "image_header_errors": {"count": len(errors), "sample": sample(errors)},
                "annotation_member": json_name, **annotation}
        cross: dict[str, dict] = {}
        family_splits: dict[str, set[str]] = defaultdict(set)
        named_derivative_counts = Counter()
        for split in SPLITS:
            for name in filenames[split]:
                stem = Path(name).stem
                family_splits[source_stem(stem)].add(split)
                if source_stem(stem) != stem:
                    named_derivative_counts[split] += 1
        for idx, a in enumerate(SPLITS):
            for b in SPLITS[idx + 1:]:
                exact = filenames[a] & filenames[b]
                stems_a = {Path(name).stem for name in filenames[a]}
                stems_b = {Path(name).stem for name in filenames[b]}
                canonical_a = defaultdict(set)
                canonical_b = defaultdict(set)
                for name in filenames[a]:
                    canonical_a[source_stem(Path(name).stem)].add(name)
                for name in filenames[b]:
                    canonical_b[source_stem(Path(name).stem)].add(name)
                shared = set(canonical_a) & set(canonical_b)
                identical_annotations = {stem for stem in shared if any(
                    records[a][left] == records[b][right]
                    for left in canonical_a[stem] for right in canonical_b[stem]
                    if left in records[a] and right in records[b])}
                cross[f"{a}_vs_{b}"] = {
                    "exact_filename_overlap": {"count": len(exact), "sample": sample(exact)},
                    "stem_overlap": {"count": len(stems_a & stems_b), "sample": sample(stems_a & stems_b)},
                    "augmentation_suffix_family_overlap_candidates": {
                        "count": len(shared), "sample": [
                            {"stem": stem, a: sample(canonical_a[stem]), b: sample(canonical_b[stem])}
                            for stem in sample(shared)]},
                    "shared_families_with_identical_region_lists": {
                        "count": len(identical_annotations), "sample": sample(identical_annotations)}}
        pairs: dict[str, dict] = {}
        for split in SPLITS:
            resized = split + "_resize"
            original_only = filenames[split] - filenames[resized]
            resize_only = filenames[resized] - filenames[split]
            pairs[split] = {
                "matching_image_filenames": len(filenames[split] & filenames[resized]),
                "original_only_images": {"count": len(original_only), "sample": sample(original_only)},
                "resize_only_images": {"count": len(resize_only), "sample": sample(resize_only)},
                "annotation_comparison": compare_annotations(records[split], records[resized],
                                                              dimensions[split], dimensions[resized])}
        return {
            "archive_filename": path.name, "archive_bytes": path.stat().st_size,
            "zip_members": len(infos), "root_directory": root, "splits": summaries,
            "original_to_resize": pairs, "original_cross_split_filename_checks": cross,
            "named_derivative_files_by_original_split": dict(sorted(named_derivative_counts.items())),
            "source_name_families_spanning_multiple_original_splits": sum(
                len(groups) > 1 for groups in family_splits.values()),
            "original_split_image_percent": {s: round(len(filenames[s]) * 100 /
                                                      sum(len(filenames[k]) for k in SPLITS), 3)
                                             for s in SPLITS},
            "limitations": [
                "Reads ZIP image headers and six JSON files; does not fully decode image pixels.",
                "Coordinate scaling is checked for aligned region/vertex order with a 1.5-pixel tolerance.",
                "Filename overlap and _brightness/_noise-style stem grouping do not exclude near duplicates,"
                " shared fruit/tree/session or other unnamed augmentation.",
                "No raw member is modified, extracted, removed or rewritten; no split is created."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"not a file: {args.archive}")
    try:
        report = audit_zip(args.archive)
    except (ValueError, KeyError, OSError, UnicodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        parser.error(f"cannot complete ZIP audit: {exc}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
