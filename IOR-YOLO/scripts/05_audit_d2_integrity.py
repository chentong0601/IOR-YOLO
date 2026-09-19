"""Read-only D2 ZIP decode and relation audit; only CSV manifests are written."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

try:
    from PIL import Image, ImageChops, ImageFile, ImageStat
except ImportError as exc:
    raise SystemExit("Pillow required; use uv run --with pillow python scripts/05_audit_d2_integrity.py ...") from exc

ImageFile.LOAD_TRUNCATED_IMAGES = False
SPLITS = ("train", "val", "test")
FAMILY_SUFFIX = re.compile(r"_(brightness|noise|gaussian|hsv|gamma)$", re.I)
FILE_FIELDS = ("split", "variant", "filename", "stem", "extension", "width", "height",
               "bytes", "sha256", "source_family", "candidate_group_id",
               "decode_status", "image_format")
PAIR_FIELDS = ("source_family", "file_a", "split_a", "variant_a", "file_b", "split_b",
               "variant_b", "relation_type", "status", "confidence", "evidence")
CLASS_FIELDS = ("split", "variant", "class_label", "instance_count")


def family_of(filename: str) -> str:
    """Strip one explicit terminal augmentation suffix; no numeric-name guessing."""
    return FAMILY_SUFFIX.sub("", Path(filename).stem).casefold()


def signature(image: Image.Image) -> bytes:
    return image.convert("L").resize((64, 64), Image.Resampling.BILINEAR).tobytes()


def correlation(a: bytes, b: bytes) -> float:
    if len(a) != len(b) or not a:
        return float("nan")
    count = len(a)
    sa, sb = sum(a), sum(b)
    numerator = sum(x * y for x, y in zip(a, b)) * count - sa * sb
    va = sum(x * x for x in a) * count - sa * sa
    vb = sum(y * y for y in b) * count - sb * sb
    if va == 0 or vb == 0:
        return 1.0 if a == b else 0.0
    return numerator / math.sqrt(va * vb)


def difference_hash(image_signature: bytes) -> int:
    """256-bit directional hash sampled from the already decoded 64x64 luminance."""
    value = 0
    for y in range(16):
        for x in range(16):
            value = (value << 1) | (
                image_signature[(y * 4) * 64 + x * 4]
                > image_signature[(y * 4) * 64 + (x + 1) * 4])
    return value


def decode(payload: bytes, extension: str) -> tuple[Image.Image | None, str, str]:
    """Strictly verify and load every pixel; caller already verified ZIP CRC on read."""
    try:
        with Image.open(io.BytesIO(payload)) as first:
            actual_format = first.format or "[unknown]"
            first.verify()
        with Image.open(io.BytesIO(payload)) as second:
            second.load()
            image = second.convert("RGB")
    except (OSError, SyntaxError, ValueError, EOFError) as exc:
        error = str(exc)
        status = "truncated" if "truncated" in error.lower() or "broken data stream" in error.lower() else "unreadable"
        return None, status, error
    expected = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}.get(extension)
    return image, "decoded" if actual_format == expected else "unexpected_format", actual_format


def resize_metrics(original: Image.Image, resized: Image.Image) -> tuple[float, float]:
    estimated = original.resize(resized.size, Image.Resampling.LANCZOS)
    difference = ImageChops.difference(estimated, resized)
    mae = sum(ImageStat.Stat(difference).mean) / 3
    return round(mae, 4), round(correlation(signature(original), signature(resized)), 6)


def csv_write(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def classify_cross_split(left: dict, right: dict, annotations: dict[str, dict[str, list]],
                         signatures: dict[tuple[str, str], bytes]) -> tuple[str, str, str, str]:
    if left["sha256"] and left["sha256"] == right["sha256"]:
        return "exact_duplicate", "Confirmed", "High", "identical uncompressed SHA256"
    key_left = (left["split"], left["filename"])
    key_right = (right["split"], right["filename"])
    if key_left not in signatures or key_right not in signatures:
        return "unresolved", "Unresolved", "Unknown", "one or both images did not fully decode"
    corr = correlation(signatures[key_left], signatures[key_right])
    same_geometry = annotations[left["split"]].get(left["filename"]) == annotations[right["split"]].get(right["filename"])
    evidence = f"64x64 luminance correlation={corr:.6f}; identical region list={same_geometry}"
    if corr >= 0.995:
        return "likely_augmented_derivative", "Supported", "High" if same_geometry else "Moderate", evidence
    if same_geometry:
        return "same_annotation_geometry", "Candidate", "Moderate", evidence
    return "naming_only_candidate", "Candidate", "Low", evidence


def audit(archive_path: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    files: list[dict] = []
    pairs: list[dict] = []
    classes: list[dict] = []
    annotations: dict[str, dict[str, list]] = {}
    signatures: dict[tuple[str, str], bytes] = {}
    resize_comparisons: dict[str, list[dict]] = defaultdict(list)
    decode_errors: list[dict] = []
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        roots = {PurePosixPath(info.filename).parts[0] for info in infos if info.filename}
        if len(roots) != 1:
            raise ValueError("expected one ZIP root directory")
        root = roots.pop()
        images: dict[tuple[str, str], zipfile.ZipInfo] = {}
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if (not info.is_dir() and len(parts) == 3 and parts[0] == root
                    and parts[1] in (*SPLITS, *(s + "_resize" for s in SPLITS))
                    and PurePosixPath(info.filename).suffix.lower() in {".jpg", ".jpeg", ".png"}):
                images[(parts[1], parts[2])] = info
        for split in SPLITS:
            for variant in ("original", "resize"):
                folder = split if variant == "original" else split + "_resize"
                data = json.loads(archive.read(f"{root}/{folder}.json"))
                if not isinstance(data, dict):
                    raise ValueError(f"{folder}.json is not an object")
                counts = Counter()
                annotations[folder] = {}
                for record in data.values():
                    if not isinstance(record, dict) or not isinstance(record.get("filename"), str):
                        raise ValueError(f"malformed record in {folder}.json")
                    regions = record.get("regions")
                    if isinstance(regions, dict):
                        regions = list(regions.values())
                    if not isinstance(regions, list):
                        raise ValueError(f"malformed regions in {folder}.json")
                    annotations[folder][record["filename"]] = regions
                    for region in regions:
                        attrs = region.get("region_attributes", {}) if isinstance(region, dict) else {}
                        label = attrs.get("name") if isinstance(attrs, dict) else None
                        counts[label if isinstance(label, str) and label.strip() else "Unlabeled / Unknown"] += 1
                classes.extend({"split": split, "variant": variant, "class_label": label,
                                "instance_count": count} for label, count in sorted(counts.items()))
        # CRC validation, SHA256 and full pixel decode for every ZIP image.
        for folder, filename in sorted(images):
            info = images[(folder, filename)]
            split = folder.removesuffix("_resize")
            variant = "resize" if folder.endswith("_resize") else "original"
            extension = Path(filename).suffix.lower()
            try:
                payload = archive.read(info)
            except (OSError, EOFError, zipfile.BadZipFile) as exc:
                files.append({"split": split, "variant": variant, "filename": filename,
                              "stem": Path(filename).stem, "extension": extension, "width": "", "height": "",
                              "bytes": info.file_size, "sha256": "", "source_family": family_of(filename),
                              "decode_status": "corrupt_zip_member", "image_format": ""})
                decode_errors.append({"split": split, "variant": variant, "filename": filename,
                                      "status": "corrupt_zip_member", "error": str(exc)})
                continue
            digest = hashlib.sha256(payload).hexdigest()
            image, status, detail = decode(payload, extension)
            if status != "decoded":
                decode_errors.append({"split": split, "variant": variant, "filename": filename,
                                      "status": status, "error": detail})
            if image is not None:
                signatures[(folder, filename)] = signature(image)
            width, height = image.size if image is not None else ("", "")
            files.append({"split": split, "variant": variant, "filename": filename,
                          "stem": Path(filename).stem, "extension": extension,
                          "width": width, "height": height, "bytes": len(payload), "sha256": digest,
                          "source_family": family_of(filename), "decode_status": status,
                          "image_format": detail if status in {"decoded", "unexpected_format"} else ""})
            if variant == "resize" and image is not None and (split, filename) in images:
                raw = archive.read(images[(split, filename)])
                source, source_status, _ = decode(raw, extension)
                if source is not None and source_status in {"decoded", "unexpected_format"}:
                    mae, corr = resize_metrics(source, image)
                    resize_comparisons[split].append({"filename": filename, "mae": mae, "correlation": corr})
        by_key = {(row["split"], row["variant"], row["filename"]): row for row in files}
        metric_by_key = {(split, metric["filename"]): metric
                         for split, values in resize_comparisons.items() for metric in values}
        for split in SPLITS:
            original_names = {name for folder, name in images if folder == split}
            resized_names = {name for folder, name in images if folder == split + "_resize"}
            for filename in sorted(original_names & resized_names):
                left = by_key[(split, "original", filename)]
                right = by_key[(split, "resize", filename)]
                metric = metric_by_key.get((split, filename))
                if left["sha256"] and left["sha256"] == right["sha256"]:
                    relation, status, confidence = "exact_duplicate", "Confirmed", "High"
                    evidence = "identical uncompressed SHA256"
                elif metric is None:
                    relation, status, confidence = "unresolved", "Unresolved", "Unknown"
                    evidence = "pixel comparison unavailable"
                else:
                    relation = "resized_duplicate"
                    status = "Supported" if metric["correlation"] >= 0.995 and metric["mae"] <= 20 else "Unresolved"
                    confidence = "High" if status == "Supported" else "Unknown"
                    evidence = (f"same filename; {left['width']}x{left['height']}->"
                                f"{right['width']}x{right['height']}; LANCZOS RGB MAE={metric['mae']}; "
                                f"64x64 luminance correlation={metric['correlation']}")
                pairs.append({"source_family": family_of(filename), "file_a": filename,
                              "split_a": split, "variant_a": "original", "file_b": filename,
                              "split_b": split, "variant_b": "resize", "relation_type": relation,
                              "status": status, "confidence": confidence, "evidence": evidence})
        families: dict[str, list[dict]] = defaultdict(list)
        for row in files:
            if row["variant"] == "original":
                families[row["source_family"]].append(row)
        ranking = {"Confirmed": 3, "Supported": 2, "Candidate": 1, "Unresolved": 0}
        family_statuses = Counter()
        family_details = {}
        cross_pair_keys = set()
        for family, members in sorted(families.items()):
            if len({member["split"] for member in members}) < 2:
                continue
            statuses = []
            for index, left in enumerate(members):
                for right in members[index + 1:]:
                    if left["split"] == right["split"]:
                        continue
                    relation, status, confidence, evidence = classify_cross_split(
                        left, right, annotations, signatures)
                    statuses.append(status)
                    cross_pair_keys.add(frozenset(((left["split"], left["filename"]),
                                                  (right["split"], right["filename"]))))
                    pairs.append({"source_family": family, "file_a": left["filename"],
                                  "split_a": left["split"], "variant_a": "original",
                                  "file_b": right["filename"], "split_b": right["split"],
                                  "variant_b": "original", "relation_type": relation,
                                  "status": status, "confidence": confidence, "evidence": evidence})
            family_status = max(statuses, key=lambda value: ranking[value])
            family_statuses[family_status] += 1
            family_details[family] = {"status": family_status,
                                      "splits": sorted({row["split"] for row in members}),
                                      "files": sorted(f"{row['split']}/{row['filename']}" for row in members)}
        # Find cross-split exact SHA matches even where filenames share no known suffix family.
        by_hash: dict[str, list[dict]] = defaultdict(list)
        for row in files:
            if row["variant"] == "original" and row["sha256"]:
                by_hash[row["sha256"]].append(row)
        exact_cross_split_groups = []
        exact_within_split_groups = []
        for digest, members in by_hash.items():
            if len(members) < 2:
                continue
            group_record = {"sha256": digest,
                            "files": sorted(f"{row['split']}/{row['filename']}" for row in members)}
            if len({row["split"] for row in members}) < 2:
                exact_within_split_groups.append(group_record)
            else:
                exact_cross_split_groups.append(group_record)
            for index, left in enumerate(members):
                for right in members[index + 1:]:
                    key = frozenset(((left["split"], left["filename"]),
                                     (right["split"], right["filename"])))
                    if key not in cross_pair_keys:
                        same_annotation = (annotations[left["split"]].get(left["filename"])
                                           == annotations[right["split"]].get(right["filename"]))
                        pairs.append({"source_family": f"sha256:{digest[:16]}",
                                      "file_a": left["filename"], "split_a": left["split"],
                                      "variant_a": "original", "file_b": right["filename"],
                                      "split_b": right["split"], "variant_b": "original",
                                      "relation_type": "exact_duplicate", "status": "Confirmed",
                                      "confidence": "High",
                                      "evidence": f"identical SHA256 {digest}; identical region list={same_annotation}"})
        # Different filenames are demonstrably sometimes byte-identical here, so
        # use a light directional hash to find additional *candidates*. A high
        # luminance correlation supports, but does not prove, derivation.
        originals = {split: [row for row in files if row["variant"] == "original" and row["split"] == split
                             and (split, row["filename"]) in signatures] for split in SPLITS}
        hashes = {(split, row["filename"]): difference_hash(signatures[(split, row["filename"])])
                  for split in SPLITS for row in originals[split]}
        perceptual_rows = []
        for index, split_a in enumerate(SPLITS):
            for split_b in SPLITS[index + 1:]:
                for left in originals[split_a]:
                    for right in originals[split_b]:
                        if left["source_family"] == right["source_family"]:
                            continue
                        key_a, key_b = (split_a, left["filename"]), (split_b, right["filename"])
                        distance = (hashes[key_a] ^ hashes[key_b]).bit_count()
                        if distance > 12 or left["sha256"] == right["sha256"]:
                            continue
                        corr = correlation(signatures[key_a], signatures[key_b])
                        same_geometry = (annotations[split_a].get(left["filename"])
                                         == annotations[split_b].get(right["filename"]))
                        status = "Supported" if corr >= 0.995 else "Candidate"
                        relation = "likely_augmented_derivative" if status == "Supported" else "unresolved"
                        confidence = ("High" if same_geometry else "Moderate") if status == "Supported" else "Low"
                        evidence = (f"256-bit dHash distance={distance}; 64x64 luminance correlation={corr:.6f}; "
                                    f"identical region list={same_geometry}; different source names")
                        row = {"source_family": f"visual:{left['source_family']}|{right['source_family']}",
                               "file_a": left["filename"], "split_a": split_a, "variant_a": "original",
                               "file_b": right["filename"], "split_b": split_b, "variant_b": "original",
                               "relation_type": relation, "status": status, "confidence": confidence,
                               "evidence": evidence}
                        perceptual_rows.append(row)
                        pairs.append(row)
        # Candidate source graph: naming-family links plus byte-identical and
        # high-confidence visual edges. No split assignment is made here.
        original_nodes = {(row["split"], row["filename"]) for row in files if row["variant"] == "original"}
        parent = {node: node for node in original_nodes}

        def find(node: tuple[str, str]) -> tuple[str, str]:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: tuple[str, str], b: tuple[str, str]) -> None:
            a, b = find(a), find(b)
            if a != b:
                parent[b] = a

        for members in families.values():
            nodes = [(row["split"], row["filename"]) for row in members]
            for node in nodes[1:]:
                union(nodes[0], node)
        for row in pairs:
            if (row["variant_a"] == row["variant_b"] == "original"
                    and row["status"] in {"Confirmed", "Supported"}):
                union((row["split_a"], row["file_a"]), (row["split_b"], row["file_b"]))
        components: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        for node in sorted(original_nodes):
            components[find(node)].append(node)
        ordered_components = sorted(components.values(), key=lambda members: sorted(members)[0])
        group_ids = {node: f"candidate-{index:04d}"
                     for index, members in enumerate(ordered_components, start=1) for node in members}
        for row in files:
            row["candidate_group_id"] = group_ids[(row["split"], row["filename"])]
        cross_components = [members for members in ordered_components
                            if len({split for split, _ in members}) > 1]
        summary = {
            "archive": archive_path.name, "zip_members": len(infos), "pillow_version": Image.__version__,
            "image_files": len(files),
            "decode_status_counts": dict(sorted(Counter(row["decode_status"] for row in files).items())),
            "decode_errors": decode_errors,
            "resize_pixel_comparison": {
                split: {"paired": len(resize_comparisons[split]),
                        "mae_over_20": [row for row in resize_comparisons[split] if row["mae"] > 20],
                        "correlation_below_0_995": [row for row in resize_comparisons[split]
                                                    if row["correlation"] < 0.995],
                        "min_correlation": min((row["correlation"] for row in resize_comparisons[split]), default=None),
                        "max_mae": max((row["mae"] for row in resize_comparisons[split]), default=None)}
                for split in SPLITS},
            "cross_split_named_families": len(family_details),
            "family_status_counts": {status: family_statuses[status] for status in ranking},
            "family_details": family_details,
            "exact_hash_groups_cross_split": exact_cross_split_groups,
            "exact_hash_groups_within_split": exact_within_split_groups,
            "additional_perceptual_cross_split_candidates": {
                "pairs": len(perceptual_rows),
                "status_counts": dict(Counter(row["status"] for row in perceptual_rows)),
                "different_source_name_pairs": [
                    {"a": f"{row['split_a']}/{row['file_a']}",
                     "b": f"{row['split_b']}/{row['file_b']}", "evidence": row["evidence"]}
                    for row in perceptual_rows if row["status"] == "Supported"]},
            "candidate_source_graph": {
                "components": len(ordered_components),
                "cross_split_components": len(cross_components),
                "original_images_in_cross_split_components": sum(len(members) for members in cross_components),
                "largest_component_images": max((len(members) for members in ordered_components), default=0),
                "rule": "union one terminal-suffix family, identical original SHA256, and Supported cross-split visual edges; Candidate/Unresolved visual edges are not unioned"},
            "limitations": [
                "Pillow full decode validates readability, not label semantics.",
                "Similarity is screening evidence, not proof of a deterministic transform.",
                "Source families use one terminal augmentation suffix only; unnamed near duplicates and fruit/tree/session links remain unverified.",
                "No raw file or train/val/test split is changed."]}
    return summary, files, pairs, classes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.archive.is_file() or not args.manifest_dir.is_dir():
        parser.error("archive and existing manifest directory are required")
    summary, files, pairs, classes = audit(args.archive)
    csv_write(args.manifest_dir / "files_sha256.csv", FILE_FIELDS, files)
    csv_write(args.manifest_dir / "duplicate_report.csv", PAIR_FIELDS, pairs)
    csv_write(args.manifest_dir / "class_counts.csv", CLASS_FIELDS, classes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
