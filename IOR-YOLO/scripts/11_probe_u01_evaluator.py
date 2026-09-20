"""Small, model-free U01 evaluator probe; never edits the original annotation."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

CLASSES = {"immature apple", "semi-mature apple", "mature apple"}


def bounds(region: dict) -> tuple[float, float, float, float]:
    shape = region["shape_attributes"]
    xs, ys = shape["all_points_x"], shape["all_points_y"]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union else 0.0


def evaluate(targets: list[tuple[str, tuple]], predictions: list[tuple[str, tuple]],
             ignore_boxes: list[tuple] | None = None, threshold: float = 0.5) -> dict[str, int]:
    """Single-image, class-aware greedy matching for a synthetic policy probe."""
    matched = set()
    true_positive = false_positive = ignored = 0
    for label, box in predictions:
        eligible = [(iou(box, target_box), index) for index, (target_label, target_box)
                    in enumerate(targets) if index not in matched and target_label == label]
        score, match = max(eligible, default=(0.0, None))
        if match is not None and score >= threshold:
            matched.add(match)
            true_positive += 1
        elif any(iou(box, region) >= threshold for region in (ignore_boxes or [])):
            ignored += 1
        else:
            false_positive += 1
    return {"tp": true_positive, "fp": false_positive,
            "fn": len(targets) - true_positive, "ignored_predictions": ignored}


def probe(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        roots = {name.split("/")[0] for name in archive.namelist() if name}
        if len(roots) != 1:
            raise ValueError("expected one ZIP root")
        records = json.loads(archive.read(f"{roots.pop()}/test.json"))
        matches = [record for record in records.values() if record["filename"] == "IMG_54350.jpg"]
        if len(matches) != 1:
            raise ValueError("U01 image not found exactly once")
        regions = matches[0]["regions"]
        regions = list(regions.values()) if isinstance(regions, dict) else regions
    if len(regions) != 4 or regions[3].get("region_attributes") != {}:
        raise ValueError("U01 raw annotation differs from audited record")
    targets = []
    for region in regions[:3]:
        label = region.get("region_attributes", {}).get("name")
        if label not in CLASSES:
            raise ValueError("one of the three retained U01 regions lacks a valid class")
        targets.append((label, bounds(region)))
    unknown_box = bounds(regions[3])
    predictions = [*targets, ("immature apple", unknown_box)]
    standard = evaluate(targets, predictions)
    sensitivity = evaluate(targets, predictions, ignore_boxes=[unknown_box])
    if standard != {"tp": 3, "fp": 1, "fn": 0, "ignored_predictions": 0}:
        raise AssertionError(f"unexpected standard evaluator probe: {standard}")
    if sensitivity != {"tp": 3, "fp": 0, "fn": 0, "ignored_predictions": 1}:
        raise AssertionError(f"unexpected ignore-zone probe: {sensitivity}")
    return {"status": "synthetic box evaluator policy probe; actual model backend not tested",
            "retained_valid_targets": len(targets), "invalid_unknown_regions": 1,
            "max_unknown_box_iou_with_valid_target": round(max(iou(unknown_box, box) for _, box in targets), 6),
            "prediction_boxes": len(predictions), "standard_no_ignore": standard,
            "optional_ignore_zone_sensitivity": sensitivity,
            "candidate_interpretation": "Do not assign a maturity class; because human review did not identify an obvious apple, standard FP counting is defensible. Optional ignore is sensitivity analysis only, not automatically selected."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(probe(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
