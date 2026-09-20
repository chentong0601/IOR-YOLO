"""Dry-run D2 group-aware splits; writes no split assignments or raw data."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ("immature apple", "semi-mature apple", "mature apple")
SPLITS = ("train", "val", "test")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def group_data(source_rows: list[dict], annotations: dict[tuple[str, str], list]) -> dict[str, dict]:
    groups = defaultdict(lambda: {"images": 0, "multi_class_images": 0,
                                  "classes": Counter(), "members": []})
    for row in source_rows:
        if row["variant"] != "original":
            continue
        node = row["official_split"], row["filename"]
        if node not in annotations:
            raise ValueError(f"missing annotation for {node}")
        labels = [region.get("region_attributes", {}).get("name")
                  for region in annotations[node]]
        valid = [label for label in labels if label in CLASSES]
        if len(valid) != int(row["instance_count"]):
            raise ValueError(f"class count mismatch for {node}")
        group = groups[row["source_group_id"]]
        group["images"] += 1
        group["multi_class_images"] += len(set(valid)) > 1
        group["classes"].update(valid)
        group["members"].append(node)
    return dict(groups)


def totals(groups: dict[str, dict]) -> dict[str, int]:
    result = {"images": 0, "groups": len(groups), "multi_class_images": 0,
              **{label: 0 for label in CLASSES}}
    for group in groups.values():
        result["images"] += group["images"]
        result["multi_class_images"] += group["multi_class_images"]
        for label in CLASSES:
            result[label] += group["classes"][label]
    return result


def score(counts: dict[str, dict], target: tuple[float, float, float],
          total: dict[str, int]) -> float:
    """L1 ratio error; image 1, each class 1/3, multiclass 1/2, group 1/4.

    Confirmed leakage is a hard validity condition checked independently.
    """
    image_error = sum(abs(counts[s]["images"] / total["images"] - target[i])
                      for i, s in enumerate(SPLITS))
    class_error = sum(abs(counts[s][label] / total[label] - target[i])
                      for i, s in enumerate(SPLITS) for label in CLASSES) / 3
    group_error = sum(abs(counts[s]["groups"] / total["groups"] - target[i])
                      for i, s in enumerate(SPLITS))
    multiclass_error = sum(abs(counts[s]["multi_class_images"] / total["multi_class_images"] - target[i])
                           for i, s in enumerate(SPLITS)) if total["multi_class_images"] else 0
    return image_error + class_error + 0.5 * multiclass_error + 0.25 * group_error


def simulate(groups: dict[str, dict], target: tuple[float, float, float], seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    total = totals(groups)
    counts = {split: {field: 0 for field in total} for split in SPLITS}
    assignments = {}
    # A seed must identify the same simulation even if manifest rows are reordered.
    order = sorted(groups)
    rng.shuffle(order)
    # Large groups first; shuffled ties make seeds meaningful while remaining deterministic.
    order.sort(key=lambda gid: -groups[gid]["images"])
    for gid in order:
        group = groups[gid]
        candidate_scores = []
        for split in SPLITS:
            counts[split]["images"] += group["images"]
            counts[split]["groups"] += 1
            counts[split]["multi_class_images"] += group["multi_class_images"]
            for label in CLASSES:
                counts[split][label] += group["classes"][label]
            candidate_scores.append(score(counts, target, total))
            counts[split]["images"] -= group["images"]
            counts[split]["groups"] -= 1
            counts[split]["multi_class_images"] -= group["multi_class_images"]
            for label in CLASSES:
                counts[split][label] -= group["classes"][label]
        best = min(candidate_scores)
        tied = [i for i, value in enumerate(candidate_scores) if abs(value - best) < 1e-12]
        chosen = SPLITS[rng.choice(tied)]
        assignments[gid] = chosen
        counts[chosen]["images"] += group["images"]
        counts[chosen]["groups"] += 1
        counts[chosen]["multi_class_images"] += group["multi_class_images"]
        for label in CLASSES:
            counts[chosen][label] += group["classes"][label]
    return assignments, counts


def edge_cuts(edges: list[dict], node_to_group: dict, assignments: dict) -> dict[str, int]:
    cuts = Counter()
    for edge in edges:
        if edge["variant_a"] != edge["variant_b"] or edge["variant_a"] != "original":
            continue
        a = node_to_group[(edge["split_a"], edge["file_a"])]
        b = node_to_group[(edge["split_b"], edge["file_b"])]
        if assignments[a] != assignments[b]:
            cuts[edge["confidence"]] += 1
    return dict(cuts)


def audit_one(assignments: dict, groups: dict, counts: dict, edges: list[dict],
              node_to_group: dict, target: tuple[float, float, float]) -> dict:
    total = totals(groups)
    if len(assignments) != len(groups):
        raise AssertionError("a source group was not assigned")
    if sum(counts[s]["images"] for s in SPLITS) != total["images"]:
        raise AssertionError("image count not conserved")
    if any(sum(counts[s][label] for s in SPLITS) != total[label] for label in CLASSES):
        raise AssertionError("class count not conserved")
    cuts = edge_cuts(edges, node_to_group, assignments)
    if any(cuts.get(level, 0) for level in ("Confirmed", "Strongly Supported", "Supported")):
        raise AssertionError(f"evidence-supported group edge crosses split: {cuts}")
    return {"score": round(score(counts, target, total), 6),
            "ratio_deviation_percentage_points": {s: round(100 * (counts[s]["images"] / total["images"] - target[i]), 3)
                                                  for i, s in enumerate(SPLITS)},
            "counts": counts, "edge_cuts": cuts}


def run(archive_path: Path, manifest_dir: Path, seeds: int) -> dict:
    if not 1 <= seeds <= 500:
        raise ValueError("seeds must be between 1 and 500")
    source_rows = read_csv(manifest_dir / "source_groups.csv")
    edges = read_csv(manifest_dir / "source_group_relations.csv")
    with zipfile.ZipFile(archive_path) as archive:
        roots = {name.split("/")[0] for name in archive.namelist() if name}
        if len(roots) != 1:
            raise ValueError("expected one ZIP root")
        root = roots.pop()
        annotations = {}
        for split in SPLITS:
            data = json.loads(archive.read(f"{root}/{split}.json"))
            for record in data.values():
                regions = record["regions"]
                annotations[(split, record["filename"])] = list(regions.values()) if isinstance(regions, dict) else regions
    groups = group_data(source_rows, annotations)
    node_to_group = {(row["official_split"], row["filename"]): row["source_group_id"]
                     for row in source_rows if row["variant"] == "original"}
    if len(node_to_group) != 1406 or totals(groups)["images"] != 1406:
        raise ValueError("expected 1406 unique original images")
    scenarios = {}
    for label, target in (("70/15/15", (0.7, 0.15, 0.15)),
                          ("70/10/20", (0.7, 0.1, 0.2))):
        trials = []
        for seed in range(seeds):
            assignments, counts = simulate(groups, target, seed)
            result = audit_one(assignments, groups, counts, edges, node_to_group, target)
            trials.append({"seed": seed, **result})
        representative = min(trials, key=lambda trial: (trial["score"], trial["seed"]))
        scenarios[label] = {"seeds_tested": seeds, "score_median": round(statistics.median(t["score"] for t in trials), 6),
                            "score_range": [min(t["score"] for t in trials), max(t["score"] for t in trials)],
                            "representative": representative}
    return {"status": "Simulation Only / Not Frozen", "source_groups": len(groups),
            "original_images": totals(groups)["images"], "scenarios": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(run(args.archive, args.manifest_dir, args.seeds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
