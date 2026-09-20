"""Stage 2B-5P: build precautionary split guards and simulate aggregate splits.

Reads the fixed D2 ZIP and existing manifests. Writes only two *candidate* CSVs.
No sample-to-split assignment is saved, and no final ratio or seed is selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("immature apple", "semi-mature apple", "mature apple")
SPLITS = ("train", "val", "test")
SUFFIX = re.compile(r"_(brightness|noise|gaussian|hsv|gamma)$", re.I)
GUARD_FIELDS = ("official_split", "filename", "source_group_id", "split_guard_cluster_id",
                "guard_reason", "highest_risk_relation", "confidence", "notes")
POOL_FIELDS = ("official_split", "filename", "variant", "source_group_id",
               "split_guard_cluster_id", "representation_type", "augmentation_type",
               "include_candidate", "exclusion_reason", "annotation_policy", "notes")
SEED_CONTEXT = b"IOR-YOLO/D2/Stage2B-5P/guard-greedy-v1"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest_file(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def seed_from_archive_sha(archive_sha: str) -> int:
    return int.from_bytes(hashlib.sha256(bytes.fromhex(archive_sha) + SEED_CONTEXT).digest()[:8], "big")


class Components:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def original_ref(row: dict) -> tuple[str, str]:
    return row["official_split"], row["filename"]


def relation_refs(edge: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    return (edge["split_a"], edge["file_a"]), (edge["split_b"], edge["file_b"])


def candidate_guard_edges(relations: list[dict], groups_by_ref: dict) -> list[dict]:
    """Guard every cross-group original Candidate, without promoting its evidence."""
    result = []
    for edge in relations:
        if edge["variant_a"] != edge["variant_b"] or edge["variant_a"] != "original":
            continue
        if edge["confidence"] != "Candidate":
            continue
        a, b = relation_refs(edge)
        if groups_by_ref[a] == groups_by_ref[b]:
            continue
        result.append(edge)
    return result


def build_guards(originals: list[dict], relations: list[dict]) -> tuple[list[dict], dict, dict]:
    groups_by_ref = {original_ref(row): row["source_group_id"] for row in originals}
    source_groups = sorted(set(groups_by_ref.values()))
    if len(source_groups) != 1112 or len(originals) != 1406:
        raise ValueError("expected adjudicated D2 original manifests (1112 groups / 1406 images)")
    risk_edges = candidate_guard_edges(relations, groups_by_ref)
    dhash_risks = sum("dHash" in edge["evidence"] for edge in risk_edges)
    if dhash_risks != 23 or len(risk_edges) != 29:
        raise ValueError(f"expected 23 dHash + 6 other cross-group Candidate edges, found {dhash_risks} / {len(risk_edges)}")
    components = Components(source_groups)
    for edge in risk_edges:
        a, b = relation_refs(edge)
        components.union(groups_by_ref[a], groups_by_ref[b])
    grouped = defaultdict(list)
    for group_id in source_groups:
        grouped[components.find(group_id)].append(group_id)
    ordered = sorted(grouped.values(), key=lambda members: members[0])
    cluster_for_group = {group_id: f"gc-{index:04d}"
                         for index, members in enumerate(ordered, 1) for group_id in members}
    risk_by_cluster = defaultdict(list)
    for edge in risk_edges:
        a, _ = relation_refs(edge)
        risk_by_cluster[cluster_for_group[groups_by_ref[a]]].append(edge)

    def strongest(edge: dict) -> tuple[int, float]:
        match_hash = re.search(r"dHash distance=(\d+)", edge["evidence"])
        match_corr = re.search(r"64x64 luminance correlation=([0-9.]+)", edge["evidence"])
        return (int(match_hash.group(1)) if match_hash else 999,
                -float(match_corr.group(1)) if match_corr else 0.0)

    rows = []
    for item in sorted(originals, key=original_ref):
        group_id = item["source_group_id"]
        cluster_id = cluster_for_group[group_id]
        risks = risk_by_cluster[cluster_id]
        if risks:
            representative = min(risks, key=strongest)
            a, b = relation_refs(representative)
            lead = f"{a[0]}/{a[1]} <-> {b[0]}/{b[1]}"
            reason = "residual_candidate_transitive_guard"
            confidence = "Candidate (guard only)"
        else:
            lead, reason, confidence = "none", "source_group_only", "N/A"
        rows.append({"official_split": item["official_split"], "filename": item["filename"],
                     "source_group_id": group_id, "split_guard_cluster_id": cluster_id,
                     "guard_reason": reason, "highest_risk_relation": lead,
                     "confidence": confidence,
                     "notes": "Precautionary split constraint; not proof of common physical acquisition"
                              if risks else "One evidence-based source group; physical acquisition ID unverified"})
    cluster_groups = defaultdict(set)
    cluster_images = Counter()
    for row in rows:
        cluster_groups[row["split_guard_cluster_id"]].add(row["source_group_id"])
        cluster_images[row["split_guard_cluster_id"]] += 1
    summary = {
        "source_groups": len(source_groups), "split_guard_clusters": len(ordered),
        "guarded_candidate_edges": len(risk_edges),
        "guarded_dhash_candidate_edges": dhash_risks,
        "guarded_other_candidate_edges": len(risk_edges) - dhash_risks,
        "unguarded_residual_risk_edges": sum(
            cluster_for_group[groups_by_ref[relation_refs(edge)[0]]] !=
            cluster_for_group[groups_by_ref[relation_refs(edge)[1]]] for edge in risk_edges),
        "largest_cluster_source_groups": max(map(len, cluster_groups.values())),
        "largest_cluster_original_images": max(cluster_images.values()),
        "clusters_with_residual_candidate_edges": sum(bool(edges) for edges in risk_by_cluster.values()),
    }
    return rows, cluster_for_group, summary


def read_original_annotations(archive: zipfile.ZipFile) -> dict[tuple[str, str], list[dict]]:
    roots = {name.split("/")[0] for name in archive.namelist() if name}
    if len(roots) != 1:
        raise ValueError("expected one ZIP root")
    root = roots.pop()
    records = {}
    for split in SPLITS:
        data = json.loads(archive.read(f"{root}/{split}.json"))
        for record in data.values():
            regions = record["regions"]
            records[(split, record["filename"])] = (list(regions.values())
                                                     if isinstance(regions, dict) else regions)
    return records


def conflict_groups(originals: list[dict], files: list[dict], annotations: dict) -> tuple[set[str], dict]:
    original_by_ref = {original_ref(row): row for row in originals}
    sha_families = defaultdict(list)
    for row in files:
        if row["variant"] == "original":
            sha_families[row["sha256"]].append((row["split"], row["filename"]))
    conflicts = []
    groups = set()
    for sha, members in sorted(sha_families.items()):
        if len(members) < 2:
            continue
        signatures = {json.dumps(annotations[ref], sort_keys=True, ensure_ascii=False)
                      for ref in members}
        if len(signatures) > 1:
            member_groups = {original_by_ref[ref]["source_group_id"] for ref in members}
            if len(member_groups) != 1:
                raise AssertionError("byte-identical annotation conflict crosses source groups")
            groups.update(member_groups)
            conflicts.append({"sha256": sha, "members": [f"{a}/{b}" for a, b in sorted(members)],
                              "source_group_id": next(iter(member_groups))})
    cross_split_conflicts = sum(len({ref.split("/", 1)[0] for ref in item["members"]}) > 1
                                for item in conflicts)
    if cross_split_conflicts != 3:
        raise ValueError(f"expected three audited cross-split conflicts, found {cross_split_conflicts}")
    return groups, {"sha_conflict_pairs": len(conflicts),
                    "cross_split_conflict_pairs": cross_split_conflicts,
                    "same_split_conflict_pairs": len(conflicts) - cross_split_conflicts,
                    "conflict_source_groups": sorted(groups),
                    "conflict_sha_groups": conflicts}


def augmentation_type(filename: str) -> str:
    match = SUFFIX.search(Path(filename).stem)
    return match.group(1).lower() if match else "none"


def build_pool(source_rows: list[dict], files: list[dict], cluster_for_group: dict,
               annotations: dict) -> tuple[list[dict], dict]:
    originals = [row for row in source_rows if row["variant"] == "original"]
    conflict, conflict_info = conflict_groups(originals, files, annotations)
    file_by_ref = {(row["split"], row["filename"], row["variant"]): row for row in files}
    base_by_group = defaultdict(list)
    for row in originals:
        if augmentation_type(row["filename"]) == "none":
            base_by_group[row["source_group_id"]].append(row)
    no_base = sorted(set(row["source_group_id"] for row in originals) - set(base_by_group))
    # A no-base group is listed with a deterministic fallback, but held out until reviewed.
    fallback = {group_id: min((row for row in originals if row["source_group_id"] == group_id),
                              key=original_ref) for group_id in no_base}
    chosen_by_sha = {}
    for row in sorted((row for members in base_by_group.values() for row in members),
                      key=original_ref):
        ref = (*original_ref(row), "original")
        chosen_by_sha.setdefault(file_by_ref[ref]["sha256"], original_ref(row))
    canonical_by_group = {group_id: original_ref(min(members, key=original_ref))
                          for group_id, members in base_by_group.items()}
    pool_rows = []
    include_refs = set()
    suffix_counts = Counter()
    for row in sorted(source_rows, key=lambda item: (item["official_split"], item["filename"], item["variant"])):
        split, name, variant = row["official_split"], row["filename"], row["variant"]
        group_id = row["source_group_id"]
        aug = augmentation_type(name)
        if variant == "original":
            suffix_counts["plain" if aug == "none" else aug] += 1
        ref = (split, name, variant)
        annotation_policy = "raw_annotation_unchanged"
        if (split, name) == ("test", "IMG_54350.jpg"):
            annotation_policy = "derived_valid_three_class_only; region_3_invalid_unknown; evaluator_FP_probe_required"
        if variant == "resize":
            reason = "dataset_provided_resize"
            kind = "dataset_resize"
        elif group_id in fallback:
            reason = "no_base_fallback_requires_review"
            kind = "fallback_augmented_only"
        elif aug != "none":
            reason = "dataset_offline_augmentation"
            kind = "offline_augmentation"
        elif group_id in conflict:
            reason = "annotation_conflict_group_holdout"
            kind = "plain_base"
        elif canonical_by_group[group_id] != (split, name):
            canonical_ref = (*canonical_by_group[group_id], "original")
            reason = ("exact_image_duplicate_representation"
                      if file_by_ref[canonical_ref]["sha256"] == file_by_ref[ref]["sha256"]
                      else "alternate_same_source_base")
            kind = "plain_base"
        else:
            reason = ""
            kind = "plain_base"
            include_refs.add((split, name))
        pool_rows.append({"official_split": split, "filename": name, "variant": variant,
                          "source_group_id": group_id,
                          "split_guard_cluster_id": cluster_for_group[group_id],
                          "representation_type": kind, "augmentation_type": aug,
                          "include_candidate": "true" if not reason else "false",
                          "exclusion_reason": reason, "annotation_policy": annotation_policy,
                          "notes": "Candidate policy only; raw ZIP/JSON unchanged; no final split assigned"})
    selected_groups = {row["source_group_id"] for row in pool_rows if row["include_candidate"] == "true"}
    summary = {"non_resize_representations": len(originals),
               "offline_augmentation_counts": dict(sorted(suffix_counts.items())),
               "plain_base_representations": suffix_counts["plain"],
               "groups_with_base": len(base_by_group),
               "groups_without_base": no_base,
               "fallback_representatives": {gid: f"{r['official_split']}/{r['filename']}"
                                            for gid, r in fallback.items()},
               "base_after_exact_sha_dedup": len(chosen_by_sha),
               "base_after_one_per_source_group": len(canonical_by_group),
               "included_candidate_images": len(include_refs),
               "included_source_groups": len(selected_groups),
               "excluded_conflict_source_groups": len(conflict),
               "excluded_conflict_canonical_images": len(canonical_by_group) - len(include_refs),
               **conflict_info}
    return pool_rows, summary


def valid_labels(regions: list[dict]) -> list[str]:
    return [name for region in regions
            if (name := region.get("region_attributes", {}).get("name")) in CLASSES]


def aggregate_for_clusters(pool_rows: list[dict], annotations: dict) -> dict[str, dict]:
    clusters = defaultdict(lambda: {"images": 0, "source_groups": set(), "classes": Counter(),
                                    "multi_class_images": 0})
    for row in pool_rows:
        if row["include_candidate"] != "true":
            continue
        cluster = clusters[row["split_guard_cluster_id"]]
        cluster["source_groups"].add(row["source_group_id"])
        labels = valid_labels(annotations[(row["official_split"], row["filename"])])
        cluster["images"] += 1
        cluster["classes"].update(labels)
        cluster["multi_class_images"] += len(set(labels)) > 1
    return dict(clusters)


def assign_clusters(clusters: dict[str, dict], target: tuple[float, float, float], seed: int) -> tuple[dict, dict]:
    """Largest-first greedy; lexicographic image/class/multiclass/cluster error."""
    totals = {"images": sum(x["images"] for x in clusters.values()),
              "source_groups": sum(len(x["source_groups"]) for x in clusters.values()),
              "guard_clusters": len(clusters),
              "multi_class_images": sum(x["multi_class_images"] for x in clusters.values()),
              **{name: sum(x["classes"][name] for x in clusters.values()) for name in CLASSES}}
    rng = random.Random(seed)
    order = sorted(clusters)
    rng.shuffle(order)
    order.sort(key=lambda cid: -clusters[cid]["images"])
    counts = {split: {key: 0 for key in totals} for split in SPLITS}
    assigned = {}

    def objective(candidate: str, cluster: dict) -> tuple:
        def projected(split: str, key: str) -> int:
            increment = (cluster["classes"][key] if key in CLASSES else
                         len(cluster["source_groups"]) if key == "source_groups" else
                         1 if key == "guard_clusters" else cluster[key])
            return counts[split][key] + (increment if split == candidate else 0)
        def error(key: str) -> float:
            denominator = totals[key]
            return sum(abs(projected(split, key) / denominator - target[i])
                       for i, split in enumerate(SPLITS)) if denominator else 0.0
        return (round(error("images"), 12),
                round(sum(error(name) for name in CLASSES) / len(CLASSES), 12),
                round(error("multi_class_images"), 12),
                round(error("guard_clusters"), 12),
                SPLITS.index(candidate))

    for cid in order:
        cluster = clusters[cid]
        chosen = min(SPLITS, key=lambda split: objective(split, cluster))
        assigned[cid] = chosen
        counts[chosen]["images"] += cluster["images"]
        counts[chosen]["source_groups"] += len(cluster["source_groups"])
        counts[chosen]["guard_clusters"] += 1
        counts[chosen]["multi_class_images"] += cluster["multi_class_images"]
        for name in CLASSES:
            counts[chosen][name] += cluster["classes"][name]
    return assigned, counts


def simulate(pool_rows: list[dict], guard_rows: list[dict], relations: list[dict],
             annotations: dict, seed: int) -> dict:
    clusters = aggregate_for_clusters(pool_rows, annotations)
    all_cluster_ids = {row["split_guard_cluster_id"] for row in guard_rows}
    excluded_only_clusters = all_cluster_ids - set(clusters)
    cluster_by_ref = {(row["official_split"], row["filename"]): row["split_guard_cluster_id"]
                      for row in guard_rows}
    total_images = sum(row["include_candidate"] == "true" for row in pool_rows)
    scenarios = {}
    for title, target in (("70/15/15", (0.7, 0.15, 0.15)), ("70/10/20", (0.7, 0.1, 0.2))):
        assigned, counts = assign_clusters(clusters, target, seed)
        assigned.update({cid: "excluded" for cid in excluded_only_clusters})
        cuts = Counter()
        for edge in relations:
            if edge["variant_a"] != edge["variant_b"] or edge["variant_a"] != "original":
                continue
            a, b = relation_refs(edge)
            if assigned[cluster_by_ref[a]] != assigned[cluster_by_ref[b]]:
                cuts[edge["confidence"]] += 1
        if sum(cuts.values()) != 0:
            raise AssertionError(f"candidate relation crossed guard assignment: {cuts}")
        if sum(counts[split]["images"] for split in SPLITS) != total_images:
            raise AssertionError("candidate image count not conserved")
        totals = {key: sum(counts[split][key] for split in SPLITS) for key in counts["train"]}
        deviations = {split: {key: round(100 * (counts[split][key] / totals[key] - target[i]), 3)
                               if totals[key] else 0.0
                               for key in ("images", *CLASSES, "multi_class_images")}
                      for i, split in enumerate(SPLITS)}
        scenarios[title] = {
            "status": "one deterministic simulation; no split saved or frozen",
            "counts": counts,
            "totals": totals,
            "excluded_only_guard_clusters": len(excluded_only_clusters),
            "deviation_percentage_points": deviations,
            "max_abs_image_deviation_pp": max(abs(x["images"]) for x in deviations.values()),
            "max_abs_class_deviation_pp": max(abs(x[name]) for x in deviations.values() for name in CLASSES),
            "max_abs_multiclass_deviation_pp": max(abs(x["multi_class_images"]) for x in deviations.values()),
            "all_original_relation_crossings": dict(cuts),
            "all_original_relation_crossings_total": sum(cuts.values()),
        }
    return scenarios


def build(archive_path: Path, manifest_dir: Path) -> dict:
    raw_sha = digest_file(archive_path)
    expected = "049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce"
    if raw_sha != expected:
        raise ValueError(f"D2 v4 raw ZIP SHA256 mismatch: {raw_sha}")
    source_rows = read_csv(manifest_dir / "source_groups.csv")
    files = read_csv(manifest_dir / "files_sha256.csv")
    relations = read_csv(manifest_dir / "source_group_relations.csv")
    originals = [row for row in source_rows if row["variant"] == "original"]
    guards, cluster_for_group, guard_summary = build_guards(originals, relations)
    with zipfile.ZipFile(archive_path) as archive:
        annotations = read_original_annotations(archive)
    pool, pool_summary = build_pool(source_rows, files, cluster_for_group, annotations)
    seed = seed_from_archive_sha(raw_sha)
    scenarios = simulate(pool, guards, relations, annotations, seed)
    write_csv(manifest_dir / "split_guard_clusters.csv", GUARD_FIELDS, guards)
    write_csv(manifest_dir / "d2_experiment_pool_candidate.csv", POOL_FIELDS, pool)
    return {"status": "Stage 2B-5P candidate only; D2 NOT FROZEN",
            "archive_sha256": raw_sha,
            "seed_policy": "first 64 bits of SHA256(raw ZIP SHA256 bytes || fixed protocol context)",
            "seed_context": SEED_CONTEXT.decode("ascii"),
            "simulation_seed_candidate": seed,
            "guard": guard_summary, "pool": pool_summary, "scenarios": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.manifest_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
