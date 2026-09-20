"""Build and validate the approved D2 protocol; never change the raw ZIP.

Run twice into independent directories and compare every output byte before
publishing artifacts. Split membership is determined only by the fixed ZIP,
evidence graph, approved policies and the protocol seed context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
spec = importlib.util.spec_from_file_location("d2_candidate", SCRIPT.with_name("10_prepare_d2_freeze_candidate.py"))
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)

ZIP_SHA = "049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce"
DOI = "10.17632/gfcmdbvw65.4"
PROTOCOL = "d2-v4-stage2b5-v1"
GENERATOR = "12_freeze_d2_protocol.py@1 + 10_prepare_d2_freeze_candidate.py@1"
RATIO = "70/15/15"
REPRESENTATION = "one plain original per evidence source group; no dataset resize/offline augmentation; pipeline online only"
ANNOTATION = "valid three-class raw regions; U01 drop raw region #3; standard background/FP; no ignore region"
NAMES = ("d2_experiment_pool_frozen.csv", "d2_split_frozen.csv", "d2_exclusions_frozen.csv")
FIELDS = (
    "dataset_doi", "dataset_version", "zip_filename", "zip_sha256", "protocol_version",
    "generator_version", "generator_sha256", "git_head", "split_seed_64", "split_seed_32",
    "split_ratio", "official_split", "filename", "raw_image_zip_member", "raw_annotation_zip_member",
    "variant", "source_group_id", "split_guard_cluster_id", "representation_policy",
    "representation_type", "augmentation_type", "include", "exclusion_reason", "annotation_policy",
    "dropped_raw_region_indices", "final_split", "immature_instances", "semi_mature_instances",
    "mature_instances", "derived_target_count", "multi_class_image",
)


def derived_regions(regions: list[dict], split: str, filename: str) -> tuple[list[dict], str]:
    if (split, filename) == ("test", "IMG_54350.jpg"):
        if (len(regions) != 4 or
                regions[3].get("region_attributes", {}).get("name") in candidate.CLASSES or
                len(candidate.valid_labels(regions[:3])) != 3):
            raise ValueError("U01 differs from human-reviewed raw annotation")
        result, dropped = regions[:3], "3"
    else:
        result, dropped = regions, ""
    if len(candidate.valid_labels(result)) != len(result):
        raise ValueError(f"unexpected invalid class in {split}/{filename}")
    return result, dropped


def git_head() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent,
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def yaml_scalar(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_yaml(path: Path, report: dict, hashes: dict, generator_sha: str, supporting_sha: str) -> None:
    """A simple, deterministic YAML mapping; JSON values are legal YAML scalars."""
    meta = {
        "dataset_doi": DOI, "dataset_version": "v4", "zip_filename": "dataset-20260508.zip",
        "zip_size_bytes": report["zip_size_bytes"], "zip_sha256": ZIP_SHA,
        "protocol_version": PROTOCOL, "generator_version": GENERATOR,
        "generator_sha256": generator_sha, "supporting_generator_sha256": supporting_sha,
        "git_head": report["git_head"], "split_seed_64": report["seed64"],
        "split_seed_32": report["seed32"],
        "split_seed_formula": "first_64_bits_be(SHA256(bytes.fromhex(zip_sha256) || ASCII(seed_context)))",
        "seed_32_formula": "seed_64 & 0xffffffff (only if a future dependency requires 32-bit)",
        "seed_context": candidate.SEED_CONTEXT.decode("ascii"), "split_ratio": RATIO,
        "split_unit": "split_guard_cluster", "assignment_algorithm": "10_prepare_d2_freeze_candidate.py@1 assign_clusters; largest first, seeded tie order, lexicographic global L1 images/classes/multiclass/clusters",
        "source_group_semantics": "evidence-based grouping, not independent physical acquisition",
        "representation_policy": REPRESENTATION, "annotation_policy": ANNOTATION,
        "class_mapping": {str(i): name for i, name in enumerate(candidate.CLASSES)},
        "all_source_groups": report["all_source_groups"], "all_guard_clusters": report["all_guard_clusters"],
        "pool_images": report["pool_images"], "pool_source_groups": report["pool_source_groups"],
        "pool_guard_clusters": report["pool_guard_clusters"], "split_statistics": report["counts"],
        "exclusion_counts": report["exclusions"], "relation_crossings": report["relation_crossings"],
        "source_groups_sha256": report["source_groups_sha256"],
        "relations_sha256": report["relations_sha256"],
        "frozen_csv_sha256": hashes,
        "raw_annotation_policy_note": "ZIP JSON unchanged; derived targets are the valid regions referenced by CSV counts; U01 skips zero-based region 3",
        "evaluation_limit": "No fruit/tree/session acquisition IDs; actual YOLO evaluator integration deferred to Baseline setup",
    }
    path.write_text("# D2 frozen protocol; generated deterministically, do not edit in place.\n" +
                    "\n".join(f"{key}: {yaml_scalar(value)}" for key, value in meta.items()) + "\n",
                    encoding="utf-8", newline="\n")


def build(archive_path: Path, manifest_dir: Path, output_dir: Path) -> dict:
    if candidate.digest_file(archive_path) != ZIP_SHA:
        raise ValueError("raw ZIP SHA256 changed")
    if archive_path.name != "dataset-20260508.zip":
        raise ValueError("unexpected archive filename")
    sources = candidate.read_csv(manifest_dir / "source_groups.csv")
    files = candidate.read_csv(manifest_dir / "files_sha256.csv")
    relations = candidate.read_csv(manifest_dir / "source_group_relations.csv")
    originals = [row for row in sources if row["variant"] == "original"]
    guards, cluster_by_group, guard_stats = candidate.build_guards(originals, relations)
    existing_guards = candidate.read_csv(manifest_dir / "split_guard_clusters.csv")
    if guards != existing_guards:
        raise AssertionError("persisted guard manifest differs from recomputed graph")
    with zipfile.ZipFile(archive_path) as archive:
        annotations = candidate.read_original_annotations(archive)
        member_names = set(archive.namelist())
        root = {name.split("/")[0] for name in member_names if name}
        if len(root) != 1:
            raise ValueError("unexpected ZIP root")
        root = root.pop()
    pool, pool_stats = candidate.build_pool(sources, files, cluster_by_group, annotations)
    if pool != candidate.read_csv(manifest_dir / "d2_experiment_pool_candidate.csv"):
        raise AssertionError("candidate pool changed since protocol approval")
    selected = [row for row in pool if row["include_candidate"] == "true"]
    clusters = candidate.aggregate_for_clusters(pool, annotations)
    seed64 = candidate.seed_from_archive_sha(ZIP_SHA)
    if seed64 != 13436313853456744620:
        raise AssertionError("seed policy changed")
    seed32 = seed64 & 0xFFFFFFFF
    assigned, counts = candidate.assign_clusters(clusters, (0.7, 0.15, 0.15), seed64)
    selected_by_ref = {(r["official_split"], r["filename"]): r for r in selected}
    if len(selected_by_ref) != 1099:
        raise AssertionError("unexpected canonical selection")
    by_ref = {}
    generator_sha = candidate.digest_file(SCRIPT)
    supporting_sha = candidate.digest_file(SCRIPT.with_name("10_prepare_d2_freeze_candidate.py"))
    commit = git_head()
    included, excluded = [], []
    for item in pool:
        split, filename, variant = item["official_split"], item["filename"], item["variant"]
        group, cluster = item["source_group_id"], item["split_guard_cluster_id"]
        is_included = item["include_candidate"] == "true"
        if is_included and (variant != "original" or item["augmentation_type"] != "none"):
            raise AssertionError("offline/resize in formal pool")
        regions, dropped = derived_regions(annotations[(split, filename)], split, filename)
        labels = candidate.valid_labels(regions)
        class_counts = Counter(labels)
        row = dict(dataset_doi=DOI, dataset_version="v4", zip_filename=archive_path.name,
                   zip_sha256=ZIP_SHA, protocol_version=PROTOCOL, generator_version=GENERATOR,
                   generator_sha256=generator_sha, git_head=commit, split_seed_64=seed64,
                   split_seed_32=seed32, split_ratio=RATIO, official_split=split, filename=filename,
                   raw_image_zip_member=f"{root}/{split}{'_resize' if variant == 'resize' else ''}/{filename}",
                   raw_annotation_zip_member=f"{root}/{split}{'_resize' if variant == 'resize' else ''}.json",
                   variant=variant, source_group_id=group, split_guard_cluster_id=cluster,
                   representation_policy=REPRESENTATION, representation_type=item["representation_type"],
                   augmentation_type=item["augmentation_type"], include=str(is_included).lower(),
                   exclusion_reason=item["exclusion_reason"], annotation_policy=ANNOTATION,
                   dropped_raw_region_indices=dropped,
                   final_split=assigned[cluster] if is_included else "excluded",
                   immature_instances=class_counts[candidate.CLASSES[0]],
                   semi_mature_instances=class_counts[candidate.CLASSES[1]],
                   mature_instances=class_counts[candidate.CLASSES[2]],
                   derived_target_count=len(regions), multi_class_image=int(len(set(labels)) > 1))
        if row["raw_image_zip_member"] not in member_names or row["raw_annotation_zip_member"] not in member_names:
            raise AssertionError("manifest references missing ZIP member")
        if is_included:
            included.append(row)
            by_ref[(split, filename)] = row
        else:
            excluded.append(row)
    if len(included) != 1099 or len(excluded) != 1713:
        raise AssertionError("pool/exclusion conservation failed")
    if any(row["final_split"] not in candidate.SPLITS for row in included):
        raise AssertionError("unassigned pool image")
    conflict, conflict_info = candidate.conflict_groups(originals, files, annotations)
    no_base = set(pool_stats["groups_without_base"])
    if len(conflict) != 7 or len(no_base) != 6:
        raise AssertionError("conflict/no-base cardinality changed")
    if {row["source_group_id"] for row in included} & (conflict | no_base):
        raise AssertionError("conflict/no-base group entered pool")
    if any(row["source_group_id"] in (conflict | no_base) for row in selected):
        raise AssertionError("policy mismatch")
    u01 = by_ref[("test", "IMG_54350.jpg")]
    if u01["derived_target_count"] != 3 or u01["dropped_raw_region_indices"] != "3":
        raise AssertionError("U01 did not retain exactly three targets")
    if sum(len(candidate.valid_labels(annotations[("test", "IMG_54350.jpg")])) for _ in [0]) != 3:
        raise AssertionError("U01 raw class count changed")
    image_sha = {(r["split"], r["filename"]): r["sha256"] for r in files if r["variant"] == "original"}
    sha_to_split = defaultdict(set)
    for ref, row in by_ref.items():
        sha_to_split[image_sha[ref]].add(row["final_split"])
    if any(len(value) > 1 for value in sha_to_split.values()):
        raise AssertionError("exact duplicate crossing")
    group_to_split, cluster_to_split = defaultdict(set), defaultdict(set)
    for row in included:
        group_to_split[row["source_group_id"]].add(row["final_split"])
        cluster_to_split[row["split_guard_cluster_id"]].add(row["final_split"])
    if any(len(value) > 1 for value in (*group_to_split.values(), *cluster_to_split.values())):
        raise AssertionError("group/guard crossing")
    edge_audit = Counter()
    for edge in relations:
        if edge["variant_a"] != "original" or edge["variant_b"] != "original":
            continue
        a, b = candidate.relation_refs(edge)
        if a not in by_ref or b not in by_ref:
            edge_audit["excluded_endpoint"] += 1
            continue
        if by_ref[a]["final_split"] != by_ref[b]["final_split"]:
            edge_audit[edge["confidence"]] += 1
        else:
            edge_audit["within_same_split"] += 1
    crossing = {key: edge_audit[key] for key in ("Confirmed", "Strongly Supported", "Supported", "Candidate", "Unresolved")}
    if any(crossing.values()) or guard_stats["unguarded_residual_risk_edges"]:
        raise AssertionError(f"registered relation crossing {crossing}")
    computed = {}
    for split in candidate.SPLITS:
        rows = [row for row in included if row["final_split"] == split]
        computed[split] = dict(images=len(rows), source_groups=len({r["source_group_id"] for r in rows}),
                               guard_clusters=len({r["split_guard_cluster_id"] for r in rows}),
                               immature=sum(r["immature_instances"] for r in rows),
                               semi_mature=sum(r["semi_mature_instances"] for r in rows),
                               mature=sum(r["mature_instances"] for r in rows),
                               multi_class_images=sum(r["multi_class_image"] for r in rows))
        expected = counts[split]
        if (computed[split]["images"] != expected["images"] or
            computed[split]["source_groups"] != expected["source_groups"] or
            computed[split]["guard_clusters"] != expected["guard_clusters"] or
            computed[split]["multi_class_images"] != expected["multi_class_images"] or
            any(computed[split][label] != expected[name] for label, name in
                zip(("immature", "semi_mature", "mature"), candidate.CLASSES))):
            raise AssertionError("derived class and assignment totals disagree")
    report = dict(zip_size_bytes=archive_path.stat().st_size, git_head=commit, seed64=seed64,
                  seed32=seed32, all_source_groups=guard_stats["source_groups"],
                  all_guard_clusters=guard_stats["split_guard_clusters"],
                  pool_images=len(included), pool_source_groups=len(group_to_split),
                  pool_guard_clusters=len(cluster_to_split), counts=computed,
                  exclusions=dict(sorted(Counter(r["exclusion_reason"] for r in excluded).items())),
                  conflict_groups=sorted(conflict), no_base_groups=sorted(no_base),
                  annotation_conflict_sha_pairs=conflict_info["sha_conflict_pairs"],
                  relation_crossings=crossing, relation_edges_in_pool=edge_audit["within_same_split"],
                  relation_edges_with_excluded_endpoint=edge_audit["excluded_endpoint"],
                  source_groups_sha256=candidate.digest_file(manifest_dir / "source_groups.csv"),
                  relations_sha256=candidate.digest_file(manifest_dir / "source_group_relations.csv"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / NAMES[0], included)
    write_rows(output_dir / NAMES[1], sorted(included, key=lambda r: (candidate.SPLITS.index(r["final_split"]), r["official_split"], r["filename"])))
    write_rows(output_dir / NAMES[2], excluded)
    hashes = {name: candidate.digest_file(output_dir / name) for name in NAMES}
    write_yaml(output_dir / "d2_frozen_protocol.yaml", report, hashes, generator_sha, supporting_sha)
    report["artifact_sha256"] = {**hashes, "d2_frozen_protocol.yaml": candidate.digest_file(output_dir / "d2_frozen_protocol.yaml")}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/multistage_apple_v4/dataset-20260508.zip")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.manifest_dir, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
