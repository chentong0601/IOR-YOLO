"""Resolve D2 image relations into conservative, deterministic source groups.

Reads the original ZIP and Stage 2B-3 manifests. Never edits raw data or splits.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Pillow required: uv run --with pillow python scripts/06_build_source_groups.py ...") from exc


SUFFIX = re.compile(r"_(brightness|noise|gaussian|hsv|gamma)$", re.I)
GROUP_FIELDS = ("filename", "variant", "official_split", "source_group_id",
                "canonical_sample", "relation_to_canonical", "relation_type",
                "evidence", "confidence", "class_labels", "instance_count",
                "unlabeled_count", "width", "height", "notes")
EDGE_FIELDS = ("file_a", "split_a", "variant_a", "file_b", "split_b", "variant_b",
               "relation_type", "evidence", "confidence", "merged")
FAMILY_FIELDS = ("source_family", "final_status", "original_members", "cross_split_pair_count",
                 "decision_evidence", "remaining_uncertainty")
RANK = {"Unresolved": 0, "Candidate": 1, "Supported": 2,
        "Strongly Supported": 3, "Confirmed": 4}
ADJUDICATIONS = Path(__file__).resolve().parents[1] / "configs" / "data" / "d2_source_group_adjudications.json"
ABNORMAL_RESIZE = {("train", "172_brightness.jpg"),
                   ("val", "172_noise.jpg"), ("train", "327.jpg")}
UNKNOWN_REGION = ("test", "IMG_54350.jpg")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict) -> tuple[str, str]:
    return row["split"], row["filename"]


def image_ref(row: dict) -> str:
    return f"{row['split']}/{row['filename']}"


def source_family(filename: str) -> str:
    return SUFFIX.sub("", Path(filename).stem).casefold()


def load_human_adjudications(path: Path) -> dict[frozenset[str], dict]:
    """Load explicit human decisions without changing automatic evidence thresholds."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("stage") != "Stage 2B-4H" or len(data.get("accepted_edges", [])) != 4:
        raise ValueError("expected four Stage 2B-4H accepted edges")
    decisions = {}
    for item in data["accepted_edges"]:
        pair = frozenset(item["files"])
        if (len(pair) != 2 or item["reviewed_confidence"] != "Strongly Supported"
                or pair in decisions):
            raise ValueError(f"invalid or duplicate human adjudication: {item}")
        decisions[pair] = item
    return decisions


class Components:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left, right) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return False
        self.parent[max(a, b)] = min(a, b)
        return True

    def groups(self):
        groups = defaultdict(list)
        for node in sorted(self.parent):
            groups[self.find(node)].append(node)
        return sorted(groups.values(), key=lambda members: members[0])


def correlation(left: bytes, right: bytes) -> float:
    n = len(left)
    if n != len(right) or not n:
        return float("nan")
    sx, sy = sum(left), sum(right)
    vx = n * sum(x * x for x in left) - sx * sx
    vy = n * sum(y * y for y in right) - sy * sy
    if not vx or not vy:
        return 1.0 if left == right else 0.0
    return (n * sum(x * y for x, y in zip(left, right)) - sx * sy) / math.sqrt(vx * vy)


def image_gray(archive, root: str, row: dict, size=None) -> bytes:
    with Image.open(io.BytesIO(archive.read(f"{root}/{row['split']}/{row['filename']}"))) as image:
        image = image.convert("L")
        if size:
            image = image.resize(size, Image.Resampling.BILINEAR)
        return image.tobytes()


def load_annotations(archive, root: str) -> dict[tuple[str, str], list]:
    annotations = {}
    for split in ("train", "val", "test"):
        data = json.loads(archive.read(f"{root}/{split}.json"))
        for record in data.values():
            regions = record["regions"]
            annotations[(split, record["filename"])] = (
                list(regions.values()) if isinstance(regions, dict) else regions)
    return annotations


def instance_labels(regions: list) -> tuple[list[str], int]:
    labels = []
    unlabeled = 0
    for region in regions:
        value = region.get("region_attributes", {}).get("name")
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())
        else:
            unlabeled += 1
    return labels, unlabeled


def classify_existing(row: dict, originals: dict, annotations: dict,
                      archive, root: str) -> tuple[str, str]:
    """Promote a relation only with evidence beyond naming or dHash alone."""
    if row["relation_type"] == "exact_duplicate":
        return "exact_duplicate", "Confirmed"
    if row["status"] not in ("Supported", "Confirmed"):
        return ("same_annotation_geometry" if row["relation_type"] == "same_annotation_geometry"
                else "unresolved_candidate"), row["status"]
    left = originals[(row["split_a"], row["file_a"])]
    right = originals[(row["split_b"], row["file_b"])]
    if row["source_family"].startswith("visual:"):
        # Previous dHash + low-resolution correlation is only a shortlist.
        # Full-resolution correlation is an independent additional check.
        if (left["width"], left["height"]) == (right["width"], right["height"]):
            full_corr = correlation(image_gray(archive, root, left),
                                    image_gray(archive, root, right))
        else:
            full_corr = float("nan")
        if math.isfinite(full_corr) and full_corr >= 0.96:
            return "high_visual_similarity", "Supported"
        return "unresolved_candidate", "Candidate"
    same_geometry = annotations[key(left)] == annotations[key(right)]
    if same_geometry:
        return "augmentation_derivative", "Strongly Supported"
    return "high_visual_similarity", "Supported"


def build(archive_path: Path, manifest_dir: Path) -> dict:
    adjudications = load_human_adjudications(ADJUDICATIONS)
    adjudicated_seen = set()
    files = read_csv(manifest_dir / "files_sha256.csv")
    previous_edges = read_csv(manifest_dir / "duplicate_report.csv")
    originals = {(r["split"], r["filename"]): r for r in files if r["variant"] == "original"}
    if len(originals) != 1406 or len(files) != 2812:
        raise ValueError("expected audited D2 v4 manifests with 1406 original and 1406 resize images")
    if any(r["decode_status"] != "decoded" for r in files):
        raise ValueError("decode errors present; resolve before grouping")
    components = Components(originals)
    edges = []
    named_cross = defaultdict(list)
    dhash_effective = 0
    dhash_supported = 0
    with zipfile.ZipFile(archive_path) as archive:
        roots = {name.split("/")[0] for name in archive.namelist() if name}
        if len(roots) != 1:
            raise ValueError("expected one ZIP root")
        root = roots.pop()
        annotations = load_annotations(archive, root)
        if set(annotations) != set(originals):
            raise ValueError("original JSON records do not match file manifest")
        # Within-split suffix families were not covered by prior cross-split report.
        families = defaultdict(list)
        for row in originals.values():
            families[row["source_family"]].append(row)
        compact_signatures = {}
        for family, members in sorted(families.items()):
            for i, left in enumerate(sorted(members, key=key)):
                for right in sorted(members, key=key)[i + 1:]:
                    if left["split"] != right["split"]:
                        continue
                    for item in (left, right):
                        compact_signatures.setdefault(key(item), image_gray(archive, root, item, (64, 64)))
                    corr = correlation(compact_signatures[key(left)], compact_signatures[key(right)])
                    same_geometry = annotations[key(left)] == annotations[key(right)]
                    confidence = ("Strongly Supported" if corr >= 0.995 and same_geometry
                                  else "Supported" if corr >= 0.995 else "Candidate")
                    merged = confidence in ("Supported", "Strongly Supported")
                    if merged:
                        components.union(key(left), key(right))
                    edges.append({"file_a": left["filename"], "split_a": left["split"],
                                  "variant_a": "original", "file_b": right["filename"],
                                  "split_b": right["split"], "variant_b": "original",
                                  "relation_type": "augmentation_derivative" if merged else "filename_family",
                                  "evidence": f"same suffix-normalized family={family}; 64x64 correlation={corr:.6f}; identical polygons={same_geometry}",
                                  "confidence": confidence, "merged": str(merged).lower()})
        # Exact duplicates before visual relations; visual impact is measured only
        # after all other supported relations are already in the graph.
        ordered = sorted(previous_edges, key=lambda r: (
            0 if r["relation_type"] == "exact_duplicate" else
            1 if not r["source_family"].startswith("visual:") else 2,
            r["split_a"], r["file_a"], r["split_b"], r["file_b"]))
        for row in ordered:
            if row["variant_a"] != row["variant_b"] or row["variant_a"] != "original":
                continue
            relation, confidence = classify_existing(row, originals, annotations, archive, root)
            automatic_confidence = confidence
            left_ref = f"{row['split_a']}/{row['file_a']}"
            right_ref = f"{row['split_b']}/{row['file_b']}"
            pair = frozenset((left_ref, right_ref))
            decision = adjudications.get(pair)
            if decision is not None:
                if confidence != decision["prior_confidence"]:
                    raise ValueError(f"pre-review evidence changed for {decision['case_id']}: {confidence}")
                confidence = decision["reviewed_confidence"]
                adjudicated_seen.add(pair)
            visual = row["source_family"].startswith("visual:")
            if visual and automatic_confidence == "Supported":
                dhash_supported += 1
            merged = confidence in ("Confirmed", "Strongly Supported", "Supported")
            changed = components.union((row["split_a"], row["file_a"]),
                                       (row["split_b"], row["file_b"])) if merged else False
            if visual and changed:
                dhash_effective += 1
            if not visual and row["split_a"] != row["split_b"] and not row["source_family"].startswith("sha256:"):
                named_cross[row["source_family"]].append(confidence)
            evidence = row["evidence"]
            if visual and relation == "high_visual_similarity":
                left = originals[(row["split_a"], row["file_a"])]
                right = originals[(row["split_b"], row["file_b"])]
                evidence += ("; full-resolution grayscale correlation="
                             f"{correlation(image_gray(archive, root, left), image_gray(archive, root, right)):.6f}")
            if decision is not None:
                evidence += (f"; Stage 2B-4H {decision['case_id']} human accepted same-source relation"
                             "; physical acquisition ID unverified")
            edges.append({"file_a": row["file_a"], "split_a": row["split_a"],
                          "variant_a": "original", "file_b": row["file_b"],
                          "split_b": row["split_b"], "variant_b": "original",
                          "relation_type": relation, "evidence": evidence,
                          "confidence": confidence, "merged": str(merged).lower()})
        # Resized representation inherits its same-name original's group for
        # accounting. Three questionable image pairs remain explicitly unresolved.
        for row in previous_edges:
            if row["variant_a"] != "original" or row["variant_b"] != "resize":
                continue
            confidence = row["status"]
            abnormal = (row["split_a"], row["file_a"]) in ABNORMAL_RESIZE
            evidence = row["evidence"]
            if abnormal:
                if confidence != "Unresolved":
                    raise ValueError(f"abnormal resize status changed: {row['file_a']}")
                evidence += ("; Stage 2B-4H human accepted same-source companion;"
                             " deterministic resize equivalence unresolved;"
                             " dataset-provided resize excluded from formal experiments")
            edges.append({"file_a": row["file_a"], "split_a": row["split_a"],
                          "variant_a": "original", "file_b": row["file_b"],
                          "split_b": row["split_b"], "variant_b": "resize",
                          "relation_type": "same_source_companion" if abnormal else "resized_duplicate" if confidence != "Unresolved" else "unresolved_candidate",
                          "evidence": evidence, "confidence": confidence,
                          "merged": "true" if confidence == "Supported" else "false"})
    if adjudicated_seen != set(adjudications):
        missing = [adjudications[pair]["case_id"] for pair in set(adjudications) - adjudicated_seen]
        raise ValueError(f"human-adjudicated edges missing from source report: {missing}")
    ordered_groups = components.groups()
    canonical = {}
    group_for = {}
    for index, members in enumerate(ordered_groups, 1):
        gid = f"sg-{index:04d}"
        # Prefer a non-augmented original; no assertion that it is a capture ID.
        representative = min(members, key=lambda x: (bool(SUFFIX.search(Path(x[1]).stem)), x))
        canonical[gid] = representative
        for member in members:
            group_for[member] = gid
    by_original = defaultdict(list)
    for edge in edges:
        if edge["variant_a"] == edge["variant_b"] == "original":
            by_original[(edge["split_a"], edge["file_a"])].append(edge)
            by_original[(edge["split_b"], edge["file_b"])].append(edge)
    resize_edge = {(edge["split_b"], edge["file_b"]): edge for edge in edges
                   if edge["variant_a"] == "original" and edge["variant_b"] == "resize"}
    output = []
    for row in sorted(files, key=lambda r: (r["split"], r["filename"], r["variant"])):
        node = key(row)
        gid = group_for[node]
        labels, unknown = instance_labels(annotations[node])
        if row["variant"] == "resize":
            relation_edge = resize_edge.get(node)
            if relation_edge is None:
                raise ValueError(f"missing resize edge: {node}")
            relation = relation_edge["relation_type"]
            confidence = relation_edge["confidence"]
            evidence = relation_edge["evidence"]
            notes = "same-name companion only; representation equivalence unresolved" if confidence == "Unresolved" else "derived representation; excluded from independent image universe"
        elif node == canonical[gid]:
            relation, confidence, evidence, notes = "canonical", "Confirmed", "deterministic representative selection", "representative, not verified acquisition original"
        else:
            incident = [edge for edge in by_original[node]
                        if edge["merged"] == "true"
                        and group_for[(edge["split_a"], edge["file_a"])] == gid
                        and group_for[(edge["split_b"], edge["file_b"])] == gid]
            strongest = max(incident, key=lambda edge: RANK[edge["confidence"]])
            relation, confidence, evidence = (strongest["relation_type"],
                                              strongest["confidence"], strongest["evidence"])
            notes = "relation through component path; direct relation to canonical not asserted"
        if node == UNKNOWN_REGION:
            notes += "; Stage 2B-4H: region #3 invalid/unknown, excluded from derived three-class targets; other three labels retained"
        output.append({"filename": row["filename"], "variant": row["variant"],
                       "official_split": row["split"], "source_group_id": gid,
                       "canonical_sample": f"{canonical[gid][0]}/{canonical[gid][1]}",
                       "relation_to_canonical": "self" if node == canonical[gid] and row["variant"] == "original" else "component_path" if row["variant"] == "original" else "same_name_original",
                       "relation_type": relation, "evidence": evidence,
                       "confidence": confidence, "class_labels": "|".join(sorted(set(labels))),
                       "instance_count": len(labels), "unlabeled_count": unknown,
                       "width": row["width"], "height": row["height"], "notes": notes})
    write_csv(manifest_dir / "source_groups.csv", GROUP_FIELDS, output)
    write_csv(manifest_dir / "source_group_relations.csv", EDGE_FIELDS,
              sorted(edges, key=lambda r: (r["split_a"], r["file_a"], r["variant_a"],
                                           r["split_b"], r["file_b"], r["variant_b"])))
    family_review = []
    for family, statuses in sorted(named_cross.items()):
        status = max(statuses, key=lambda value: RANK[value])
        members = sorted(image_ref(row) for row in originals.values()
                         if row["source_family"] == family)
        relevant = [edge for edge in edges if edge["variant_a"] == edge["variant_b"] == "original"
                    and edge["split_a"] != edge["split_b"]
                    and source_family(edge["file_a"]) == source_family(edge["file_b"]) == family]
        strongest = max(relevant, key=lambda edge: RANK[edge["confidence"]])
        family_review.append({"source_family": family, "final_status": status,
                              "original_members": "|".join(members),
                              "cross_split_pair_count": len(relevant),
                              "decision_evidence": strongest["evidence"],
                              "remaining_uncertainty": (
                                  "acquisition identity unverified; other family edges may be weaker"
                                  if status != "Candidate" else
                                  "same polygons and suffix are insufficient; pixel correlation below 0.995")})
    write_csv(manifest_dir / "source_family_review.csv", FAMILY_FIELDS, family_review)
    cross = sum(len({node[0] for node in group}) > 1 for group in ordered_groups)
    named_status = Counter(max(values, key=lambda value: RANK[value])
                           for values in named_cross.values())
    return {"original_images": len(originals), "source_groups": len(ordered_groups),
            "group_size_distribution": dict(sorted(Counter(map(len, ordered_groups)).items())),
            "largest_group": max(map(len, ordered_groups)), "cross_official_split_groups": cross,
            "named_cross_families": len(named_cross), "named_family_status": dict(named_status),
            "dhash_supported_pairs": dhash_supported,
            "dhash_pairs_merging_previous_components": dhash_effective,
            "relation_confidence_counts": dict(Counter(edge["confidence"] for edge in edges)),
            "valid_instances": sum(int(row["instance_count"]) for row in output if row["variant"] == "original"),
            "unlabeled_regions": sum(int(row["unlabeled_count"]) for row in output if row["variant"] == "original")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.archive, args.manifest_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
