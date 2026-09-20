"""Independent, read-only D2 source-graph verification from the ZIP and JSON.

This deliberately does not import or use the Stage 2B-3/4 relation builders.
It compares a freshly computed partition with source_groups.csv only afterward.
All output is confined to a verification report directory; no split is written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import statistics
import platform
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageFilter, ImageStat
except ImportError as exc:
    raise SystemExit("Pillow required: uv run --with pillow python scripts/08_verify_d2_groups.py ...") from exc


SPLITS = ("train", "val", "test")
CLASSES = ("immature apple", "semi-mature apple", "mature apple")
SUFFIX = re.compile(r"_(brightness|noise|gaussian|hsv|gamma)$", re.I)
RANK = {"Unresolved": 0, "Candidate": 1, "Supported": 2,
        "Strongly Supported": 3, "Confirmed": 4}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, fields, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data)


def corr(left: bytes, right: bytes) -> float:
    n = len(left)
    if n != len(right) or not n:
        return float("nan")
    mx, my = sum(left) / n, sum(right) / n
    covariance = sum((x - mx) * (y - my) for x, y in zip(left, right))
    vx = sum((x - mx) ** 2 for x in left)
    vy = sum((y - my) ** 2 for y in right)
    if vx <= 0 or vy <= 0:
        return 1.0 if left == right else 0.0
    return covariance / math.sqrt(vx * vy)


def dhash(signature: bytes) -> int:
    bits = 0
    for y in range(16):
        for x in range(16):
            bits = (bits << 1) | (signature[y * 4 * 64 + x * 4]
                                   > signature[y * 4 * 64 + (x + 1) * 4])
    return bits


def blocks_ssim(left: Image.Image, right: Image.Image) -> float:
    """Mean 8x8-block SSIM on 64x64 gray images; diagnostic, not skimage SSIM."""
    a = left.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    b = right.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    x, y = a.tobytes(), b.tobytes()
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    values = []
    for top in range(0, 64, 8):
        for side in range(0, 64, 8):
            aa = [x[(top + j) * 64 + side + i] for j in range(8) for i in range(8)]
            bb = [y[(top + j) * 64 + side + i] for j in range(8) for i in range(8)]
            ma, mb = sum(aa) / 64, sum(bb) / 64
            va = sum((v - ma) ** 2 for v in aa) / 64
            vb = sum((v - mb) ** 2 for v in bb) / 64
            cov = sum((p - ma) * (q - mb) for p, q in zip(aa, bb)) / 64
            values.append(((2 * ma * mb + c1) * (2 * cov + c2)) /
                          ((ma * ma + mb * mb + c1) * (va + vb + c2)))
    return statistics.mean(values)


def hist_affinity(left: Image.Image, right: Image.Image) -> float:
    a = left.convert("L").resize((64, 64)).histogram()
    b = right.convert("L").resize((64, 64)).histogram()
    return sum(math.sqrt(x * y) for x, y in zip(a, b)) / 4096


def load_image(archive: zipfile.ZipFile, root: str, ref: str, variant="original"):
    split, name = ref.split("/", 1)
    folder = split if variant == "original" else split + "_resize"
    with Image.open(io.BytesIO(archive.read(f"{root}/{folder}/{name}"))) as image:
        image.load()
        return image.convert("RGB"), tuple(sorted(image.getexif().keys()))


def diagnostics(archive, root, left, right, variant_right="original"):
    a, meta_a = load_image(archive, root, left)
    b, meta_b = load_image(archive, root, right, variant_right)
    aa = a.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    bb = b.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
    edge_a = aa.filter(ImageFilter.FIND_EDGES).tobytes()
    edge_b = bb.filter(ImageFilter.FIND_EDGES).tobytes()
    full = corr(a.convert("L").tobytes(), b.convert("L").tobytes()) if a.size == b.size else None
    return {"size_a": a.size, "size_b": b.size,
            "corr64": round(corr(aa.tobytes(), bb.tobytes()), 6),
            "full_corr": round(full, 6) if full is not None else None,
            "ssim_8x8_blocks": round(blocks_ssim(a, b), 6),
            "grayscale_histogram_affinity": round(hist_affinity(a, b), 6),
            "edge_corr64": round(corr(edge_a, edge_b), 6),
            "exif_keys_a": meta_a, "exif_keys_b": meta_b}


def family(name: str) -> str:
    return SUFFIX.sub("", Path(name).stem).casefold()


def key_pair(a, b):
    return tuple(sorted((a, b)))


class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a, b):
        x, y = self.find(a), self.find(b)
        if x != y:
            self.parent[max(x, y)] = min(x, y)

    def groups(self):
        data = defaultdict(list)
        for node in sorted(self.parent):
            data[self.find(node)].append(node)
        return sorted(data.values(), key=lambda x: x[0])


def partition(nodes, edges, named_threshold=.995, visual_dhash=12,
              visual_corr=.995, full_corr=.96, mode="current"):
    uf = UnionFind(nodes)
    for edge in edges:
        if edge["kind"] == "exact":
            merge = True
        elif edge["kind"] == "named":
            strong = edge["corr64"] >= named_threshold and edge["same_regions"]
            supported = edge["corr64"] >= named_threshold
            merge = strong if mode == "conservative" else supported
            if mode == "worst" and not merge:
                merge = True
        else:
            if mode == "conservative":
                merge = False
            else:
                considered = edge["dhash"] <= visual_dhash
                supported = (considered and edge["corr64"] >= visual_corr
                             and edge["full_corr"] is not None
                             and edge["full_corr"] >= full_corr)
                merge = supported or (mode == "worst" and considered)
        if merge:
            uf.union(edge["a"], edge["b"])
    return uf.groups()


def partition_stats(groups):
    return {"groups": len(groups), "size_distribution": dict(sorted(Counter(map(len, groups)).items())),
            "largest_group": max(map(len, groups)),
            "cross_official_split_groups": sum(len({ref.split("/")[0] for ref in group}) > 1 for group in groups)}


def build_edges(archive, root, images, annotations):
    edges = {}
    by_hash = defaultdict(list)
    by_family = defaultdict(list)
    refs = sorted(images)
    for ref in refs:
        by_hash[images[ref]["sha256"]].append(ref)
        by_family[images[ref]["family"]].append(ref)
    for digest, members in by_hash.items():
        if len(members) > 1:
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    edges[key_pair(a, b)] = {"a": min(a, b), "b": max(a, b), "kind": "exact",
                                             "sha256": digest, "same_regions": annotations[a] == annotations[b]}
    for fam, members in by_family.items():
        for i, a in enumerate(members):
            for b in members[i + 1:]:
                pair = key_pair(a, b)
                if pair in edges:
                    continue
                images_a, images_b = images[a], images[b]
                edges[pair] = {"a": pair[0], "b": pair[1], "kind": "named", "family": fam,
                               "corr64": corr(images_a["signature"], images_b["signature"]),
                               "dhash": (images_a["dhash"] ^ images_b["dhash"]).bit_count(),
                               "same_regions": annotations[a] == annotations[b],
                               "same_geometry": [r.get("shape_attributes") for r in annotations[a]]
                                                == [r.get("shape_attributes") for r in annotations[b]]}
    # Only cross-split, differently named-family pairs can enter the dHash shortlist.
    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            left = [r for r in refs if r.startswith(split_a + "/")]
            right = [r for r in refs if r.startswith(split_b + "/")]
            for a in left:
                for b in right:
                    if images[a]["family"] == images[b]["family"]:
                        continue
                    distance = (images[a]["dhash"] ^ images[b]["dhash"]).bit_count()
                    if distance > 14 or images[a]["sha256"] == images[b]["sha256"]:
                        continue
                    pair = key_pair(a, b)
                    if pair in edges:
                        continue
                    c64 = corr(images[a]["signature"], images[b]["signature"])
                    full = None
                    if images[a]["size"] == images[b]["size"]:
                        ia, _ = load_image(archive, root, a)
                        ib, _ = load_image(archive, root, b)
                        full = corr(ia.convert("L").tobytes(), ib.convert("L").tobytes())
                    edges[pair] = {"a": pair[0], "b": pair[1], "kind": "visual",
                                   "dhash": distance, "corr64": c64, "full_corr": full,
                                   "same_regions": annotations[a] == annotations[b],
                                   "same_geometry": [r.get("shape_attributes") for r in annotations[a]]
                                                    == [r.get("shape_attributes") for r in annotations[b]]}
    return list(edges.values())


def current_tier(edge):
    if edge["kind"] == "exact":
        return "Confirmed"
    if edge["kind"] == "named":
        if edge["corr64"] >= .995:
            return "Strongly Supported" if edge["same_regions"] else "Supported"
        return "Candidate"
    if edge["dhash"] > 12:
        return "Outside current shortlist"
    if edge["corr64"] >= .995 and edge["full_corr"] is not None and edge["full_corr"] >= .96:
        return "Supported"
    return "Candidate"


def qstats(values):
    values = sorted(values)
    if not values:
        return None
    at95 = values[math.ceil(.95 * len(values)) - 1]
    return {"minimum": round(values[0], 6), "median": round(statistics.median(values), 6),
            "mean": round(statistics.mean(values), 6), "p95": round(at95, 6),
            "maximum": round(values[-1], 6)}


def load_simulator():
    source = Path(__file__).with_name("07_simulate_group_split.py")
    spec = importlib.util.spec_from_file_location("d2_split_simulator", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_families(images, annotations, edges):
    named = defaultdict(list)
    for edge in edges:
        if edge["kind"] == "named" and edge["a"].split("/")[0] != edge["b"].split("/")[0]:
            named[edge["family"]].append(edge)
    result = []
    for fam, pairs in sorted(named.items()):
        members = sorted(ref for ref in images if images[ref]["family"] == fam)
        strongest = max(pairs, key=lambda edge: RANK[current_tier(edge)])
        tier = current_tier(strongest)
        corrs = [edge["corr64"] for edge in pairs]
        distances = [edge["dhash"] for edge in pairs]
        result.append({"family_id": fam, "members": "|".join(members),
                       "official_splits": "|".join(sorted({r.split("/")[0] for r in members})),
                       "filenames": "|".join(r.split("/", 1)[1] for r in members),
                       "image_dimensions": "|".join(f"{r}={images[r]['size'][0]}x{images[r]['size'][1]}" for r in members),
                       "class_labels": "|".join(sorted({label for r in members for label in images[r]["labels"]})),
                       "annotation_similarity": f"identical region lists {sum(e['same_regions'] for e in pairs)}/{len(pairs)}; identical geometry {sum(e['same_geometry'] for e in pairs)}/{len(pairs)}",
                       "pixel_similarity": f"64x64 Pearson min={min(corrs):.6f}; max={max(corrs):.6f}",
                       "dhash": f"256-bit distance min={min(distances)}; max={max(distances)}",
                       "suffix_relation": "|".join(sorted({r.split("/", 1)[1].rsplit(".", 1)[0].rsplit("_", 1)[-1]
                                                       for r in members if SUFFIX.search(Path(r).stem)})) or "no explicit suffix",
                       "final_confidence": tier,
                       "reason": ("best cross-split pair has 64x64 correlation >=0.995 and identical region list; acquisition identity unverified"
                                  if tier == "Strongly Supported" else
                                  "suffix/geometry evidence exists but 64x64 correlation below 0.995; do not merge")})
    return result, named


def exact_duplicate_review(images, annotations, edges):
    result = []
    for edge in edges:
        if edge["kind"] != "exact" or edge["a"].split("/")[0] == edge["b"].split("/")[0]:
            continue
        a, b = edge["a"], edge["b"]
        ra, rb = annotations[a], annotations[b]
        labels_a = [r.get("region_attributes", {}).get("name") for r in ra]
        labels_b = [r.get("region_attributes", {}).get("name") for r in rb]
        shapes_a = [r.get("shape_attributes", {}) for r in ra]
        shapes_b = [r.get("shape_attributes", {}) for r in rb]
        ordered_equal = shapes_a == shapes_b
        unordered_equal = Counter(json.dumps(s, sort_keys=True) for s in shapes_a) == Counter(
            json.dumps(s, sort_keys=True) for s in shapes_b)
        result.append({"file_a": a, "file_b": b, "sha256": edge["sha256"],
                       "regions_a": len(ra), "regions_b": len(rb),
                       "class_labels_a": labels_a, "class_labels_b": labels_b,
                       "polygon_coordinates_a": shapes_a, "polygon_coordinates_b": shapes_b,
                       "class_disagreement": labels_a != labels_b,
                       "region_count_disagreement": len(ra) != len(rb),
                       "polygon_coordinates_disagree": not unordered_equal,
                       "region_order_only_difference": not ordered_equal and unordered_equal})
    return sorted(result, key=lambda row: (row["file_a"], row["file_b"]))


def scaled_polygon_residual(source_regions, target_regions, source_size, target_size):
    if len(source_regions) != len(target_regions):
        return None
    differences = []
    for source, target in zip(source_regions, target_regions):
        a = source.get("shape_attributes", {})
        b = target.get("shape_attributes", {})
        for axis, factor in (("all_points_x", target_size[0] / source_size[0]),
                             ("all_points_y", target_size[1] / source_size[1])):
            if axis not in a or axis not in b or len(a[axis]) != len(b[axis]):
                return None
            differences.extend(abs(float(x) * factor - float(y)) for x, y in zip(a[axis], b[axis]))
    if not differences:
        return None
    return {"mean_abs_px": round(statistics.mean(differences), 6),
            "max_abs_px": round(max(differences), 6), "coordinates": len(differences)}


def abnormal_resize_review(archive, root, images, annotations):
    result = []
    for ref in ("train/172_brightness.jpg", "val/172_noise.jpg", "train/327.jpg"):
        a, meta_a = load_image(archive, root, ref)
        b, meta_b = load_image(archive, root, ref, "resize")
        estimated = a.resize(b.size, Image.Resampling.LANCZOS)
        diff = ImageChops.difference(estimated, b)
        mae = sum(ImageStat.Stat(diff).mean) / 3
        relation = diagnostics(archive, root, ref, ref, "resize")
        split, name = ref.split("/", 1)
        resize_metadata = json.loads(archive.read(f"{root}/{split}_resize.json"))
        resize_regions = next(record["regions"] for record in resize_metadata.values()
                              if record["filename"] == name)
        if isinstance(resize_regions, dict):
            resize_regions = list(resize_regions.values())
        same_polygon = scaled_polygon_residual(annotations[ref], resize_regions, a.size, b.size)
        close_family = sorted(other for other in images if other != ref and images[other]["family"] == images[ref]["family"])
        alternatives = []
        for other in close_family:
            candidate, _ = load_image(archive, root, other)
            candidate_resized = candidate.resize(b.size, Image.Resampling.LANCZOS)
            candidate_corr = corr(candidate_resized.convert("L").resize((64, 64)).tobytes(),
                                  b.convert("L").resize((64, 64)).tobytes())
            alternatives.append({"file": other, "corr64_to_resize": round(candidate_corr, 6)})
            alternatives[-1]["scaled_polygon_residual"] = scaled_polygon_residual(
                annotations[other], resize_regions, candidate.size, b.size)
        result.append({"file": ref, "sha256_original": images[ref]["sha256"],
                       "sha256_resize": hashlib.sha256(archive.read(f"{root}/{ref.split('/')[0]}_resize/{ref.split('/',1)[1]}")).hexdigest(),
                       "original_dimensions": a.size, "resize_dimensions": b.size,
                       "lanczos_rgb_mae": round(mae, 6), "pixel_metrics": relation,
                       "region_count_original": len(annotations[ref]),
                       "same_name_scaled_polygon_residual": same_polygon,
                       "other_originals_same_filename_family": alternatives,
                       "exif_original": meta_a, "exif_resize": meta_b,
                       "status": "Unresolved"})
    return result


def candidate_visual_review(archive, root, edges, current_map):
    result = []
    for edge in edges:
        if edge["kind"] != "visual" or edge["dhash"] > 12 or current_tier(edge) != "Candidate":
            continue
        a, b = edge["a"], edge["b"]
        metrics = diagnostics(archive, root, a, b)
        if edge["same_regions"] and edge["corr64"] >= .99 and edge["full_corr"] is not None and edge["full_corr"] >= .95:
            category = "truly likely same source (unconfirmed)"
        elif metrics["edge_corr64"] >= .95 and edge["corr64"] >= .98:
            category = "orchard scene similarity or same source; insufficient evidence"
        elif metrics["grayscale_histogram_affinity"] >= .97:
            category = "similar appearance or scene; insufficient evidence"
        else:
            category = "insufficient evidence"
        result.append({"file_a": a, "file_b": b, "dhash_distance": edge["dhash"],
                       "corr64": round(edge["corr64"], 6),
                       "full_corr": round(edge["full_corr"], 6) if edge["full_corr"] is not None else "",
                       "ssim_8x8_blocks": metrics["ssim_8x8_blocks"],
                       "histogram_affinity": metrics["grayscale_histogram_affinity"],
                       "edge_corr64": metrics["edge_corr64"],
                       "same_region_list": edge["same_regions"],
                       "same_geometry": edge["same_geometry"],
                       "current_same_group": current_map[a] == current_map[b],
                       "evidence_category": category,
                       "remaining_question": "no capture/fruit/tree/session ID; visual resemblance cannot establish source identity"})
    return sorted(result, key=lambda row: (row["file_a"], row["file_b"]))


def threshold_sensitivity(nodes, edges, baseline):
    specs = [("named_corr64", x, {"named_threshold": x}) for x in (.993, .995, .997)]
    specs += [("dhash_distance", x, {"visual_dhash": x}) for x in (10, 12, 14)]
    specs += [("visual_corr64", x, {"visual_corr": x}) for x in (.993, .995, .997)]
    specs += [("visual_full_corr", x, {"full_corr": x}) for x in (.95, .96, .97)]
    result = []
    base_members = {frozenset(group) for group in baseline}
    for name, value, option in specs:
        groups = partition(nodes, edges, **option)
        data = partition_stats(groups)
        named_limit = option.get("named_threshold", .995)
        dhash_limit = option.get("visual_dhash", 12)
        visual_limit = option.get("visual_corr", .995)
        full_limit = option.get("full_corr", .96)
        candidates = sum(
            (edge["kind"] == "named" and edge["corr64"] < named_limit)
            or (edge["kind"] == "visual" and edge["dhash"] <= dhash_limit
                and not (edge["corr64"] >= visual_limit and edge["full_corr"] is not None
                         and edge["full_corr"] >= full_limit))
            for edge in edges)
        result.append({"parameter": name, "value": value, **data,
                       "candidate_relation_count": candidates,
                       "components_changed_vs_current": len({frozenset(g) for g in groups} ^ base_members)})
    return result


def build_simulation_groups(groups, images, annotations):
    data = {}
    ref_to_gid = {}
    for index, members in enumerate(groups, 1):
        gid = f"verify-{index:04d}"
        counts = Counter()
        multiclass = 0
        for ref in members:
            labels = [r.get("region_attributes", {}).get("name") for r in annotations[ref]]
            valid = [label for label in labels if label in CLASSES]
            counts.update(valid)
            multiclass += len(set(valid)) > 1
            ref_to_gid[ref] = gid
        data[gid] = {"images": len(members), "groups": 1,
                     "multi_class_images": multiclass, "classes": counts,
                     "members": members}
    return data, ref_to_gid


def simulation_stress(groups, images, annotations, edges, seeds,
                      hard_tiers=("Confirmed", "Strongly Supported", "Supported")):
    simulator = load_simulator()
    data, ref_to_gid = build_simulation_groups(groups, images, annotations)
    total_images = sum(v["images"] for v in data.values())
    total_groups = len(data)
    total_classes = Counter()
    total_multi = 0
    for value in data.values():
        total_classes.update(value["classes"])
        total_multi += value["multi_class_images"]
    results = {}
    for ratio, target in (("70/15/15", (.7, .15, .15)), ("70/10/20", (.7, .1, .2))):
        measures = defaultdict(list)
        for seed in range(seeds):
            assignments, counts = simulator.simulate(data, target, seed)
            # Independent recount from the actual group assignment, not simulator's counters.
            for split in SPLITS:
                members = [v for gid, v in data.items() if assignments[gid] == split]
                if counts[split]["images"] != sum(v["images"] for v in members):
                    raise ValueError("simulation image counts do not match independent recount")
                if counts[split]["groups"] != len(members):
                    raise ValueError("simulation group counts do not match independent recount")
                for label in CLASSES:
                    if counts[split][label] != sum(v["classes"][label] for v in members):
                        raise ValueError("simulation class counts do not match independent recount")
            measures["image_max_abs_pp"].append(max(abs(100 * (counts[s]["images"] / total_images - target[i]))
                                                   for i, s in enumerate(SPLITS)))
            measures["group_max_abs_pp"].append(max(abs(100 * (counts[s]["groups"] / total_groups - target[i]))
                                                   for i, s in enumerate(SPLITS)))
            measures["class_max_abs_pp"].append(max(abs(100 * (counts[s][c] / total_classes[c] - target[i]))
                                                   for i, s in enumerate(SPLITS) for c in CLASSES))
            measures["multiclass_max_abs_pp"].append(max(abs(100 * (counts[s]["multi_class_images"] / total_multi - target[i]))
                                                        for i, s in enumerate(SPLITS)))
            largest_by_split = [max((v["images"] for gid, v in data.items() if assignments[gid] == s), default=0)
                                for s in SPLITS]
            measures["largest_group_size_range"].append(max(largest_by_split) - min(largest_by_split))
            cuts = Counter()
            for edge in edges:
                if edge["kind"] == "visual" and edge["dhash"] > 12:
                    continue
                if assignments[ref_to_gid[edge["a"]]] != assignments[ref_to_gid[edge["b"]]]:
                    cuts[current_tier(edge)] += 1
            for tier in ("Confirmed", "Strongly Supported", "Supported", "Candidate"):
                measures[tier + "_crossings"].append(cuts[tier])
            if any(cuts[tier] for tier in hard_tiers):
                raise AssertionError("evidence-supported relation crossed a simulated split")
        joint = sum(image <= 2 and maturity <= 5 and multi <= 5
                    for image, maturity, multi in zip(
                        measures["image_max_abs_pp"], measures["class_max_abs_pp"],
                        measures["multiclass_max_abs_pp"]))
        results[ratio] = {"seeds": seeds,
                          "within_2pp_images_and_5pp_each_class_and_5pp_multiclass": joint,
                          "metrics": {name: qstats(vals) for name, vals in sorted(measures.items())}}
    return results


def resize_screen(archive, root, images):
    values = []
    for ref in sorted(images):
        source, _ = load_image(archive, root, ref)
        target, _ = load_image(archive, root, ref, "resize")
        resized = source.resize(target.size, Image.Resampling.LANCZOS)
        mae = sum(ImageStat.Stat(ImageChops.difference(resized, target)).mean) / 3
        target_signature = target.convert("L").resize((64, 64), Image.Resampling.BILINEAR).tobytes()
        values.append((ref, mae, corr(images[ref]["signature"], target_signature)))
    experiments = []
    for mae_limit in (18, 20, 22):
        experiments.append({"parameter": "resize_LANCZOS_RGB_MAE", "value": mae_limit,
                            "supported_pairs": sum(mae <= mae_limit and c >= .995 for _, mae, c in values),
                            "original_group_count_effect": 0})
    for corr_limit in (.993, .995, .997):
        experiments.append({"parameter": "resize_corr64", "value": corr_limit,
                            "supported_pairs": sum(mae <= 20 and c >= corr_limit for _, mae, c in values),
                            "original_group_count_effect": 0})
    return experiments, sorted(((ref, round(mae, 6), round(c, 6)) for ref, mae, c in values
                                if mae > 20 or c < .995), key=lambda x: x[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.seeds <= 500:
        parser.error("--seeds must be 1..500")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
    images = {}
    annotations = {}
    manifest_files = {(row["split"], row["variant"], row["filename"]): row
                      for row in rows(args.manifest_dir / "files_sha256.csv")}
    with zipfile.ZipFile(args.archive) as archive:
        root_names = {name.split("/")[0] for name in archive.namelist() if name}
        if len(root_names) != 1:
            raise ValueError("D2 ZIP must have one root directory")
        root = root_names.pop()
        for split in SPLITS:
            metadata = json.loads(archive.read(f"{root}/{split}.json"))
            for record in metadata.values():
                regions = record["regions"]
                annotations[f"{split}/{record['filename']}"] = list(regions.values()) if isinstance(regions, dict) else regions
        for info in archive.infolist():
            parts = info.filename.split("/")
            if len(parts) != 3 or parts[1] not in SPLITS or not parts[2].lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ref = f"{parts[1]}/{parts[2]}"
            payload = archive.read(info)
            with Image.open(io.BytesIO(payload)) as original:
                original.load()
                size = original.size
                signature = original.convert("L").resize((64, 64), Image.Resampling.BILINEAR).tobytes()
            labels = [r.get("region_attributes", {}).get("name") for r in annotations[ref]]
            images[ref] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": size,
                           "family": family(parts[2]), "signature": signature,
                           "dhash": dhash(signature), "labels": [x for x in labels if x]}
        if len(images) != 1406 or set(images) != set(annotations):
            raise ValueError(f"expected 1406 matching original JPG/JSON records, found {len(images)}")
        original_manifest_mismatches = []
        resize_manifest_mismatches = []
        for ref, record in images.items():
            split, name = ref.split("/", 1)
            old = manifest_files.get((split, "original", name))
            if (old is None or old["sha256"] != record["sha256"]
                    or (int(old["width"]), int(old["height"])) != record["size"]):
                original_manifest_mismatches.append(ref)
            resize_record = manifest_files.get((split, "resize", name))
            resize_payload = archive.read(f"{root}/{split}_resize/{name}")
            if resize_record is None or resize_record["sha256"] != hashlib.sha256(resize_payload).hexdigest():
                resize_manifest_mismatches.append(ref)
        edges = build_edges(archive, root, images, annotations)
        current = partition(images, edges)
        conservative = partition(images, edges, mode="conservative")
        worst = partition(images, edges, mode="worst")
        groups_by_id = defaultdict(list)
        for row in rows(args.manifest_dir / "source_groups.csv"):
            if row["variant"] == "original":
                groups_by_id[row["source_group_id"]].append(f"{row['official_split']}/{row['filename']}")
        manifest_members = {frozenset(members) for members in groups_by_id.values()}
        independent_members = {frozenset(members) for members in current}
        missing = [sorted(g) for g in sorted(independent_members - manifest_members, key=lambda x: sorted(x))]
        extra = [sorted(g) for g in sorted(manifest_members - independent_members, key=lambda x: sorted(x))]
        independent_ids = {ref: f"sg-{i:04d}" for i, members in enumerate(current, 1) for ref in members}
        existing_ids = {f"{row['official_split']}/{row['filename']}": row["source_group_id"]
                        for row in rows(args.manifest_dir / "source_groups.csv") if row["variant"] == "original"}
        id_disagreements = [ref for ref in sorted(images) if independent_ids[ref] != existing_ids.get(ref)]
        families, family_pairs = review_families(images, annotations, edges)
        fields = ("family_id", "members", "official_splits", "filenames", "image_dimensions",
                  "class_labels", "annotation_similarity", "pixel_similarity", "dhash",
                  "suffix_relation", "final_confidence", "reason")
        write_rows(args.out_dir / "family_review_67.csv", fields, families)
        weakest = []
        for row in families:
            if row["final_confidence"] != "Strongly Supported":
                continue
            pair_corrs = [e["corr64"] for e in family_pairs[row["family_id"]] if current_tier(e) == "Strongly Supported"]
            weakest.append({"family": row["family_id"], "minimum_strong_corr64": min(pair_corrs),
                            "all_cross_pairs": len(family_pairs[row["family_id"]]),
                            "identical_region_pairs": sum(e["same_regions"] for e in family_pairs[row["family_id"]]),
                            "members": row["members"]})
        weakest.sort(key=lambda row: (row["minimum_strong_corr64"], row["family"]))
        exact_review = exact_duplicate_review(images, annotations, edges)
        (args.out_dir / "exact_duplicate_annotations.json").write_text(
            json.dumps(exact_review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        candidate_families = []
        for row in families:
            if row["final_confidence"] != "Candidate":
                continue
            for pair in family_pairs[row["family_id"]]:
                candidate_families.append({"family": row["family_id"], "file_a": pair["a"],
                                           "file_b": pair["b"], "metrics": diagnostics(archive, root, pair["a"], pair["b"]),
                                           "same_regions": pair["same_regions"],
                                           "same_geometry": pair["same_geometry"]})
        candidate_visual = candidate_visual_review(archive, root, edges, independent_ids)
        write_rows(args.out_dir / "candidate_dhash_26.csv",
                   ("file_a", "file_b", "dhash_distance", "corr64", "full_corr",
                    "ssim_8x8_blocks", "histogram_affinity", "edge_corr64",
                    "same_region_list", "same_geometry", "current_same_group",
                    "evidence_category", "remaining_question"), candidate_visual)
        abnormal = abnormal_resize_review(archive, root, images, annotations)
        resize_sensitivity, resize_exceptions = resize_screen(archive, root, images)
    sensitivity = threshold_sensitivity(images, edges, current)
    write_rows(args.out_dir / "threshold_sensitivity.csv",
               ("parameter", "value", "groups", "size_distribution", "largest_group",
                "cross_official_split_groups", "candidate_relation_count",
                "components_changed_vs_current"), sensitivity)
    visual_bridge = []
    for edge in edges:
        if edge["kind"] != "visual" or current_tier(edge) != "Supported":
            continue
        without = [e for e in edges if e is not edge]
        remaining = partition(images, without)
        lookup = {ref: index for index, members in enumerate(remaining) for ref in members}
        if lookup[edge["a"]] != lookup[edge["b"]]:
            visual_bridge.append({"file_a": edge["a"], "file_b": edge["b"],
                                  "corr64": round(edge["corr64"], 6),
                                  "full_corr": round(edge["full_corr"], 6),
                                  "dhash_distance": edge["dhash"],
                                  "bridge": True})
    write_rows(args.out_dir / "visual_bridge_risks.csv",
               ("file_a", "file_b", "corr64", "full_corr", "dhash_distance", "bridge"), visual_bridge)
    current_sim = simulation_stress(current, images, annotations, edges, args.seeds)
    conservative_sim = simulation_stress(conservative, images, annotations, edges,
                                         min(100, args.seeds),
                                         hard_tiers=("Confirmed", "Strongly Supported"))
    worst_sim = simulation_stress(worst, images, annotations, edges, min(100, args.seeds))
    report = {
        "status": "Independent verification / Review Pending / Not Frozen",
        "archive_sha256": digest,
        "runtime": {"python": platform.python_version(), "pillow": Image.__version__},
        "stage2b3_file_manifest_check": {"original_sha_or_size_mismatches": original_manifest_mismatches,
                                         "resize_sha_mismatches": resize_manifest_mismatches},
        "recomputation": {"image_count": len(images), "annotation_image_count": len(annotations),
                          "current": partition_stats(current),
                          "manifest_group_count": len(groups_by_id),
                          "membership_exact_match": not missing and not extra,
                          "source_group_id_exact_match": not id_disagreements,
                          "independent_only_components": missing[:30],
                          "manifest_only_components": extra[:30],
                          "id_disagreements": id_disagreements[:30]},
        "scenarios": {"Conservative": partition_stats(conservative),
                      "Current": partition_stats(current),
                      "Worst-Case": partition_stats(worst)},
        "independent_relation_counts": dict(Counter(current_tier(e) for e in edges
                                                     if e["kind"] != "visual" or e["dhash"] <= 12)),
        "family_status_counts": dict(Counter(row["final_confidence"] for row in families)),
        "weakest_ten_strong_families": weakest[:10],
        "two_candidate_families": candidate_families,
        "cross_split_exact_duplicates": len(exact_review),
        "candidate_dhash_pairs": len(candidate_visual),
        "candidate_dhash_cross_group": sum(not row["current_same_group"] for row in candidate_visual),
        "candidate_dhash_categories": dict(Counter(row["evidence_category"] for row in candidate_visual)),
        "supported_visual_bridges": visual_bridge,
        "abnormal_resize_pairs": abnormal,
        "resize_threshold_sensitivity": resize_sensitivity,
        "resize_screen_exceptions": resize_exceptions,
        "split_stability_current_500x2": current_sim,
        "split_feasibility_conservative_100x2": conservative_sim,
        "split_feasibility_worst_100x2": worst_sim,
    }
    (args.out_dir / "verification_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"membership_exact_match": report["recomputation"]["membership_exact_match"],
                      "source_group_id_exact_match": report["recomputation"]["source_group_id_exact_match"],
                      "scenarios": report["scenarios"], "family_status_counts": report["family_status_counts"],
                      "candidate_dhash_pairs": report["candidate_dhash_pairs"],
                      "candidate_dhash_cross_group": report["candidate_dhash_cross_group"],
                      "supported_visual_bridges": len(visual_bridge)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
