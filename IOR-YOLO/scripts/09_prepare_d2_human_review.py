"""Build a local, read-only D2 human adjudication pack for 21 fixed cases.

The HTML embeds raw JPEG bytes and renders VIA polygons as separate SVG overlays.
It never extracts, edits or rewrites source images/JSON or source-group manifests.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import importlib.util
import io
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageDraw
except ImportError as exc:
    raise SystemExit("Pillow required: uv run --with pillow python scripts/09_prepare_d2_human_review.py ...") from exc


WEAK_FAMILIES = ("1770", "img_13700", "2040", "194", "img_1406",
                 "img_13650", "1180", "257", "1550", "img_1381")
ABNORMAL = ("train/172_brightness.jpg", "val/172_noise.jpg", "train/327.jpg")
CHOICES = ("Accept relation", "Reject relation", "Keep Candidate",
           "Exclude from evaluation", "Needs more evidence")
COLOR = {"immature apple": "#39d353", "semi-mature apple": "#ffd166",
         "mature apple": "#ff5d73", "Unknown": "#00d9ff"}
CORR_PATTERN = re.compile(r"64x64 luminance correlation=([0-9.]+)")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_verifier():
    source = Path(__file__).with_name("08_verify_d2_groups.py")
    spec = importlib.util.spec_from_file_location("d2_human_review_metrics", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def component_after_edge_removal(members: list[str], edges: list[dict],
                                 left: str, right: str) -> list[list[str]]:
    """Counterfactual within the CURRENT merged original-image graph."""
    allowed = set(members)
    adjacency = {ref: set() for ref in members}
    removed = frozenset((left, right))
    for edge in edges:
        if edge["variant_a"] != edge["variant_b"] or edge["variant_a"] != "original" or edge["merged"] != "true":
            continue
        a = f"{edge['split_a']}/{edge['file_a']}"
        b = f"{edge['split_b']}/{edge['file_b']}"
        if a not in allowed or b not in allowed or frozenset((a, b)) == removed:
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = set()
    components = []
    for start in sorted(members):
        if start in seen:
            continue
        todo = [start]
        block = []
        seen.add(start)
        while todo:
            node = todo.pop()
            block.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    todo.append(neighbor)
        components.append(sorted(block))
    return sorted(components, key=lambda block: block[0])


def region_label(region: dict) -> str:
    value = region.get("region_attributes", {}).get("name")
    return value if isinstance(value, str) and value.strip() else "Unknown"


def polygon_mask(regions: list[dict], size: tuple[int, int]) -> Image.Image:
    mask = Image.new("1", size, 0)
    drawing = ImageDraw.Draw(mask)
    for region in regions:
        shape = region.get("shape_attributes", {})
        xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
        if len(xs) >= 3 and len(xs) == len(ys):
            drawing.polygon(list(zip(xs, ys)), fill=1)
    return mask


def mask_iou(regions_a: list[dict], regions_b: list[dict], size: tuple[int, int]) -> float | None:
    a, b = polygon_mask(regions_a, size), polygon_mask(regions_b, size)
    intersection = sum(byte.bit_count() for byte in ImageChops.logical_and(a, b).tobytes())
    union = sum(byte.bit_count() for byte in ImageChops.logical_or(a, b).tobytes())
    return intersection / union if union else None


def polygon_svg(regions: list[dict], width: int, height: int, highlight_unknown=False) -> str:
    shapes = []
    for index, region in enumerate(regions):
        shape = region.get("shape_attributes", {})
        xs, ys = shape.get("all_points_x", []), shape.get("all_points_y", [])
        if len(xs) < 3 or len(xs) != len(ys):
            continue
        label = region_label(region)
        color = COLOR.get(label, "#ffffff")
        points = " ".join(f"{float(x):g},{float(y):g}" for x, y in zip(xs, ys))
        stroke_width = 3 if label == "Unknown" and highlight_unknown else 1.8
        dash = ' stroke-dasharray="6 4"' if label == "Unknown" else ""
        shapes.append(f'<polygon points="{html.escape(points, quote=True)}" fill="{color}" '
                      f'fill-opacity="0.12" stroke="{color}" stroke-width="{stroke_width}"{dash}/>' )
        x0, y0 = float(xs[0]), float(ys[0])
        text_label = f"#{index} {label}"
        shapes.append(f'<text x="{x0:g}" y="{max(12, y0-4):g}" fill="{color}" '
                      f'font-size="12" font-weight="700" paint-order="stroke" '
                      f'stroke="#07111d" stroke-width="3">{html.escape(text_label)}</text>')
    return f'<svg class="overlay" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-label="Polygon overlay">' + "".join(shapes) + "</svg>"


def suffix(name: str) -> str:
    match = re.search(r"_(brightness|noise|gaussian|hsv|gamma)$", Path(name).stem, re.I)
    return match.group(1).lower() if match else "none"


def render_html(cases: list[dict], path: Path) -> None:
    """Write a self-contained local page; images remain separate from SVG overlays."""
    escape = html.escape

    def panel_html(panel: dict) -> str:
        labels = ", ".join(panel["labels"]) or "none"
        overlay = panel["svg"] if panel["overlay"] else ""
        aspect = f"{panel['width']} / {panel['height']}"
        return (
            '<figure class="panel">'
            f'<figcaption>{escape(panel["title"])}</figcaption>'
            f'<div class="imagebox" style="aspect-ratio:{aspect}">'
            f'<img src="{panel["jpeg_uri"]}" alt="{escape(panel["title"], quote=True)}" loading="lazy">'
            f'{overlay}</div>'
            '<dl class="meta">'
            f'<dt>Official split</dt><dd>{escape(panel["split"])}</dd>'
            f'<dt>Filename</dt><dd>{escape(panel["filename"])}</dd>'
            f'<dt>Variant</dt><dd>{escape(panel["variant"])}</dd>'
            f'<dt>Dimensions</dt><dd>{panel["width"]}×{panel["height"]}</dd>'
            f'<dt>SHA256</dt><dd class="mono">{escape(panel["sha256"])}</dd>'
            f'<dt>Augmentation suffix</dt><dd>{escape(panel["suffix"])}</dd>'
            f'<dt>Classes</dt><dd>{escape(labels)}</dd>'
            f'<dt>Polygon count</dt><dd>{panel["polygon_count"]}</dd>'
            '</dl></figure>'
        )

    def metrics_html(case: dict) -> str:
        values = case["metrics"]
        names = {
            "corr64": "64×64 pixel correlation",
            "full_corr": "Full-resolution pixel correlation",
            "ssim_8x8_blocks": "8×8 block SSIM (64×64)",
            "dhash_distance": "dHash distance / 256 bits",
            "dhash_a": "dHash A (256-bit hex)",
            "dhash_b": "dHash B (256-bit hex)",
            "size_a": "Metric image A dimensions",
            "size_b": "Metric image B dimensions",
        }
        rows = []
        for key, label in names.items():
            if key in values:
                value = values[key]
                value_text = "Not comparable" if value is None else str(value)
                rows.append(f'<dt>{label}</dt><dd>{escape(value_text)}</dd>')
        if case.get("polygon_iou") is not None:
            rows.append(f'<dt>Polygon union-mask IoU</dt><dd>{case["polygon_iou"]:.4f}</dd>')
        a, b = case["panels"][0], case["panels"][1]
        rows.append(f'<dt>SHA256 identical (shown pair)</dt><dd>{"Yes" if a["sha256"] == b["sha256"] else "No"}</dd>')
        return '<dl class="metrics">' + "".join(rows) + '</dl>'

    navigation = "".join(f'<a href="#{case["id"]}">{case["id"]} {escape(case["title"])}</a>' for case in cases)
    sections = []
    for case in cases:
        pair = " ↔ ".join(case["primary_pair"])
        bridge = ""
        if "bridge" in case:
            side_a = ", ".join(case["bridge"]["group_a"])
            side_b = ", ".join(case["bridge"]["group_b"])
            bridge = (
                '<div class="bridge"><div><strong>Group A</strong><br>' + escape(side_a) + '</div>'
                '<div class="arrow">↕<br>bridge relation<br>↕</div>'
                '<div><strong>Group B</strong><br>' + escape(side_b) + '</div></div>'
            )
        sections.append(
            f'<section id="{case["id"]}" class="case">'
            f'<h2>{case["id"]} · {escape(case["title"])}</h2>'
            f'<p><span class="badge">{escape(case["kind"])}</span> '
            '<span class="pending">Pending Human Review</span></p>'
            f'<p><strong>Current confidence:</strong> {escape(case["confidence"])} &nbsp; '
            f'<strong>Relation type:</strong> {escape(case["relation_type"])}<br>'
            f'<strong>Primary pair:</strong> {escape(pair)}</p>'
            f'{bridge}<p class="impact"><strong>Source-group impact:</strong> {escape(case["impact"])}</p>'
            f'<p><strong>Evidence / weakest point:</strong> {escape(case["note"])}</p>'
            f'{metrics_html(case)}'
            '<div class="gallery">' + "".join(panel_html(panel) for panel in case["panels"]) + '</div>'
            '</section>'
        )
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Stage 2B-4H · D2 Human Review</title>'
        '<style>'
        ':root{font-family:system-ui,-apple-system,sans-serif;color:#e9f0f7;background:#09121d}'
        '*{box-sizing:border-box}body{margin:0}header{padding:28px max(22px,4vw);background:#142236}'
        'h1{margin:0 0 8px;font-size:1.8rem}h2{margin:0 0 10px}p{line-height:1.5}'
        'header p{max-width:85ch;color:#b9c9db}button{background:#2e6eaa;color:white;border:0;border-radius:6px;padding:9px 14px;cursor:pointer}'
        'nav{display:flex;gap:8px;overflow:auto;padding:12px max(22px,4vw);background:#101b2a;position:sticky;top:0;z-index:2}'
        'nav a{white-space:nowrap;background:#253a51;color:#e8f2fa;padding:7px 10px;border-radius:5px;text-decoration:none;font-size:.84rem}'
        'main{max-width:1550px;margin:auto;padding:20px max(16px,3vw)}'
        '.case{scroll-margin-top:72px;background:#142236;border:1px solid #35506a;border-radius:10px;padding:22px;margin:0 0 28px}'
        '.badge,.pending{display:inline-block;border-radius:4px;padding:4px 8px;font-size:.82rem}.badge{background:#265273}.pending{background:#715220;color:#fff3d8}'
        '.impact{background:#213850;border-left:4px solid #6fc1ed;padding:12px}.bridge{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center;background:#20364a;padding:14px;border-radius:7px}.arrow{text-align:center;color:#ffd166}'
        '.metrics,.meta{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:5px 12px;font-size:.82rem}.metrics{margin:14px 0 20px}.metrics dt,.meta dt{color:#a3bad0}.metrics dd,.meta dd{margin:0;overflow-wrap:anywhere}'
        '.gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,350px),1fr));gap:18px}'
        '.panel{margin:0;padding:12px;background:#0b1827;border:1px solid #3a556e;border-radius:8px;min-width:0}'
        'figcaption{font-weight:700;margin:0 0 9px;overflow-wrap:anywhere}.imagebox{position:relative;background:#060d14;overflow:hidden}.imagebox img{width:100%;height:100%;display:block}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.hide-overlays .overlay{display:none}'
        '.mono{font-family:ui-monospace,SFMono-Regular,monospace;font-size:.73rem}'
        '@media(max-width:680px){.bridge{grid-template-columns:1fr}.arrow{transform:rotate(90deg)}}'
        '</style></head><body>'
        '<header><h1>D2 · Stage 2B-4H Human Review</h1>'
        '<p>21 fixed cases. Raw JPEG bytes are embedded only in this local HTML; colored polygon outlines are separate SVG overlays. '
        'Green: immature; yellow: semi-mature; red: mature; cyan dashed: Unknown. No source-group decision has been changed.</p>'
        '<button type="button" onclick="document.body.classList.toggle(\'hide-overlays\')">Toggle polygon overlays</button>'
        '</header><nav aria-label="Cases">' + navigation + '</nav><main>' + "".join(sections) + '</main></body></html>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def render_markdown(cases: list[dict], path: Path) -> None:
    lines = [
        '# Stage 2B-4H 人工复核表', '',
        '本表仅记录待人工裁决的 21 个指定案例。自动证据及图像见 '
        '[human_review.html](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html)。'
        '所有决策均为 Pending Human Review；本轮未调整 source groups、confidence 或 split。', '',
        '| Case | 对象 | 当前状态 | Source-group 影响 |',
        '|---|---|---|---|',
    ]
    for case in cases:
        lines.append(f'| [{case["id"]}](../IOR-YOLO/reports/dataset_audit/human_review/human_review.html#{case["id"]}) '
                     f'| {case["title"]} | {case["confidence"]} | {case["impact"]} |')
    lines.extend(['', '以下选项均未勾选。人工复核时记录决定和依据，不直接编辑 Raw Data。', ''])
    for case in cases:
        lines.extend([
            f'## {case["id"]} — {case["title"]}', '',
            f'- 类型：{case["kind"]}；当前 confidence：{case["confidence"]}；relation type：{case["relation_type"]}。',
            f'- 关键文件：{case["primary_pair"][0]} ↔ {case["primary_pair"][1]}。',
            f'- Source-group 影响：{case["impact"]}',
            f'- 证据与薄弱点：{case["note"]}',
            '- Human Decision: **Pending Human Review**',
        ])
        lines.extend(f'- [ ] {choice}' for choice in CHOICES)
        lines.extend(['- Reason: _Pending Human Review_', ''])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--verification-dir", required=True, type=Path)
    parser.add_argument("--output-html", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    source_rows = csv_rows(args.manifest_dir / "source_groups.csv")
    file_rows = csv_rows(args.manifest_dir / "files_sha256.csv")
    source_sha_before = hashlib.sha256((args.manifest_dir / "source_groups.csv").read_bytes()).hexdigest()
    relation_rows = csv_rows(args.manifest_dir / "source_group_relations.csv")
    independent_families = {row["family_id"]: row for row in csv_rows(args.verification_dir / "family_review_67.csv")}
    candidate_families = {row["source_family"]: row for row in csv_rows(args.manifest_dir / "source_family_review.csv")}
    bridge_rows = csv_rows(args.verification_dir / "visual_bridge_risks.csv")
    exact_rows = json.loads((args.verification_dir / "exact_duplicate_annotations.json").read_text(encoding="utf-8"))
    verification = json.loads((args.verification_dir / "verification_summary.json").read_text(encoding="utf-8"))
    verifier = load_verifier()
    by_ref = {(row["official_split"], row["filename"], row["variant"]): row for row in source_rows}
    file_by_ref = {(row["split"], row["filename"], row["variant"]): row for row in file_rows}
    groups = defaultdict(list)
    for row in source_rows:
        if row["variant"] == "original":
            groups[row["source_group_id"]].append(f"{row['official_split']}/{row['filename']}")
    group_count = len(groups)
    annotations = {}
    cases = []
    with zipfile.ZipFile(args.archive) as archive:
        roots = {name.split("/")[0] for name in archive.namelist() if name}
        if len(roots) != 1:
            raise ValueError("expected one ZIP root")
        root = roots.pop()
        for split in ("train", "val", "test"):
            for variant in ("original", "resize"):
                folder = split if variant == "original" else split + "_resize"
                records = json.loads(archive.read(f"{root}/{folder}.json"))
                for record in records.values():
                    regions = record["regions"]
                    annotations[(split, record["filename"], variant)] = list(regions.values()) if isinstance(regions, dict) else regions

        def make_panel(ref: str, variant="original", overlay=True, background_ref=None,
                       title=None, highlight_unknown=False):
            split, filename = ref.split("/", 1)
            record = by_ref[(split, filename, variant)]
            file_record = file_by_ref[(split, filename, variant)]
            image_ref = background_ref or ref
            image_split, image_filename = image_ref.split("/", 1)
            image_folder = image_split if variant == "original" else image_split + "_resize"
            payload = archive.read(f"{root}/{image_folder}/{image_filename}")
            width, height = int(record["width"]), int(record["height"])
            regions = annotations[(split, filename, variant)]
            if background_ref and background_ref != ref:
                if file_by_ref[(image_split, image_filename, variant)]["sha256"] != file_record["sha256"]:
                    raise ValueError("exact duplicate panels must use byte-identical backgrounds")
            return {"ref": ref, "variant": variant, "title": title or f"{ref} [{variant}]",
                    "split": split, "filename": filename, "width": width, "height": height,
                    "sha256": file_record["sha256"], "suffix": suffix(filename),
                    "labels": [region_label(region) for region in regions],
                    "polygon_count": len(regions), "overlay": overlay,
                    "svg": polygon_svg(regions, width, height, highlight_unknown),
                    "jpeg_uri": "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")}

        def gid(ref: str) -> str:
            split, filename = ref.split("/", 1)
            return by_ref[(split, filename, "original")]["source_group_id"]

        def pair_metrics(a: str, b: str, b_variant="original") -> dict:
            metrics = verifier.diagnostics(archive, root, a, b, b_variant)
            image_a, _ = verifier.load_image(archive, root, a)
            image_b, _ = verifier.load_image(archive, root, b, b_variant)
            sa = image_a.convert("L").resize((64, 64), Image.Resampling.BILINEAR).tobytes()
            sb = image_b.convert("L").resize((64, 64), Image.Resampling.BILINEAR).tobytes()
            hash_a, hash_b = verifier.dhash(sa), verifier.dhash(sb)
            metrics["dhash_distance"] = (hash_a ^ hash_b).bit_count()
            metrics["dhash_a"] = f"{hash_a:064x}"
            metrics["dhash_b"] = f"{hash_b:064x}"
            return metrics

        # C01-C02: named Candidates, no automatic promotion.
        for number, family in enumerate(("img_13960", "img_14040"), 1):
            members = candidate_families[family]["original_members"].split("|")
            if len(members) != 2 or len({gid(ref) for ref in members}) != 2:
                raise ValueError(f"unexpected Candidate family structure: {family}")
            a, b = members
            cases.append({"id": f"C{number:02d}", "kind": "Candidate family", "title": family,
                          "panels": [make_panel(a), make_panel(b)], "primary_pair": (a, b),
                          "confidence": "Candidate", "relation_type": "same_annotation_geometry / filename_family",
                          "metrics": pair_metrics(a, b),
                          "impact": f"Current: two separate source groups {gid(a)} and {gid(b)} (1+1 original images). Accept relation: one 2-image group, total group count {group_count}→{group_count-1}. Reject/keep Candidate: current structure unchanged.",
                          "note": "Same polygon list and augmentation-like suffix, but no verified capture ID; do not upgrade automatically."})

        # B01-B02: the two actual Supported bridge edges.
        for number, bridge in enumerate(bridge_rows, 1):
            a, b = bridge["file_a"], bridge["file_b"]
            group_id = gid(a)
            if gid(b) != group_id:
                raise ValueError("bridge endpoints are not in the same current source group")
            components = component_after_edge_removal(groups[group_id], relation_rows, a, b)
            if len(components) != 2:
                raise ValueError(f"reported bridge did not split exactly two components: {a} {b}")
            side_a = next(block for block in components if a in block)
            side_b = next(block for block in components if b in block)
            ordered = side_a + side_b
            cases.append({"id": f"B{number:02d}", "kind": "Supported bridge", "title": f"{a} ↔ {b}",
                          "panels": [make_panel(ref, title=("Group A: " if ref in side_a else "Group B: ") + ref)
                                     for ref in ordered], "primary_pair": (a, b),
                          "confidence": "Supported", "relation_type": "high_visual_similarity",
                          "metrics": pair_metrics(a, b),
                          "bridge": {"group_a": side_a, "group_b": side_b},
                          "impact": f"Accept: {len(ordered)} original images remain one group {group_id}. Reject: split into Group A ({len(side_a)}: {', '.join(side_a)}) and Group B ({len(side_b)}: {', '.join(side_b)}); total group count {group_count}→{group_count+1}.",
                          "note": "This edge is a graph bridge: removing it disconnects the current component. Human review must assess source identity, not just visual similarity."})

        # S01-S10: weakest crossing relation in each Strongly Supported family.
        for number, family in enumerate(WEAK_FAMILIES, 1):
            family_row = independent_families[family]
            members = family_row["members"].split("|")
            eligible = []
            for edge in relation_rows:
                if edge["variant_a"] != edge["variant_b"] or edge["variant_a"] != "original":
                    continue
                a, b = f"{edge['split_a']}/{edge['file_a']}", f"{edge['split_b']}/{edge['file_b']}"
                if a not in members or b not in members or edge["split_a"] == edge["split_b"] or edge["confidence"] != "Strongly Supported":
                    continue
                match = CORR_PATTERN.search(edge["evidence"])
                if match:
                    eligible.append((float(match.group(1)), a, b))
            if not eligible:
                raise ValueError(f"no Strongly Supported crossing pair: {family}")
            weakness, a, b = min(eligible)
            group_id = gid(a)
            if gid(b) != group_id:
                raise ValueError(f"weak-family pair not grouped: {family}")
            after = component_after_edge_removal(groups[group_id], relation_rows, a, b)
            impact = (f"Current: {len(groups[group_id])} original images in {group_id}. Reject this weakest edge: "
                      + (f"component splits into sizes {'+'.join(str(len(x)) for x in after)}; group count +{len(after)-1}."
                         if len(after) > 1 else "no source-group change because another supported path remains."))
            cases.append({"id": f"S{number:02d}", "kind": "Weakest Strongly Supported", "title": family,
                          "panels": [make_panel(ref, title=("Weakest edge: " if ref in (a, b) else "Other family member: ") + ref)
                                     for ref in members], "primary_pair": (a, b),
                          "confidence": "Strongly Supported", "relation_type": "augmentation_derivative",
                          "metrics": pair_metrics(a, b), "impact": impact,
                          "note": f"Current rule: terminal augmentation family + identical region list + 64×64 correlation {weakness:.6f} ≥ 0.995. Weakest evidence: correlation near the threshold; physical capture identity remains unverified."})

        # D01-D03: identical bytes, alternate polygon overlays on the same background.
        for number, pair in enumerate(exact_rows, 1):
            a, b = pair["file_a"], pair["file_b"]
            group_id = gid(a)
            after = component_after_edge_removal(groups[group_id], relation_rows, a, b)
            size = (int(by_ref[(*a.split("/", 1), "original")]["width"]),
                    int(by_ref[(*a.split("/", 1), "original")]["height"]))
            overlap = mask_iou(annotations[(*a.split("/", 1), "original")],
                               annotations[(*b.split("/", 1), "original")], size)
            cases.append({"id": f"D{number:02d}", "kind": "Exact SHA duplicate / annotation conflict",
                          "title": f"{a} ↔ {b}",
                          "panels": [make_panel(a, title="Annotation A on identical background"),
                                     make_panel(b, background_ref=a, title="Annotation B on identical background")],
                          "primary_pair": (a, b), "confidence": "Confirmed image duplicate; annotation conflict Unresolved",
                          "relation_type": "exact_duplicate", "metrics": pair_metrics(a, b),
                          "polygon_iou": overlap,
                          "impact": f"Identical bytes require the same source group {group_id} ({len(groups[group_id])} original images). Counterfactually removing only this edge yields {len(after)} component(s) with sizes {'+'.join(str(len(x)) for x in after)}; annotation choice does not change the duplicate fact.",
                          "note": f"Region counts: {pair['regions_a']} vs {pair['regions_b']}; class lists equal={not pair['class_disagreement']}; polygon coordinates differ={pair['polygon_coordinates_disagree']}; union-mask IoU={overlap:.4f} (rasterized, not instance-matched). Cause of reannotation unknown."})

        # R01-R03: original and same-name resize exceptions.
        for number, ref in enumerate(ABNORMAL, 1):
            item = next(row for row in verification["abnormal_resize_pairs"] if row["file"] == ref)
            cases.append({"id": f"R{number:02d}", "kind": "Abnormal original ↔ resize", "title": ref,
                          "panels": [make_panel(ref, "original"), make_panel(ref, "resize")],
                          "primary_pair": (ref, ref), "confidence": "Unresolved",
                          "relation_type": "unresolved_candidate / same-name resize companion",
                          "metrics": pair_metrics(ref, ref, "resize"),
                          "impact": f"Both files are provisionally attached to source group {gid(ref)} by filename. Accept/reject pixel equivalence does not change the {group_count} original-image groups; it changes whether this resize representation may be used.",
                          "note": f"LANCZOS RGB MAE={item['lanczos_rgb_mae']}; scaled polygon mean/max residual={item['same_name_scaled_polygon_residual']['mean_abs_px']}/{item['same_name_scaled_polygon_residual']['max_abs_px']} px. Pairing/processing cause remains unresolved."})

        # U01: raw image beside the same image with every polygon; Unknown highlighted.
        unknown_ref = "test/IMG_54350.jpg"
        unlabeled = [index for index, region in enumerate(annotations[("test", "IMG_54350.jpg", "original")])
                     if region_label(region) == "Unknown"]
        if unlabeled != [3]:
            raise ValueError(f"unexpected Unknown region indices: {unlabeled}")
        cases.append({"id": "U01", "kind": "Unlabeled polygon", "title": "test/IMG_54350.jpg #3",
                      "panels": [make_panel(unknown_ref, overlay=False, title="Raw image"),
                                 make_panel(unknown_ref, overlay=True, title="All polygons; #3 Unknown highlighted", highlight_unknown=True)],
                      "primary_pair": (unknown_ref, unknown_ref), "confidence": "Unlabeled / Unknown",
                      "relation_type": "annotation integrity issue", "metrics": {},
                      "impact": f"No source-group change ({gid(unknown_ref)}). Human decision affects derived annotation and whether this image may enter scored evaluation.",
                      "note": "Four regions: #0 mature, #1 semi-mature, #2 mature, #3 Unknown. Polygon #3 has geometry but no class attribute; do not infer its maturity stage."})

        if len(cases) != 21:
            raise AssertionError(f"expected 21 cases, found {len(cases)}")
        render_html(cases, args.output_html)
        render_markdown(cases, args.output_md)

    source_sha_after = hashlib.sha256((args.manifest_dir / "source_groups.csv").read_bytes()).hexdigest()
    if source_sha_before != source_sha_after:
        raise AssertionError("source_groups.csv changed during human-review generation")
    print(json.dumps({"cases": len(cases), "html": str(args.output_html),
                      "markdown": str(args.output_md), "source_groups_unchanged": True,
                      "html_bytes": args.output_html.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
