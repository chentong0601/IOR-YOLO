"""E01 result parser and instance-error analysis; never trains or evaluates models.

Default is validation only. Test artifacts require --split test --final-test and
a run manifest that has already recorded the one authorized final test.

Two confidence levels are deliberately kept apart:

* ``predictions/predictions_<split>.csv`` is exported by script 18 as the fixed
  AP-consistent ``conf=0.001`` stream. It is read here in full and never
  rewritten, so the raw prediction stream stays available for AP-style work.
* Operational error analysis (matching, false positives, missed detections,
  maturity confusion, failure-case selection) uses only predictions at or above
  the frozen operating threshold ``OPERATING_CONFIDENCE_THRESHOLD`` (0.65),
  selected on the validation F1 curve before any final test. Formal AP/PR
  metrics are read from the run manifest and are never recomputed here.
  A validation re-run at any other threshold is recorded as sensitivity-only
  (non-official) and can never be reported as the frozen operating point.

Raw-source provenance accepts both the frozen local ZIP and the byte-verified
unpacked D2 directory used by the Kaggle formal run. All analysis output goes to
a versioned analysis directory so earlier (exploratory) analysis is preserved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import warnings
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = Path(os.environ.get("E01_RUNS_ROOT", ROOT / "runs")).expanduser().resolve()
DEFAULT_RUN = RUNS_ROOT / "e01_yolo11n_seg/seed_0"
DEFAULT_DATA = ROOT / "data/processed/d2_e01_ultralytics"
CLASSES = ("immature apple", "semi-mature apple", "mature apple")
PRED_FIELDS = ("image_id", "pred_instance_id", "pred_class", "confidence", "box",
               "mask_reference", "source_group_id", "split_guard_cluster_id", "split")
MATCH_FIELDS = ("image_id", "gt_instance_id", "gt_class", "pred_instance_id", "pred_class",
                "confidence", "box_iou", "mask_iou", "matched", "split", "source_group_id",
                "split_guard_cluster_id", "failure_type", "human_review")
IOU_THRESHOLD = 0.5

# Analysis-layer operating point. The exporter keeps the complete Ultralytics
# AP-consistent stream (conf=0.001); those 1000+ low-confidence predictions are
# not detections at the deployed operating point, so counting all of them as
# false positives is meaningless. 0.65 comes from the validation F1 curve
# (optimum ~0.652) and was frozen before any final test access.
ANALYSIS_VERSION = 2
PREDICTION_EXPORT_CONF = 0.001
OPERATING_CONFIDENCE_THRESHOLD = 0.65
THRESHOLD_SELECTION_SPLIT = "val"
THRESHOLD_SELECTION_BASIS = "validation F1 operating point; frozen before final test"

# Raw-source evidence locations (relative to the repository root).
SOURCE_ZIP_RELATIVE = Path("data/raw/multistage_apple_v4/dataset-20260508.zip")
UNPACKED_IDENTITY_RELATIVE = Path("configs/data/d2_unpacked_identity.json")
FILES_SHA256_RELATIVE = Path("data/manifests/files_sha256.csv")
FROZEN_PROTOCOL_RELATIVE = Path("configs/data/d2_frozen_protocol.yaml")
VERIFIED_UNPACKED_KIND = "unpacked_directory"
VERIFIED_IDENTITY_STATUS = "VERIFIED"
UNPACKED_IDENTITY_SCHEMA = "d2-unpacked-identity-v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_write(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checked_split(split: str, final_test: bool, manifest: dict) -> None:
    if split not in ("val", "test"):
        raise ValueError("analysis is restricted to val or final test")
    if split == "test":
        if not final_test or manifest.get("status") != "final_test_completed" or not manifest.get("test_metrics_mask"):
            raise PermissionError("FINAL TEST EVALUATION: require --final-test and completed final test manifest")
        print("FINAL TEST EVALUATION — read-only post-final-test analysis; no model selection")
    elif final_test:
        raise ValueError("--final-test only applies to --split test")


def raw_source_provenance(manifest: dict, *, repo_root: Path = ROOT) -> tuple[list[str], dict]:
    """Verify how the formal run obtained its raw D2 images.

    Two modes are accepted and neither is weaker than the other:

    * verified unpacked directory (``raw_source_kind == "unpacked_directory"``
      with ``raw_identity_status == "VERIFIED"``): the formal run consumed the
      byte-verified unpacked D2 directory, so no local ``dataset-20260508.zip``
      is expected. The manifest must instead agree with the tracked frozen
      protocol *and* the tracked unpacked identity sidecar on the expected
      source ZIP sha256, the ``files_sha256.csv`` evidence hash and the identity
      evidence hash. A locally present archive is still hashed and compared.
    * original ZIP: the manifest pins ``dataset_zip_sha256`` and the frozen
      archive must exist under ``data/raw/multistage_apple_v4/`` with exactly
      that digest (previous behaviour, unchanged).

    Only the reason for trusting the raw source changes; every genuine mismatch
    or missing piece of evidence is still reported as a provenance issue.
    """
    zip_path = repo_root / SOURCE_ZIP_RELATIVE
    identity_path = repo_root / UNPACKED_IDENTITY_RELATIVE
    files_manifest = repo_root / FILES_SHA256_RELATIVE
    protocol_path = repo_root / FROZEN_PROTOCOL_RELATIVE
    kind, status = manifest.get("raw_source_kind"), manifest.get("raw_identity_status")
    detail = {"raw_source_kind": kind, "raw_identity_status": status,
              "raw_source_path": manifest.get("raw_source_path"),
              "dataset_zip_sha256": manifest.get("dataset_zip_sha256"),
              "expected_source_zip_sha256": None,
              "raw_identity_evidence_sha256": manifest.get("raw_identity_evidence_sha256"),
              "files_sha256_manifest_sha256": manifest.get("files_sha256_manifest_sha256"),
              "local_source_zip_present": zip_path.is_file()}
    issues: list[str] = []
    protocol = None
    if protocol_path.is_file():
        try:
            protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            issues.append("frozen protocol YAML not readable for raw-source cross-check")
    if isinstance(protocol, dict):
        detail["expected_source_zip_sha256"] = protocol.get("zip_sha256")
        detail["expected_source_zip_filename"] = protocol.get("zip_filename")
    if kind == VERIFIED_UNPACKED_KIND and status == VERIFIED_IDENTITY_STATUS:
        detail["mode"] = "verified_unpacked_directory"
        if not manifest.get("raw_source_path"):
            issues.append("missing raw_source_path for verified unpacked directory")
        if not identity_path.is_file():
            issues.append(f"missing provenance file: {identity_path.name}")
        else:
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if identity.get("schema_version") != UNPACKED_IDENTITY_SCHEMA:
                issues.append("unexpected unpacked identity schema")
            if manifest.get("raw_identity_evidence_sha256") != digest(identity_path):
                issues.append("provenance hash mismatch: raw_identity_evidence_sha256")
            if identity.get("source_zip_sha256"):
                detail["expected_source_zip_sha256"] = identity["source_zip_sha256"]
            if not files_manifest.is_file():
                issues.append(f"missing provenance file: {files_manifest.name}")
            elif manifest.get("files_sha256_manifest_sha256") != digest(files_manifest):
                issues.append("provenance hash mismatch: files_sha256_manifest_sha256")
            elif identity.get("files_sha256_manifest_sha256") != digest(files_manifest):
                issues.append("unpacked identity and tracked image-hash evidence disagree")
            elif identity.get("expected_image_files") != len(csv_rows(files_manifest)):
                issues.append("image-hash evidence cardinality changed")
            if (identity.get("source_zip_filename") and isinstance(protocol, dict)
                    and identity["source_zip_filename"] != protocol.get("zip_filename")):
                issues.append("unpacked identity and frozen protocol disagree on source ZIP filename")
        expected = detail["expected_source_zip_sha256"]
        if expected is None:
            issues.append("cannot cross-check dataset_zip_sha256 against frozen source ZIP sha256")
        else:
            if manifest.get("dataset_zip_sha256") != expected:
                issues.append("provenance hash mismatch: dataset_zip_sha256 (frozen source ZIP)")
            if isinstance(protocol, dict) and protocol.get("zip_sha256") != expected:
                issues.append("frozen protocol and unpacked identity disagree on source ZIP sha256")
        if zip_path.is_file():
            detail["local_source_zip_sha256"] = digest(zip_path)
            if detail["local_source_zip_sha256"] != manifest.get("dataset_zip_sha256"):
                issues.append("provenance hash mismatch: dataset_zip_sha256 (local archive)")
        return issues, detail
    if kind == VERIFIED_UNPACKED_KIND:
        issues.append(f"unpacked raw source is not identity-verified: {status!r}")
    detail["mode"] = kind or "original_zip"
    if not zip_path.is_file():
        issues.append(f"missing provenance file: {zip_path.name}")
    elif manifest.get("dataset_zip_sha256") != digest(zip_path):
        issues.append("provenance hash mismatch: dataset_zip_sha256")
    return issues, detail


def provenance_report(manifest: dict, run_dir: Path, *, repo_root: Path = ROOT) -> tuple[list[str], dict]:
    """Full provenance report as ``(issues, detail)``; legacy issue wording kept."""
    required = ("run_id", "git_commit", "python", "pytorch", "ultralytics", "training_seed",
                "dataset_zip_sha256", "frozen_pool_sha256", "frozen_split_sha256",
                "protocol_yaml_sha256", "best_checkpoint_sha256", "resolved_config_sha256")
    issues = [f"missing {key}" for key in required if manifest.get(key) is None or manifest.get(key) == ""]
    if manifest.get("experiment_relevant_git_dirty") is not False:
        issues.append("E01-relevant Git paths not recorded clean")
    if manifest.get("ultralytics") != "8.3.220" or manifest.get("training_seed") != 0:
        issues.append("pinned Ultralytics/training seed disagreement")
    source_issues, source_detail = raw_source_provenance(manifest, repo_root=repo_root)
    issues.extend(source_issues)
    checks = {
        "frozen_pool_sha256": repo_root / "data/manifests/d2_experiment_pool_frozen.csv",
        "frozen_split_sha256": repo_root / "data/manifests/d2_split_frozen.csv",
        "protocol_yaml_sha256": repo_root / FROZEN_PROTOCOL_RELATIVE,
        "best_checkpoint_sha256": run_dir / "weights/best.pt",
        "resolved_config_sha256": run_dir / "resolved_train_config.yaml",
        "experiment_config_sha256": repo_root / "configs/experiments/e01_yolo11n_seg.yaml",
    }
    for field, path in checks.items():
        if not path.is_file():
            issues.append(f"missing provenance file: {path.name}")
        elif manifest.get(field) != digest(path):
            issues.append(f"provenance hash mismatch: {field}")
    for message in issues:
        warnings.warn(f"E01 provenance: {message}", UserWarning, stacklevel=2)
    return issues, {"status": "verified" if not issues else "incomplete",
                    "raw_source": source_detail,
                    "checked_files": {field: str(path) for field, path in checks.items()}}


def provenance(manifest: dict, run_dir: Path, *, repo_root: Path = ROOT) -> list[str]:
    """Backward-compatible issue list used by the exporter and other callers."""
    issues, _ = provenance_report(manifest, run_dir, repo_root=repo_root)
    return issues


def box_iou(a: list[float], b: list[float]) -> float:
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = aa + bb - intersection
    return intersection / union if union > 0 else 0.0


def bbox(points: list[list[float]], width: int, height: int) -> list[float]:
    xs, ys = [p[0]*width for p in points], [p[1]*height for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def mask_iou(a: list[list[float]], b: list[list[float]], width: int, height: int) -> float:
    masks = []
    for polygon in (a, b):
        mask = Image.new("1", (width, height), 0)
        ImageDraw.Draw(mask).polygon([(x*width, y*height) for x, y in polygon], fill=1)
        masks.append(mask)
    x, y = masks[0].tobytes(), masks[1].tobytes()
    intersection = sum((left & right).bit_count() for left, right in zip(x, y))
    union = sum((left | right).bit_count() for left, right in zip(x, y))
    return intersection/union if union else 0.0


def match_image(gt: list[dict], pred: list[dict], *, threshold: float = IOU_THRESHOLD) -> list[dict]:
    """Class-independent, one-to-one maximum-IoU greedy matching with stable ties."""
    pairs = []
    for i, target in enumerate(gt):
        for j, detection in enumerate(pred):
            iou = box_iou(target["box"], detection["box"])
            if iou >= threshold:
                pairs.append((-iou, -float(detection["confidence"]), str(target["gt_instance_id"]),
                              str(detection["pred_instance_id"]), i, j))
    used_gt, used_pred = set(), set()
    matches = []
    for negative_iou, _, _, _, i, j in sorted(pairs):
        if i not in used_gt and j not in used_pred:
            used_gt.add(i)
            used_pred.add(j)
            target, detection = gt[i], pred[j]
            m_iou = (mask_iou(target["polygon"], detection["polygon"], target["width"], target["height"])
                     if target.get("polygon") and detection.get("polygon") else None)
            matches.append({"kind": "matched", "gt": target, "pred": detection,
                            "box_iou": -negative_iou, "mask_iou": m_iou})
    matches.extend({"kind": "missed", "gt": gt[i], "pred": None, "box_iou": None, "mask_iou": None}
                   for i in range(len(gt)) if i not in used_gt)
    matches.extend({"kind": "false_positive", "gt": None, "pred": pred[j], "box_iou": None, "mask_iou": None}
                   for j in range(len(pred)) if j not in used_pred)
    return matches


def maturity_summary(matches: list[dict]) -> dict:
    matrix = [[0]*3 for _ in range(3)]
    misses = false_positives = 0
    differences = Counter()
    for item in matches:
        if item["kind"] == "missed":
            misses += 1
        elif item["kind"] == "false_positive":
            false_positives += 1
        else:
            gt, pred = int(item["gt"]["gt_class"]), int(item["pred"]["pred_class"])
            matrix[gt][pred] += 1
            differences[abs(gt-pred)] += 1
    n = sum(map(sum, matrix))
    by_class = {CLASSES[i]: {"matched_gt": sum(matrix[i]),
                             "correct": matrix[i][i],
                             "classification_accuracy_matched": matrix[i][i]/sum(matrix[i]) if sum(matrix[i]) else None}
                for i in range(3)}
    return {"gt_rows_pred_columns": matrix, "matched_instances": n,
            "missed_detections": misses, "false_positives": false_positives,
            "matched_gt_coverage": n/(n+misses) if n+misses else None,
            "MASE_stage_matched_only": sum(k*v for k,v in differences.items())/n if n else None,
            "adjacent_error_rate_matched_only": differences[1]/n if n else None,
            "severe_error_rate_matched_only": differences[2]/n if n else None,
            "off_by_one_accuracy_matched_only": (differences[0]+differences[1])/n if n else None,
            "correct": differences[0], "adjacent": differences[1], "severe": differences[2],
            "per_gt_class": by_class}


def failures(matches: list[dict]) -> list[dict]:
    rows = []
    for item in matches:
        target, detection = item["gt"], item["pred"]
        if item["kind"] == "missed":
            category = "F1 Missed Fruit"
        elif item["kind"] == "false_positive":
            category = "F2 False Positive"
        else:
            distance = abs(int(target["gt_class"])-int(detection["pred_class"]))
            category = ("F4 Severe Maturity Confusion" if distance == 2 else
                        "F3 Adjacent Maturity Confusion" if distance == 1 else
                        "F5 Poor Mask Localization" if item["mask_iou"] is not None and item["mask_iou"] < .5 else "")
        rows.append({"category": category, "image_id": (target or detection)["image_id"],
                     "confidence": detection["confidence"] if detection else None,
                     "mask_iou": item["mask_iou"], "gt_instance_id": target["gt_instance_id"] if target else None,
                     "pred_instance_id": detection["pred_instance_id"] if detection else None,
                     "human_review": "Needs Human Review" if category else "not flagged"})
    return rows


def select_cases(rows: list[dict], limit: int = 10) -> dict[str, list[dict]]:
    def chosen(category: str, key):
        return sorted((row for row in rows if row["category"] == category), key=key)[:limit]
    return {
        "top_false_positives": chosen("F2 False Positive", lambda r: (-float(r["confidence"]), r["image_id"], str(r["pred_instance_id"]))),
        "top_false_negatives": chosen("F1 Missed Fruit", lambda r: (r["image_id"], str(r["gt_instance_id"]))),
        "lowest_mask_iou": sorted((r for r in rows if r["mask_iou"] is not None),
                                  key=lambda r: (float(r["mask_iou"]), r["image_id"], str(r["gt_instance_id"])))[:limit],
        "highest_confidence_wrong_class": sorted((r for r in rows if r["category"] in ("F3 Adjacent Maturity Confusion", "F4 Severe Maturity Confusion")),
                                                 key=lambda r: (-float(r["confidence"]), r["image_id"]))[:limit],
        "severe_0_to_2": chosen("F4 Severe Maturity Confusion", lambda r: (r["image_id"], str(r["gt_instance_id"]))),
        "adjacent_errors": chosen("F3 Adjacent Maturity Confusion", lambda r: (r["image_id"], str(r["gt_instance_id"]))),
    }


def parse_training_csv(path: Path) -> dict:
    rows = csv_rows(path)
    if not rows:
        raise ValueError("no real training rows")
    last = {key.strip(): value for key, value in rows[-1].items()}
    keys = {"box": ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"),
            "mask": ("metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)")}
    result = {"epochs_logged": len(rows), "last_epoch": int(float(last["epoch"])),
              "last_epoch_validation": {}, "loss_curves": {}, "validation_metric_curves": {}}
    for kind, names in keys.items():
        result["last_epoch_validation"][kind] = {label: float(last[key]) for label, key in zip(("P", "R", "mAP50", "mAP50_95"), names)}
        for key in names:
            result["validation_metric_curves"][key] = [float({k.strip(): v for k,v in row.items()}[key]) for row in rows]
    for key in last:
        if "loss" in key:
            result["loss_curves"][key] = [float({k.strip(): v for k,v in row.items()}[key]) for row in rows]
    return result


def load_ground_truth(data_root: Path, split: str, image_id: str) -> list[dict]:
    image = data_root / "images" / split / image_id
    label = data_root / "labels" / split / (Path(image_id).stem + ".txt")
    with Image.open(image) as im:
        width, height = im.size
    result = []
    for i, line in enumerate(label.read_text(encoding="utf-8").splitlines()):
        vals = line.split()
        klass = int(vals[0])
        if klass not in (0,1,2) or len(vals) < 7 or len(vals)%2 != 1:
            raise ValueError(f"invalid GT segmentation line: {image_id}#{i}")
        coords = list(map(float, vals[1:]))
        poly = [[coords[j], coords[j+1]] for j in range(0, len(coords), 2)]
        result.append({"image_id": image_id, "gt_instance_id": i, "gt_class": klass,
                       "polygon": poly, "box": bbox(poly,width,height), "width": width, "height": height})
    return result


def read_predictions(run_dir: Path, split: str) -> dict[str, list[dict]]:
    path = run_dir / "predictions" / f"predictions_{split}.csv"
    records = csv_rows(path)
    with path.open(newline="", encoding="utf-8") as stream:
        fields = csv.DictReader(stream).fieldnames or []
    if not set(PRED_FIELDS).issubset(fields):
        raise ValueError("prediction schema incomplete")
    grouped = {}
    unique_ids = set()
    for row in records:
        if row["split"] != split or int(row["pred_class"]) not in (0,1,2):
            raise ValueError("prediction split/class mismatch")
        identity = (row["image_id"], row["pred_instance_id"])
        if identity in unique_ids:
            raise ValueError("duplicate prediction instance ID")
        unique_ids.add(identity)
        conf = float(row["confidence"])
        box = json.loads(row["box"])
        if not math.isfinite(conf) or not 0 <= conf <= 1 or len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
            raise ValueError("invalid prediction confidence/box")
        polygon = None
        if row["mask_reference"]:
            mask_path = (run_dir / row["mask_reference"]).resolve()
            if not mask_path.is_relative_to(run_dir.resolve()):
                raise ValueError("mask reference outside run directory")
            polygon = json.loads(mask_path.read_text(encoding="utf-8"))["polygon_normalized"]
        grouped.setdefault(row["image_id"], []).append({"image_id": row["image_id"],
            "pred_instance_id": row["pred_instance_id"], "pred_class": int(row["pred_class"]),
            "confidence": conf, "box": list(map(float,box)), "polygon": polygon,
            "source_group_id": row["source_group_id"],
            "split_guard_cluster_id": row["split_guard_cluster_id"]})
    return grouped


def threshold_label(threshold: float) -> str:
    """Filesystem-safe label such as ``0p65`` used in the versioned analysis path."""
    return f"{float(threshold):.4g}".replace(".", "p")


def operating_threshold(split: str, override: float | None = None) -> float:
    """Resolve the operating confidence threshold for operational error analysis.

    The exporter keeps the complete AP-consistent stream (``conf=0.001``); this
    threshold only decides which of those predictions count as detections in
    matching, false-positive/missed counts, maturity confusion and failure-case
    selection. The frozen operating point is the validation F1 optimum
    (``OPERATING_CONFIDENCE_THRESHOLD`` = 0.65). Validation may be re-run at
    another threshold for sensitivity analysis, but the final test must reuse the
    frozen validation value: test-guided threshold selection is forbidden. A
    validation override is marked ``sensitivity_only_non_official`` in the report
    and is never presented as the frozen operating point.
    """
    if split == "test":
        if override is not None and float(override) != OPERATING_CONFIDENCE_THRESHOLD:
            raise PermissionError(
                "final test must reuse the validation-frozen operating threshold "
                f"{OPERATING_CONFIDENCE_THRESHOLD}; test-guided threshold selection is forbidden")
        return OPERATING_CONFIDENCE_THRESHOLD
    if override is None:
        return OPERATING_CONFIDENCE_THRESHOLD
    value = float(override)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("operating confidence threshold must be in (0, 1]")
    return value


def analysis_directory(run_dir: Path, split: str, threshold: float) -> Path:
    """Versioned analysis directory; the legacy unversioned analysis is never touched."""
    return run_dir / "analysis" / f"{split}_conf_{threshold_label(threshold)}_v{ANALYSIS_VERSION}"


def filter_predictions(groups: dict[str, list[dict]], threshold: float) -> tuple[dict[str, list[dict]], dict]:
    """Split the exported prediction stream into the operational operating point."""
    kept: dict[str, list[dict]] = {}
    exported = 0
    for name, predictions in groups.items():
        exported += len(predictions)
        above = [item for item in predictions if float(item["confidence"]) >= threshold]
        if above:
            kept[name] = above
    retained = sum(len(items) for items in kept.values())
    return kept, {"prediction_export_conf": PREDICTION_EXPORT_CONF,
                  "operating_confidence_threshold": threshold,
                  "exported_predictions": exported, "operational_predictions": retained,
                  "predictions_below_threshold_excluded": exported - retained,
                  "prediction_export_modified": False}


def git_state(repo_root: Path = ROOT.parent) -> dict:
    """Current HEAD and worktree state; never raises when Git is unavailable."""
    def captured(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=False)

    head, status = captured("rev-parse", "HEAD"), captured("status", "--porcelain=v1", "--untracked-files=all")
    if head.returncode != 0:
        return {"available": False, "head": None, "dirty": None, "status": []}
    lines = status.stdout.splitlines() if status.returncode == 0 else []
    return {"available": True, "head": head.stdout.strip() or None, "dirty": bool(lines), "status": lines}


def analyze(run_dir: Path, data_root: Path, *, split: str = "val", final_test: bool = False,
            operating_confidence: float | None = None) -> dict:
    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
    checked_split(split, final_test, manifest)
    # Threshold lock first: the final test may never pick its own operating point,
    # so that must fail before any file of the run is read.
    threshold = operating_threshold(split, operating_confidence)
    official_point = threshold == OPERATING_CONFIDENCE_THRESHOLD
    if not official_point:
        warnings.warn(
            f"SENSITIVITY-ONLY ANALYSIS: operating confidence {threshold} differs from the frozen "
            f"official operating point {OPERATING_CONFIDENCE_THRESHOLD}; this analysis is non-official "
            "and must never be reported as the frozen E01 operating point",
            UserWarning, stacklevel=2)
    issues, provenance_detail = provenance_report(manifest, run_dir, repo_root=ROOT)
    output = analysis_directory(run_dir, split, threshold)
    legacy_analysis = run_dir / "analysis" / f"analysis_{split}.json"
    if (output / f"analysis_{split}.json").exists():
        raise FileExistsError(
            f"versioned analysis already exists ({output.name}); earlier analysis must not be silently overwritten")
    metrics = manifest.get("validation_metrics_box" if split == "val" else "test_metrics_box")
    mask_metrics = manifest.get("validation_metrics_mask" if split == "val" else "test_metrics_mask")
    if not metrics or not mask_metrics:
        raise ValueError("real box AND mask validation/test metrics required; no placeholders")
    training = parse_training_csv(run_dir / "results.csv")
    # The exported stream is read in full and left untouched; only the operational
    # operating point below feeds matching, FP/missed counts and failure cases.
    prediction_groups = read_predictions(run_dir, split)
    operational_groups, operating_point = filter_predictions(prediction_groups, threshold)
    frozen = [r for r in csv_rows(ROOT / "data/manifests/d2_split_frozen.csv") if r["final_split"] == split]
    names = {r["filename"]: r for r in frozen}
    if set(prediction_groups) - set(names):
        raise ValueError("prediction refers to image outside frozen split")
    if len(names) != len(frozen):
        raise ValueError("frozen split contains ambiguous filenames")
    matched = []
    for name in sorted(names):
        row = names[name]
        predictions = operational_groups.get(name, [])
        for pred in predictions:
            if (pred["source_group_id"], pred["split_guard_cluster_id"]) != (
                    row["source_group_id"], row["split_guard_cluster_id"]):
                raise ValueError("prediction provenance group differs from frozen manifest")
        for item in match_image(load_ground_truth(data_root,split,name), predictions):
            gt, pred = item["gt"],item["pred"]
            diff = abs(int(gt["gt_class"])-int(pred["pred_class"])) if gt and pred else None
            category = ("F1 Missed Fruit" if item["kind"] == "missed" else
                        "F2 False Positive" if item["kind"] == "false_positive" else
                        "F4 Severe Maturity Confusion" if diff == 2 else
                        "F3 Adjacent Maturity Confusion" if diff == 1 else
                        "F5 Poor Mask Localization" if item["mask_iou"] is not None and item["mask_iou"] < .5 else "")
            matched.append({"image_id": name, "gt_instance_id": gt["gt_instance_id"] if gt else "",
                            "gt_class": gt["gt_class"] if gt else "",
                            "pred_instance_id": pred["pred_instance_id"] if pred else "",
                            "pred_class": pred["pred_class"] if pred else "",
                            "confidence": pred["confidence"] if pred else "",
                            "box_iou": item["box_iou"] if gt and pred else "",
                            "mask_iou": item["mask_iou"] if item["mask_iou"] is not None else "",
                            "matched": str(item["kind"] == "matched").lower(), "split": split,
                            "source_group_id": row["source_group_id"],
                            "split_guard_cluster_id": row["split_guard_cluster_id"],
                            "failure_type": category,
                            "human_review": "Needs Human Review" if category else "not flagged"})
    csv_write(output / f"matched_predictions_{split}.csv", MATCH_FIELDS, matched)
    simple = []
    for row in matched:
        simple.append({"kind": "matched" if row["matched"] == "true" else
                       "missed" if row["gt_instance_id"] != "" else "false_positive",
                       "gt": {"gt_class": int(row["gt_class"])} if row["gt_instance_id"] != "" else None,
                       "pred": {"pred_class": int(row["pred_class"])} if row["pred_instance_id"] != "" else None})
    summary = maturity_summary(simple)
    cases = select_cases([{"category": row["failure_type"], "image_id": row["image_id"],
                           "confidence": row["confidence"] or None, "mask_iou": row["mask_iou"] or None,
                           "gt_instance_id": row["gt_instance_id"] or None,
                           "pred_instance_id": row["pred_instance_id"] or None}
                          for row in matched])
    case_root = output / "failure_cases"
    for category, records in cases.items():
        destination = case_root / category
        destination.mkdir(parents=True, exist_ok=True)
        for i, record in enumerate(records):
            suffix = Path(record["image_id"]).suffix
            filename = f"{i:02d}_{Path(record['image_id']).stem}{suffix}"
            shutil.copyfile(data_root / "images" / split / record["image_id"], destination / filename)
            (destination / f"{i:02d}_{Path(record['image_id']).stem}.json").write_text(
                json.dumps({**record, "human_review": "Needs Human Review"}, ensure_ascii=False, indent=2)+"\n",
                encoding="utf-8", newline="\n")
    git = git_state()
    report = {"split":split,"run_id":manifest.get("run_id"),"provenance_warnings":issues,
              "provenance_status":"verified" if not issues else "incomplete",
              "provenance_detail":provenance_detail,
              "training_git_commit":manifest.get("git_commit"),
              "analysis_git_commit":git["head"],"analysis_git_dirty":git["dirty"],
              "analysis_script_sha256":digest(Path(__file__)),
              "analysis_version":ANALYSIS_VERSION,"analysis_directory":str(output),
              "superseded_analysis":str(legacy_analysis) if legacy_analysis.is_file() else None,
              "prediction_export_conf":PREDICTION_EXPORT_CONF,
              "operating_confidence_threshold":threshold,
              "threshold_selection_split":THRESHOLD_SELECTION_SPLIT,
              "threshold_selection_basis":THRESHOLD_SELECTION_BASIS,
              "test_guided_threshold_tuning":False,
              "official_operating_point":official_point,
              "analysis_classification":("official_frozen_operating_point" if official_point else
                                         "sensitivity_only_non_official"),
              "sensitivity_only":not official_point,
              "requested_operating_confidence":None if official_point else threshold,
              "operating_point":operating_point,
              "metrics_source":"formal Ultralytics validation/test metrics read from run_manifest.yaml; "
                               "not recomputed and not affected by the operating confidence threshold",
              "input_sha256":{"run_manifest":digest(run_dir / "run_manifest.yaml"),
                              "results_csv":digest(run_dir / "results.csv"),
                              "prediction_csv":digest(run_dir / "predictions" / f"predictions_{split}.csv"),
                              "analysis_script":digest(Path(__file__))},
              "box_metrics":metrics,"mask_metrics":mask_metrics,"training":training,
              "maturity":summary,"failure_case_counts":{k:len(v) for k,v in cases.items()},
              "matching":f"class-independent greedy box IoU>={IOU_THRESHOLD} on confidence>={threshold}; "
                         "one-to-one; mask IoU separately",
              "interpretation":(f"{'official' if official_point else 'SENSITIVITY-ONLY/non-official'} "
                                f"operational analysis at confidence>={threshold}; stage metrics on matched "
                                "instances only; missed/FP reported separately")}
    (output / f"analysis_{split}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                                                     encoding="utf-8",newline="\n")
    plot_actual_results(report, run_dir, output)
    return report


def plot_actual_results(report: dict, run_dir: Path, output: Path) -> None:
    """Make figures only from a completed real analysis and run files.

    Figures are written into the versioned analysis directory so the frozen
    formal figures copied by the runner into ``run_dir/figures/`` stay untouched.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    training = report["training"]
    for name, curves in (("training_losses", training["loss_curves"]),
                         ("validation_metrics", training["validation_metric_curves"])):
        if not curves:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, values in curves.items():
            ax.plot(range(1, len(values)+1), values, label=label)
        ax.set(xlabel="logged epoch", ylabel=name.replace("_", " "))
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures / f"{name}_{report['split']}.png", dpi=180)
        plt.close(fig)
    matrix = report["maturity"]["gt_rows_pred_columns"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="predicted maturity", ylabel="GT maturity",
           title="Matched instances only; misses and false positives excluded")
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(figures / f"maturity_confusion_{report['split']}.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    errors = report["maturity"]
    ax.bar(["correct", "adjacent", "severe", "missed", "false positive"],
           [errors["correct"], errors["adjacent"], errors["severe"],
            errors["missed_detections"], errors["false_positives"]])
    ax.set(ylabel="count", title="Maturity errors and separate detection failures")
    fig.autofmt_xdate(rotation=20)
    fig.tight_layout()
    fig.savefig(figures / f"maturity_errors_{report['split']}.png", dpi=180)
    plt.close(fig)
    for kind in ("box", "mask"):
        per_class = report[f"{kind}_metrics"].get("per_class", {})
        if per_class:
            fig, ax = plt.subplots(figsize=(7, 4))
            ids = sorted(per_class, key=int)
            ax.bar([per_class[i]["name"] for i in ids], [per_class[i]["mAP50_95"] for i in ids])
            ax.set(ylabel=f"{kind} mAP50-95", ylim=(0, 1))
            fig.autofmt_xdate(rotation=20)
            fig.tight_layout()
            fig.savefig(figures / f"per_class_{kind}_ap_{report['split']}.png", dpi=180)
            plt.close(fig)
    # Ultralytics creates actual PR curves during the explicit val/test run.
    source = run_dir / ("validation" if report["split"] == "val" else "final_test")
    for curve in source.glob("*PR_curve.png"):
        shutil.copyfile(curve, figures / f"{report['split']}_{curve.name}")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--data-root",type=Path,default=DEFAULT_DATA)
    parser.add_argument("--split",choices=("val","test"),default="val")
    parser.add_argument("--final-test",action="store_true")
    parser.add_argument("--operating-confidence",type=float,default=None,
                        help="operating confidence threshold for operational error analysis "
                             f"(default {OPERATING_CONFIDENCE_THRESHOLD}, the frozen validation F1 "
                             "point; the final test must reuse it and may not tune on test; any other "
                             "value is only possible for --split val and is recorded as sensitivity-only "
                             "/ non-official)")
    args=parser.parse_args()
    print(json.dumps(analyze(args.run_dir,args.data_root,split=args.split,final_test=args.final_test,
                            operating_confidence=args.operating_confidence),
                     ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
