"""E01 result parser and instance-error analysis; never trains or evaluates models.

Default is validation only. Test artifacts require --split test --final-test and
a run manifest that has already recorded the one authorized final test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
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


def provenance(manifest: dict, run_dir: Path, *, repo_root: Path = ROOT) -> list[str]:
    required = ("run_id", "git_commit", "python", "pytorch", "ultralytics", "training_seed",
                "dataset_zip_sha256", "frozen_pool_sha256", "frozen_split_sha256",
                "protocol_yaml_sha256", "best_checkpoint_sha256", "resolved_config_sha256")
    issues = [f"missing {key}" for key in required if manifest.get(key) is None or manifest.get(key) == ""]
    if manifest.get("experiment_relevant_git_dirty") is not False:
        issues.append("E01-relevant Git paths not recorded clean")
    if manifest.get("ultralytics") != "8.3.220" or manifest.get("training_seed") != 0:
        issues.append("pinned Ultralytics/training seed disagreement")
    checks = {
        "dataset_zip_sha256": repo_root / "data/raw/multistage_apple_v4/dataset-20260508.zip",
        "frozen_pool_sha256": repo_root / "data/manifests/d2_experiment_pool_frozen.csv",
        "frozen_split_sha256": repo_root / "data/manifests/d2_split_frozen.csv",
        "protocol_yaml_sha256": repo_root / "configs/data/d2_frozen_protocol.yaml",
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


def analyze(run_dir: Path, data_root: Path, *, split: str = "val", final_test: bool = False) -> dict:
    manifest = yaml.safe_load((run_dir / "run_manifest.yaml").read_text(encoding="utf-8"))
    checked_split(split, final_test, manifest)
    issues = provenance(manifest, run_dir)
    output = run_dir / "analysis"
    if (output / f"analysis_{split}.json").exists():
        raise FileExistsError("analysis already exists; original analysis must not be silently overwritten")
    metrics = manifest.get("validation_metrics_box" if split == "val" else "test_metrics_box")
    mask_metrics = manifest.get("validation_metrics_mask" if split == "val" else "test_metrics_mask")
    if not metrics or not mask_metrics:
        raise ValueError("real box AND mask validation/test metrics required; no placeholders")
    training = parse_training_csv(run_dir / "results.csv")
    prediction_groups = read_predictions(run_dir, split)
    frozen = [r for r in csv_rows(ROOT / "data/manifests/d2_split_frozen.csv") if r["final_split"] == split]
    names = {r["filename"]: r for r in frozen}
    if set(prediction_groups) - set(names):
        raise ValueError("prediction refers to image outside frozen split")
    if len(names) != len(frozen):
        raise ValueError("frozen split contains ambiguous filenames")
    matched = []
    for name in sorted(names):
        row = names[name]
        predictions = prediction_groups.get(name, [])
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
    report = {"split":split,"run_id":manifest.get("run_id"),"provenance_warnings":issues,
              "provenance_status":"verified" if not issues else "incomplete",
              "input_sha256":{"run_manifest":digest(run_dir / "run_manifest.yaml"),
                              "results_csv":digest(run_dir / "results.csv"),
                              "prediction_csv":digest(run_dir / "predictions" / f"predictions_{split}.csv"),
                              "analysis_script":digest(Path(__file__))},
              "box_metrics":metrics,"mask_metrics":mask_metrics,"training":training,
              "maturity":summary,"failure_case_counts":{k:len(v) for k,v in cases.items()},
              "matching":"class-independent greedy box IoU>=0.5; one-to-one; mask IoU separately",
              "interpretation":"stage metrics on matched instances only; missed/FP reported separately"}
    (output / f"analysis_{split}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",
                                                     encoding="utf-8",newline="\n")
    plot_actual_results(report, run_dir, output)
    return report


def plot_actual_results(report: dict, run_dir: Path, output: Path) -> None:
    """Make figures only from a completed real analysis and run files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
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
    args=parser.parse_args()
    print(json.dumps(analyze(args.run_dir,args.data_root,split=args.split,final_test=args.final_test),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
