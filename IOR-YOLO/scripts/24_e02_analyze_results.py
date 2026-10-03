"""E02 ordinal-run analysis: E01-identical metrics behind E02 provenance gates.

E02 must be compared with the frozen E01 baseline on exactly the same operational
statistics, so this module is an E02 wrapper over the frozen E01 analysis
primitives instead of a second implementation of matching and maturity metrics.

Read-only reuse (the files are never edited; ``scripts/17_analyze_e01_results.py``
is pinned in ``configs/experiments/e02_yolo11n_seg_ordinal.yaml::e01_reference.files_sha256``
and both scripts are tracked as E02-relevant Git paths by
``scripts/22_e02_ordinal_run.py``):

* ``scripts/17_analyze_e01_results.py`` - matching, maturity summary, failure-case
  selection, training-curve parsing, plots and the versioned output directory.
* ``scripts/22_e02_ordinal_run.py`` - E02 identity/parity/immutability helpers.

E02 gates run before any E02 run file is read (see
:func:`required_manifest_evidence`):

* frozen E01 immutability plus E02 run-directory isolation;
* the E02 config, resolved config and best checkpoint must still match the
  digests recorded at training time (stricter than E01, where a config digest is
  only a warning): a formal E02 run is never reinterpreted against a modified
  config or checkpoint;
* the run must have completed the independent validation
  (``status = validated_not_final_tested`` with real Box AND Mask metrics);
* the ordinal manipulation must be the predeclared one: ``lambda_ord = 0.5``, the
  implemented criterion really ran with a recorded normalization scale, and
  ``ordinal_loss`` is part of the recorded loss vector;
* the training artifacts are re-verified from the run itself (``results.csv``
  columns and the TensorBoard tag contract) instead of trusting the manifest.

The two confidence levels stay apart exactly as in E01: the untouched
``conf=0.001`` prediction export (script 23) is read in full, while operational
error analysis uses only the frozen operating point ``conf=0.65``.

The Final Test stays locked: only the frozen validation split is analysed here
and no option of this script selects a split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    """Import a sibling script by path; each script keeps its own module namespace."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analysis = load_script("e02_frozen_e01_analysis", "17_analyze_e01_results.py")
runner = load_script("e02_ordinal_entrypoint", "22_e02_ordinal_run.py")

SPLIT = "val"
ANALYSIS_VERSION = 1
ANALYSIS_LABEL = "ordinal"
SCHEMA_VERSION = "e02-analysis-v1"
VALIDATED_STATUS = "validated_not_final_tested"
OPERATING_CONFIDENCE_THRESHOLD = analysis.OPERATING_CONFIDENCE_THRESHOLD
PREDICTION_EXPORT_CONF = analysis.PREDICTION_EXPORT_CONF
THRESHOLD_SELECTION_BASIS = ("E01 validation F1 operating point (conf=0.65) frozen before the single final "
                             "test; E02 round 1 reuses that value and never re-selects it")
ORDINAL_LAMBDA = 0.5
ORDINAL_CRITERION = "OrdinalSegmentationLoss"
ORDINAL_LOSS_COLUMN = "ordinal_loss"
# Keys of ``e01_reference.frozen_results``; E02 reproduces exactly this set, so the
# comparison with the frozen E01 baseline can never silently drop a statistic.
FROZEN_RESULT_KEYS = ("box_mAP50", "box_mAP50_95", "mask_mAP50", "mask_mAP50_95", "operational_confidence",
                      "gt_instances", "matched", "missed", "false_positives", "gt_coverage_percent",
                      "maturity_percent", "maturity_errors", "mase")
REQUIRED_MANIFEST_FIELDS = ("run_id", "experiment_id", "status", "git_commit", "training_seed",
                            "ultralytics", "experiment_config_sha256", "resolved_config_sha256",
                            "best_checkpoint_sha256", "dataset_zip_sha256", "frozen_pool_sha256",
                            "frozen_split_sha256", "protocol_yaml_sha256", "validation_metrics_box",
                            "validation_metrics_mask", "loss_names", "ordinal", "ordinal_diagnostics",
                            "tensorboard", "e01_reference", "train_e01_parity", "final_test")
# Keys of ``e01_runner.metric_record``; the independent validation must have produced
# a real number for each of them before the run may be analysed.
METRIC_FIELDS = ("precision", "recall", "mAP50", "mAP50_95")


def recorded(value) -> bool:
    """A manifest field counts as recorded only when it carries a real value."""
    if value is None or value == "":
        return False
    if isinstance(value, (dict, list, tuple)) and not value:
        return False
    return True


def measured(value) -> bool:
    """A validation metric counts only when it is a finite real number, never text or a flag."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def assert_validation_split(split: str) -> None:
    """E02 round 1 analyses the frozen validation split only; the Final Test stays locked."""
    if split != SPLIT:
        raise PermissionError(
            "E02 analysis and export are restricted to the frozen validation split; the single Final Test "
            "stays locked in the E01 protocol and is not unlocked by the E02 scripts")


def operating_threshold(override: float | None = None) -> float:
    """Reuse the frozen E01 operating point instead of selecting a new one.

    Operational error analysis (matching, false positives, missed detections,
    maturity confusion, failure cases) is only comparable to E01 at the frozen
    operating point, and the final test must always reuse the validation-frozen
    value. A different threshold would be a sensitivity study: it needs its own
    protocol decision, so it is refused here instead of being silently recorded
    as an unofficial operating point.
    """
    if override is None:
        return OPERATING_CONFIDENCE_THRESHOLD
    value = float(override)
    if value != OPERATING_CONFIDENCE_THRESHOLD:
        raise PermissionError(
            f"E02 round 1 is frozen at the E01 operating confidence {OPERATING_CONFIDENCE_THRESHOLD}; "
            f"{value} is not an authorised operating point (a sensitivity study requires its own protocol "
            "decision and can never replace the frozen operating point)")
    return value


def analysis_directory(run_dir: Path, split: str, threshold: float) -> Path:
    """Versioned E02 analysis directory; the frozen E01 analysis directories are never touched."""
    return (Path(run_dir) / "analysis" /
            f"{split}_conf_{analysis.threshold_label(threshold)}_{ANALYSIS_LABEL}_v{ANALYSIS_VERSION}")


def guard(run_dir: Path, *, config: dict, repo_root: Path) -> dict:
    """E01 immutability + E02 run isolation, executed before any E02 run file is read."""
    immutability = runner.assert_e01_immutable(config, repo_root=Path(repo_root))
    isolation = runner.assert_run_isolation(Path(run_dir), e01_run_dir=runner.E01_RUN_DIR)
    return {"e01_immutability": immutability, "run_isolation": isolation}


def required_manifest_evidence(manifest: dict, config: dict, run_dir: Path) -> dict:
    """Hard E02 provenance gates that must all pass before any run file is read.

    Stricter than the E01 analysis warnings on purpose: the E02 result may only be
    interpreted when the run is the predeclared ordinal experiment (identity,
    config, resolved config, checkpoint, ordinal evidence) and its independent
    validation is finished, so a formal run can never be reinterpreted loosely.
    """
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if not recorded(manifest.get(field))]
    if missing:
        raise RuntimeError(f"E02 run manifest is missing required provenance fields: {missing}")
    if manifest["experiment_id"] != runner.EXPERIMENT_ID:
        raise RuntimeError(f"unexpected experiment_id: {manifest['experiment_id']}")
    if str(manifest["status"]) != VALIDATED_STATUS:
        raise RuntimeError(
            f"E02 analysis requires the completed independent validation (status {VALIDATED_STATUS!r}); "
            f"the run manifest records {manifest['status']!r}")
    for label in ("validation_metrics_box", "validation_metrics_mask"):
        metrics = manifest[label] if isinstance(manifest[label], dict) else {}
        unmeasured = [name for name in METRIC_FIELDS if not measured(metrics.get(name))]
        if unmeasured:
            raise RuntimeError(f"{label} carries no real validation value for {unmeasured}; the E02 analysis "
                               "requires the finished independent validation of both heads")
    if manifest["experiment_config_sha256"] != runner.sha(runner.CONFIG):
        raise RuntimeError("E02 experiment config changed after the formal run; a formal run is never "
                           "reinterpreted against a modified config")
    if manifest["resolved_config_sha256"] != runner.sha(Path(run_dir) / "resolved_train_config.yaml"):
        raise RuntimeError("resolved E02 training config changed after the formal run")
    if manifest["best_checkpoint_sha256"] != runner.sha(Path(run_dir) / "weights/best.pt"):
        raise RuntimeError("E02 best checkpoint differs from the run provenance")
    if manifest["training_seed"] != 0:
        raise RuntimeError(f"E02 round 1 is the single pinned training seed 0; found {manifest['training_seed']}")
    ordinal = manifest["ordinal"]
    if float(ordinal["lambda_ord"]) != ORDINAL_LAMBDA:
        raise RuntimeError(f"lambda_ord must stay {ORDINAL_LAMBDA}, recorded {ordinal['lambda_ord']}; the "
                           "lambda is predeclared and is never re-tuned from results")
    if float(config["ordinal"]["lambda_ord"]) != ORDINAL_LAMBDA:
        raise RuntimeError("the current E02 config no longer declares the frozen lambda_ord")
    if ordinal.get("loss") != config["ordinal"]["loss"]:
        raise RuntimeError("the ordinal loss definition differs between the run manifest and the E02 config")
    diagnostics = manifest["ordinal_diagnostics"]
    if diagnostics.get("criterion_class") != ORDINAL_CRITERION:
        raise RuntimeError(f"the ordinal criterion never ran ({diagnostics.get('criterion_class')!r}); the "
                           "E02 manipulation is not verified")
    if not recorded(diagnostics.get("scale_history")):
        raise RuntimeError("no recorded ordinal loss scale; the auxiliary term never scored an instance")
    loss_names = [str(name) for name in manifest["loss_names"]]
    if not any(ORDINAL_LOSS_COLUMN in name for name in loss_names):
        raise RuntimeError(f"{ORDINAL_LOSS_COLUMN} is missing from the recorded loss vector: {loss_names}")
    integration = manifest["tensorboard"].get("builtin_ultralytics_integration")
    if not isinstance(integration, dict) or integration.get("neutralized") is not True:
        raise RuntimeError("the built-in Ultralytics TensorBoard writer must be neutralized for E02 so the "
                           f"dashboard has exactly one writer; recorded {integration!r}")
    artifacts = runner.verify_training_artifacts(Path(run_dir))
    if artifacts["tensorboard"]["missing_tags"]:
        raise RuntimeError(f"TensorBoard tag contract incomplete: {artifacts['tensorboard']['missing_tags']}")
    return {"experiment_config_sha256": manifest["experiment_config_sha256"],
            "resolved_config_sha256": manifest["resolved_config_sha256"],
            "best_checkpoint_sha256": manifest["best_checkpoint_sha256"],
            "training_commit": manifest["git_commit"], "training_seed": manifest["training_seed"],
            "status": manifest["status"], "ordinal": ordinal, "ordinal_diagnostics": diagnostics,
            "validation_metrics_box": manifest["validation_metrics_box"],
            "validation_metrics_mask": manifest["validation_metrics_mask"],
            "e01_reference": manifest["e01_reference"], "final_test": manifest["final_test"],
            "train_e01_parity": manifest["train_e01_parity"],
            "tensorboard": manifest["tensorboard"], "training_artifacts": artifacts,
            "gates": "all hard E02 provenance gates passed"}


def soft_provenance_issues(manifest: dict, *, repo_root: Path) -> tuple[list[str], dict]:
    """E01-style warnings for the E02 run, documented as warnings and not as gates.

    The raw-source identity check is reused from the frozen E01 analysis. That
    helper's parent function is not reused because it compares
    ``experiment_config_sha256`` against the E01 config path, which is the E02
    config by construction here; the E02 config gates live in
    :func:`required_manifest_evidence`.
    """
    issues, raw_detail = analysis.raw_source_provenance(manifest, repo_root=Path(repo_root))
    if manifest.get("experiment_relevant_git_dirty") is not False:
        issues.append("E01/E02-relevant Git paths were not recorded clean at training time")
    if manifest.get("ultralytics") != "8.3.220" or manifest.get("training_seed") != 0:
        issues.append("pinned Ultralytics/training seed disagreement")
    return issues, {"raw_source": raw_detail, "provenance_mode": "warning_only_E01_style_checks"}


def git_provenance() -> dict:
    """Repository state at analysis time, including the E02-relevant paths; never raises on Git."""
    state = analysis.git_state(repo_root=runner.REPO_ROOT)
    try:
        relevant = runner.git_state()
    except Exception as error:  # pragma: no cover - only when Git is unavailable
        return {**state, "e02_relevant_dirty": None, "e02_relevant_status": [],
                "git_error": f"{type(error).__name__}: {error}"}
    return {**state, "e02_relevant_dirty": relevant["e02_relevant_dirty"],
            "e02_relevant_status": relevant["e02_relevant_status"],
            "experiment_relevant_dirty": relevant["experiment_relevant_dirty"]}


def delta(value, reference) -> float | None:
    """Signed E02 minus E01 difference; ``None`` when a side is missing or non-numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(reference, bool) or not isinstance(reference, (int, float)):
        return None
    return round(float(value) - float(reference), 10)


def e01_analysis_report_record() -> dict | None:
    """Provenance of the frozen E01 analysis report when it is present next to the E01 run."""
    label = analysis.threshold_label(OPERATING_CONFIDENCE_THRESHOLD)
    path = runner.E01_RUN_DIR / "analysis" / f"val_conf_{label}_v{analysis.ANALYSIS_VERSION}" / "analysis_val.json"
    if not path.is_file():
        return None
    return {"path": str(path), "sha256": analysis.digest(path)}


def compare_to_e01(manifest: dict, summary: dict, box: dict, mask: dict) -> dict:
    """Signed E02 minus E01 deltas against the frozen E01 numbers read from the E02 config.

    The E01 values are the frozen reference declared in
    ``e02_yolo11n_seg_ordinal.yaml``; they are never recomputed here, so the
    analysis can never quote a "re-derived" baseline. The E02 side is assembled
    key-for-key from ``FROZEN_RESULT_KEYS``; any key that cannot be rebuilt is
    reported explicitly instead of being dropped.
    """
    frozen = manifest["e01_reference"]["frozen_results"]
    matched = summary["matched_instances"]
    per_class = {}
    for name, class_name in zip(("immature", "semi-mature", "mature"), analysis.CLASSES):
        accuracy = summary["per_gt_class"][class_name]["classification_accuracy_matched"]
        per_class[name] = accuracy * 100.0 if accuracy is not None else None
    coverage = summary["matched_gt_coverage"]
    operational = {
        "box_mAP50": box.get("mAP50"), "box_mAP50_95": box.get("mAP50_95"),
        "mask_mAP50": mask.get("mAP50"), "mask_mAP50_95": mask.get("mAP50_95"),
        "operational_confidence": OPERATING_CONFIDENCE_THRESHOLD,
        "gt_instances": matched + summary["missed_detections"], "matched": matched,
        "missed": summary["missed_detections"], "false_positives": summary["false_positives"],
        "gt_coverage_percent": 100.0 * coverage if coverage is not None else None,
        "maturity_percent": per_class,
        "maturity_errors": {"correct": summary["correct"], "adjacent": summary["adjacent"],
                            "severe": summary["severe"]},
        "mase": summary["MASE_stage_matched_only"],
    }
    differences: dict = {}
    for key, value in operational.items():
        reference = frozen.get(key)
        if isinstance(value, dict):
            differences[key] = {name: delta(entry, (reference or {}).get(name)) for name, entry in value.items()}
        else:
            differences[key] = delta(value, reference)
    return {"e01_frozen_results": frozen,
            "e01_reference_config": "configs/experiments/e02_yolo11n_seg_ordinal.yaml::e01_reference",
            "e01_training_commit": manifest["e01_reference"].get("training_commit"),
            "e01_analysis_commit": manifest["e01_reference"].get("analysis_commit"),
            "e01_analysis_report": e01_analysis_report_record(),
            "expected_keys": list(FROZEN_RESULT_KEYS),
            "e02_operational": operational, "delta_e02_minus_e01": differences,
            "not_reproduced": sorted(set(frozen) - set(operational)),
            "rounding_note": "the frozen E01 reference stores percentages at 2 dp and MASE at 4 dp, so deltas "
                             "at that precision are indicative; the unrounded E01 analysis report stays the "
                             "E01 source of record",
            "note": "E01 values are read from the frozen reference and are never recomputed here"}


def analyze(run_dir: Path = runner.RUN_DIR, data_root: Path = analysis.DEFAULT_DATA, *, split: str = SPLIT,
            operating_confidence: float | None = None, config: dict | None = None,
            repo_root: Path | None = None) -> dict:
    """Official E02 operational analysis at the frozen operating point.

    The order is deliberate: split lock, operating point and every provenance gate
    (E01 immutability, run isolation, config/checkpoint digests, ordinal evidence)
    run before one file of the E02 run is opened, so an untrustworthy run cannot
    produce an analysis artifact at all. The E01 analysis directory is never
    written to: the E02 report and figures live in their own versioned directory.
    """
    assert_validation_split(split)
    run_dir, data_root = Path(run_dir), Path(data_root)
    repo_root = Path(repo_root) if repo_root is not None else runner.ROOT
    config = runner.load_config() if config is None else config
    threshold = operating_threshold(operating_confidence)
    guards = guard(run_dir, config=config, repo_root=repo_root)
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E02 run manifest absent: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    evidence = required_manifest_evidence(manifest, config, run_dir)
    output = analysis_directory(run_dir, split, threshold)
    if (output / f"analysis_{split}.json").exists():
        raise FileExistsError(f"versioned E02 analysis already exists ({output.name}); earlier analysis must "
                              "not be silently overwritten")
    metrics, mask_metrics = manifest["validation_metrics_box"], manifest["validation_metrics_mask"]
    training = analysis.parse_training_csv(run_dir / "results.csv")
    # The conf=0.001 export is read in full and left untouched; only the frozen
    # operating point below feeds matching, FP/missed counts and failure cases.
    prediction_groups = analysis.read_predictions(run_dir, split)
    operational_groups, operating_point = analysis.filter_predictions(prediction_groups, threshold)
    frozen = [row for row in analysis.csv_rows(repo_root / "data/manifests/d2_split_frozen.csv")
              if row["final_split"] == split]
    names = {row["filename"]: row for row in frozen}
    if set(prediction_groups) - set(names):
        raise ValueError("prediction refers to an image outside the frozen split")
    if len(names) != len(frozen):
        raise ValueError("frozen split contains ambiguous filenames")
    export_detail = {"frozen_split_images": len(frozen),
                     "images_with_any_export": len(prediction_groups),
                     "images_without_prediction": len(set(names) - set(prediction_groups)),
                     "images_with_operational_prediction": len(operational_groups)}
    matched = []
    for name in sorted(names):
        row = names[name]
        predictions = operational_groups.get(name, [])
        for pred in predictions:
            if (pred["source_group_id"], pred["split_guard_cluster_id"]) != (row["source_group_id"],
                                                                            row["split_guard_cluster_id"]):
                raise ValueError("prediction provenance group differs from the frozen manifest")
        for item in analysis.match_image(analysis.load_ground_truth(data_root, split, name), predictions):
            gt, pred = item["gt"], item["pred"]
            difference = abs(int(gt["gt_class"]) - int(pred["pred_class"])) if gt and pred else None
            category = ("F1 Missed Fruit" if item["kind"] == "missed" else
                        "F2 False Positive" if item["kind"] == "false_positive" else
                        "F4 Severe Maturity Confusion" if difference == 2 else
                        "F3 Adjacent Maturity Confusion" if difference == 1 else
                        "F5 Poor Mask Localization" if item["mask_iou"] is not None and item["mask_iou"] < .5
                        else "")
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


    analysis.csv_write(output / f"matched_predictions_{split}.csv", analysis.MATCH_FIELDS, matched)
    simple = []
    for row in matched:
        simple.append({"kind": "matched" if row["matched"] == "true" else
                       "missed" if row["gt_instance_id"] != "" else "false_positive",
                       "gt": {"gt_class": int(row["gt_class"])} if row["gt_instance_id"] != "" else None,
                       "pred": {"pred_class": int(row["pred_class"])} if row["pred_instance_id"] != "" else None})
    summary = analysis.maturity_summary(simple)
    cases = analysis.select_cases([{"category": row["failure_type"], "image_id": row["image_id"],
                                    "confidence": row["confidence"] or None,
                                    "mask_iou": row["mask_iou"] or None,
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
                json.dumps({**record, "human_review": "Needs Human Review"}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n")
    provenance_issues, provenance_detail = soft_provenance_issues(manifest, repo_root=repo_root)
    report = build_report({"manifest": manifest, "run_dir": run_dir, "output": output, "split": split,
                           "threshold": threshold, "evidence": evidence, "guards": guards,
                           "git": git_provenance(), "summary": summary, "cases": cases, "training": training,
                           "box": metrics, "mask": mask_metrics, "operating_point": operating_point,
                           "export_detail": export_detail, "provenance_issues": provenance_issues,
                           "provenance_detail": provenance_detail})
    return persist_report(report, run_dir=run_dir, output=output, split=split)


def tensorboard_contract() -> dict:
    """The canonical E02 tag contract, read from the E02 logger so it is never re-typed here."""
    from ior_yolo.utils.tensorboard_logger import ORDINAL_LOSS_NAME, REQUIRED_TAGS, WRITER_DESCRIPTION, required_tag_list

    return {"writer": WRITER_DESCRIPTION, "ordinal_tag": ORDINAL_LOSS_NAME,
            "required_tags": required_tag_list(), "groups": {name: list(tags) for name, tags in REQUIRED_TAGS.items()}}


def build_report(context: dict) -> dict:
    """Assemble the versioned E02 report from gated run facts only.

    Every number here is either read from the run (metrics, manifest, results.csv)
    or derived with the frozen E01 analysis primitives, and the ordinal loss curves
    are surfaced explicitly so the paper can trace the auxiliary term.
    """
    manifest, run_dir, output = context["manifest"], context["run_dir"], context["output"]
    training, evidence = context["training"], context["evidence"]
    diagnostics = manifest["ordinal_diagnostics"]
    ordinal_curves = {name: values for name, values in training["loss_curves"].items() if ORDINAL_LOSS_COLUMN in name}
    if not any(name.startswith("train/") for name in ordinal_curves):
        raise RuntimeError(f"results.csv records no train/{ORDINAL_LOSS_COLUMN} curve; the E02 loss vector is "
                           "incomplete and the ordinal term is not evidenced")
    report = {"schema_version": SCHEMA_VERSION, "experiment_id": manifest["experiment_id"],
              "run_id": manifest["run_id"], "run_directory": str(run_dir), "run_status": manifest["status"],
              "split": context["split"], "analysis_label": ANALYSIS_LABEL, "analysis_version": ANALYSIS_VERSION,
              "analysis_script": "scripts/24_e02_analyze_results.py",
              "analysis_script_sha256": analysis.digest(Path(__file__)),
              "analysis_level": "E02 wrapper over the frozen E01 analysis primitives (scripts/17); E01 never edited",
              "analysis_directory": str(output), "superseded_analysis": None,
              "training_git_commit": manifest["git_commit"], "analysis_git_commit": context["git"]["head"],
              "analysis_git_dirty": context["git"]["dirty"],
              "e02_relevant_git_dirty": context["git"]["e02_relevant_dirty"],
              "e02_relevant_git_status": context["git"]["e02_relevant_status"],
              # All hard gates ran before this point and raise on failure, so the
              # soft E01-style warnings below never weaken the verified verdict.
              "provenance_status": ("verified" if not context["provenance_issues"]
                                    else "verified_with_soft_warnings"),
              "provenance_warnings": context["provenance_issues"],
              "provenance_detail": context["provenance_detail"], "e02_gates": evidence,
              "e01_immutability": context["guards"]["e01_immutability"],
              "run_isolation": context["guards"]["run_isolation"],
              "prediction_export_conf": PREDICTION_EXPORT_CONF,
              "operating_confidence_threshold": context["threshold"],
              "threshold_selection_split": SPLIT, "threshold_selection_basis": THRESHOLD_SELECTION_BASIS,
              "test_guided_threshold_tuning": False, "official_operating_point": True,
              "analysis_classification": "official_frozen_operating_point", "sensitivity_only": False,
              "requested_operating_confidence": None, "operating_point": context["operating_point"],
              "export_detail": context["export_detail"],
              "metrics_source": "formal Ultralytics validation metrics read from run_manifest.yaml; not "
                                "recomputed and not affected by the operating confidence threshold",
              "input_sha256": {"run_manifest": analysis.digest(run_dir / "run_manifest.yaml"),
                               "results_csv": analysis.digest(run_dir / "results.csv"),
                               "prediction_csv": analysis.digest(
                                   run_dir / "predictions" / f"predictions_{context['split']}.csv"),
                               "resolved_train_config": analysis.digest(run_dir / "resolved_train_config.yaml"),
                               "analysis_script": analysis.digest(Path(__file__))},
              "box_metrics": context["box"], "mask_metrics": context["mask"], "training": training,
              "maturity": context["summary"],
              "failure_case_counts": {name: len(rows) for name, rows in context["cases"].items()},
              "matching": f"class-independent greedy box IoU>={analysis.IOU_THRESHOLD} on "
                          f"confidence>={context['threshold']}; one-to-one; mask IoU separately"}
    report.update(
        ordinal={"lambda_ord": float(manifest["ordinal"]["lambda_ord"]),
                 "loss_definition": manifest["ordinal"]["loss"],
                 "lambda_selection": manifest["ordinal"].get("lambda_selection"),
                 "criterion_class": diagnostics.get("criterion_class"),
                 "assigner_calls": diagnostics.get("assigner_calls"),
                 "criterion_steps": diagnostics.get("criterion_steps"),
                 "scale_history": diagnostics.get("scale_history"), "loss_items": manifest["loss_names"],
                 "recorded_ordinal_curves": {name: {"points": len(values), "first": values[0], "last": values[-1],
                                                    "min": min(values), "max": max(values)}
                                             for name, values in sorted(ordinal_curves.items())},
                 "note": "lambda_ord is predeclared and was not selected from these numbers"},
        tensorboard={**evidence["training_artifacts"]["tensorboard"], "contract": tensorboard_contract(),
                     "builtin_ultralytics_integration": evidence["tensorboard"].get(
                         "builtin_ultralytics_integration")},
        e01_comparison=compare_to_e01(manifest, context["summary"], context["box"], context["mask"]),
        final_test={**manifest["final_test"], "analysed_here": False,
                    "note": "the single Final Test stays locked in the E01 protocol; the E02 analysis and "
                            "export scripts expose no test path"},
        interpretation=f"official E02 operational analysis at confidence>={context['threshold']}; stage metrics "
                       "on matched instances only; missed/FP reported separately; the E01 baseline is the frozen "
                       "reference and is never recomputed")
    return report


def persist_report(report: dict, *, run_dir: Path, output: Path, split: str) -> dict:
    """Write the versioned report and its figures; the frozen E01 analysis is never written to."""
    (output / f"analysis_{split}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                   encoding="utf-8", newline="\n")
    analysis.plot_actual_results(report, run_dir, output)
    return report


def compact(report: dict) -> dict:
    """Console digest of the full report; the JSON file stays the source of record."""
    comparison = report["e01_comparison"]
    ordinal = report["ordinal"]
    return {"schema_version": report["schema_version"], "experiment_id": report["experiment_id"],
            "run_id": report["run_id"], "run_status": report["run_status"], "split": report["split"],
            "analysis_directory": report["analysis_directory"],
            "operating_confidence_threshold": report["operating_confidence_threshold"],
            "analysis_classification": report["analysis_classification"],
            "provenance_status": report["provenance_status"],
            "provenance_warnings": report["provenance_warnings"],
            "box": {key: report["box_metrics"].get(key) for key in ("precision", "recall", "mAP50", "mAP50_95")},
            "mask": {key: report["mask_metrics"].get(key) for key in ("precision", "recall", "mAP50", "mAP50_95")},
            "operational": comparison["e02_operational"],
            "delta_e02_minus_e01": comparison["delta_e02_minus_e01"],
            "ordinal": {"lambda_ord": ordinal["lambda_ord"], "criterion_class": ordinal["criterion_class"],
                        "assigner_calls": ordinal["assigner_calls"], "criterion_steps": ordinal["criterion_steps"],
                        "curves": ordinal["recorded_ordinal_curves"]},
            "tensorboard_missing_tags": report["tensorboard"]["missing_tags"],
            "run_isolation": report["run_isolation"]["isolated"],
            "e01_immutability": report["e01_immutability"]["status"],
            "final_test_status": report["final_test"]["status"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=runner.RUN_DIR)
    parser.add_argument("--data-root", type=Path, default=analysis.DEFAULT_DATA)
    parser.add_argument("--operating-confidence", type=float, default=None,
                        help="must equal the frozen E01 operating point "
                             f"{OPERATING_CONFIDENCE_THRESHOLD}; any other value is refused because a new "
                             "operating point needs its own protocol decision")
    parser.add_argument("--summary", action="store_true",
                        help="print the compact console digest instead of the full report")
    args = parser.parse_args()
    report = analyze(args.run_dir, args.data_root, operating_confidence=args.operating_confidence)
    print(json.dumps(compact(report) if args.summary else report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


# APPEND-CURSOR
