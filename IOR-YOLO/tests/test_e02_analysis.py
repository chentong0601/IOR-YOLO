"""E02 analysis/export gates exercised on synthetic artifacts only.

No training, no CUDA, no model inference and no access to the frozen E01 run: the
E02 run is a temporary synthetic run directory, so the hard provenance gates, the
E01-identical matching/maturity metrics, the frozen operating point, the ordinal
evidence and the Final-Test lock are all checked without touching real evidence.
"""

import copy
import csv
import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def module(number, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{number}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


analyzer = module("24_e02_analyze_results.py", "e02_analyzer_test")
export_module = module("23_e02_export_predictions.py", "e02_export_test")
runner = analyzer.runner
a = analyzer.analysis

from ior_yolo.utils.tensorboard_logger import (  # noqa: E402
    TENSORBOARD_DIRNAME, default_writer_factory, neutralize_builtin_ultralytics_tensorboard,
    required_tag_list, tensorboard_roundtrip)

EXPERIMENT = runner.EXPERIMENT_ID
STATUS = "validated_not_final_tested"
LOSS_NAMES = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "ordinal_loss")
RESULT_COLUMNS = ("epoch", "train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss",
                  "train/ordinal_loss", "val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss",
                  "val/ordinal_loss", "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
                  "metrics/mAP50-95(B)", "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)",
                  "metrics/mAP50-95(M)", "lr/pg0")
FROZEN_E01_RESULTS = {"box_mAP50": 0.96, "box_mAP50_95": 0.91, "mask_mAP50": 0.95, "mask_mAP50_95": 0.90,
                      "operational_confidence": 0.65, "gt_instances": 100, "matched": 90, "missed": 10,
                      "false_positives": 5, "gt_coverage_percent": 90.0,
                      "maturity_percent": {"immature": 80.0, "semi-mature": 70.0, "mature": 60.0},
                      "maturity_errors": {"correct": 60, "adjacent": 25, "severe": 5}, "mase": 0.35}
BOX_METRICS = {"precision": 0.90, "recall": 0.80, "mAP50": 0.97, "mAP50_95": 0.92}
MASK_METRICS = {"precision": 0.89, "recall": 0.79, "mAP50": 0.96, "mAP50_95": 0.91}
# Exactly the keys emitted by ``ior_yolo.trainers.ordinal.summarize_scale``; the fixture mirrors
# the real producer so the analysis gate is exercised against the real shape.
SCALE_HISTORY = {"steps_recorded": 202, "steps_with_instances": 202, "mean_stock_cls_loss": 0.9,
                 "mean_raw_ordinal_loss": 0.04, "mean_lambda_times_ordinal": 0.02,
                 "mean_lambda_scaled_ordinal_batch": 0.04, "mean_ordinal_share_of_total": 0.004,
                 "mean_instances_per_step": 3.0, "mean_quality": 0.8,
                 "scope": "engineering scale sanity only; lambda_ord stays frozen at 0.5 and was NOT tuned "
                          "from these numbers"}
SPLIT_ROW = {"filename": "synthetic.jpg", "final_split": "val", "source_group_id": "sg-test",
             "split_guard_cluster_id": "gc-test"}
DETECTIONS = ((0.9, 0), (0.2, 2))


def write_results_csv(path, *, ordinal=0.5, columns=RESULT_COLUMNS, epochs=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for epoch in range(1, epochs + 1):
            row = []
            for name in columns:
                if name == "epoch":
                    row.append(float(epoch))
                elif name.endswith("ordinal_loss"):
                    row.append(round(1.0 - 0.2 * epoch, 4))
                elif name == "lr/pg0":
                    row.append(0.01)
                else:
                    row.append(0.5)
            if "train/ordinal_loss" in columns:
                row[columns.index("train/ordinal_loss")] = ordinal
            writer.writerow(row)


def write_predictions(fixture, rows=DETECTIONS, *, box=(2, 2, 14, 14), masks=True):
    """Write an E02 prediction export (default: one detection above the operating point)."""
    path = fixture["run"] / "predictions/predictions_val.csv"
    if masks:
        (fixture["run"] / "predictions/masks_val").mkdir(parents=True, exist_ok=True)
    a.csv_write(path, a.PRED_FIELDS,
                [{"image_id": SPLIT_ROW["filename"], "pred_instance_id": f"p{index:04d}", "pred_class": klass,
                  "confidence": f"{confidence:.10f}", "box": json.dumps(list(box)), "mask_reference": "",
                  "source_group_id": SPLIT_ROW["source_group_id"],
                  "split_guard_cluster_id": SPLIT_ROW["split_guard_cluster_id"], "split": "val"}
                 for index, (confidence, klass) in enumerate(rows)])
    return path


def write_manifest(fixture, manifest):
    (fixture["run"] / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def synthetic_run(root):
    """Temporary synthetic E02 repository: frozen-base file, split, run and manifest.

    The config is synthetic as well, so the E01-immutability gate is exercised against
    a digest map that lives inside the fixture instead of the real frozen baseline.
    """
    run = root / "runs" / EXPERIMENT / "seed_0"
    e01_run = root / "runs/e01_yolo11n_seg/seed_0"
    data = root / "processed"
    for path in (run / "predictions", run / "weights", e01_run, data / "images/val", data / "labels/val",
                 root / "data/manifests", root / "configs/frozen"):
        path.mkdir(parents=True, exist_ok=True)
    frozen = root / "configs/frozen/e01_analysis.py"
    frozen.write_text("# frozen synthetic E01 artifact\n", encoding="utf-8")
    ordinal_block = dict(yaml.safe_load(runner.CONFIG.read_text(encoding="utf-8"))["ordinal"])
    config = {"experiment_id": EXPERIMENT, "ordinal": ordinal_block,
              "e01_reference": {"files_sha256": {"configs/frozen/e01_analysis.py": runner.sha(frozen)},
                                "frozen_results": copy.deepcopy(FROZEN_E01_RESULTS),
                                "training_commit": "a" * 40, "analysis_commit": "b" * 40}}
    config_path = root / "configs/e02_synthetic.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    split = root / "data/manifests/d2_split_frozen.csv"
    with split.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(SPLIT_ROW), lineterminator="\n")
        writer.writeheader()
        writer.writerow(SPLIT_ROW)
    Image.new("RGB", (20, 20), (90, 90, 90)).save(data / "images/val/synthetic.jpg")
    (data / "labels/val/synthetic.txt").write_text("0 0.1 0.1 0.7 0.1 0.7 0.7 0.1 0.7\n", encoding="utf-8")
    write_results_csv(run / "results.csv")
    tensorboard_roundtrip(run / TENSORBOARD_DIRNAME)
    (run / "weights/best.pt").write_bytes(b"synthetic E02 checkpoint")
    (run / "resolved_train_config.yaml").write_text("epochs: 101\n", encoding="utf-8")
    fixture = {"root": root, "run": run, "e01_run": e01_run, "data": data, "config": config,
               "config_path": config_path, "frozen": frozen, "split": split}
    fixture["manifest"] = write_default_manifest(fixture)
    return fixture


def write_default_manifest(fixture):
    """A complete, self-consistent E02 manifest for the synthetic run."""
    run, root, config = fixture["run"], fixture["root"], fixture["config"]
    ordinal_block = config["ordinal"]
    manifest = {"run_id": "synthetic-e02-run", "experiment_id": EXPERIMENT, "status": STATUS,
                "git_commit": "c" * 40, "training_seed": 0, "ultralytics": "8.3.220",
                "python": "3.12.14", "pytorch": "2.5.1", "experiment_relevant_git_dirty": False,
                "experiment_config_sha256": runner.sha(fixture["config_path"]),
                "resolved_config_sha256": runner.sha(run / "resolved_train_config.yaml"),
                "best_checkpoint_sha256": runner.sha(run / "weights/best.pt"),
                "dataset_zip_sha256": "d" * 64, "frozen_pool_sha256": "e" * 64,
                "frozen_split_sha256": a.digest(fixture["split"]), "protocol_yaml_sha256": "f" * 64,
                "validation_metrics_box": copy.deepcopy(BOX_METRICS),
                "validation_metrics_mask": copy.deepcopy(MASK_METRICS),
                "loss_names": list(LOSS_NAMES),
                "ordinal": {"lambda_ord": 0.5, "loss": ordinal_block["loss"],
                            "lambda_selection": ordinal_block.get("lambda_selection")},
                "ordinal_diagnostics": {"criterion_class": "OrdinalSegmentationLoss", "assigner_calls": 101,
                                        "criterion_steps": 202, "scale_history": copy.deepcopy(SCALE_HISTORY)},
                "train_e01_parity": {"status": "identical", "checked_keys": ["epochs", "imgsz"]},
                "tensorboard": {"writer": "torch.utils.tensorboard.SummaryWriter",
                                "builtin_ultralytics_integration": {"neutralized": True,
                                                                    "builtin_callbacks_registered": 1}},
                "e01_reference": copy.deepcopy(config["e01_reference"]),
                "final_test": {"status": runner.FINAL_TEST_POLICY, "test_split_access": "not_accessed"}}
    (run / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest


class SyntheticFixtureTestCase(unittest.TestCase):
    """Each test gets a disposable synthetic E02 run and a patched synthetic config."""

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(prefix="e02-analysis-")
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.fixture = synthetic_run(self.root)
        patcher = patch.object(runner, "CONFIG", self.fixture["config_path"])
        patcher.start()
        self.addCleanup(patcher.stop)

    def turntable(self, mutate):
        """Apply ``mutate(manifest)`` and persist it as the run manifest, then return it."""
        manifest = copy.deepcopy(self.fixture["manifest"])
        mutate(manifest)
        write_manifest(self.fixture, manifest)
        self.fixture["manifest"] = manifest
        return manifest

    def evidence(self, manifest=None):
        selected = self.fixture["manifest"] if manifest is None else manifest
        return analyzer.required_manifest_evidence(selected, self.fixture["config"], self.fixture["run"])

    def guards(self):
        return analyzer.guard(self.fixture["run"], config=self.fixture["config"], repo_root=self.root)

    def analyze(self, **overrides):
        arguments = {"run_dir": self.fixture["run"], "data_root": self.fixture["data"],
                     "config": self.fixture["config"], "repo_root": self.root}
        arguments.update(overrides)
        return analyzer.analyze(**arguments)


class GateContractTest(SyntheticFixtureTestCase):
    """E02 hard provenance gates: a complete manifest passes, every gap raises."""

    def test_complete_manifest_passes_every_hard_gate(self):
        evidence = self.evidence()
        self.assertEqual(evidence["gates"], "all hard E02 provenance gates passed")
        self.assertEqual(evidence["status"], STATUS)
        self.assertEqual(evidence["training_seed"], 0)
        self.assertEqual(evidence["experiment_config_sha256"], runner.sha(self.fixture["config_path"]))
        self.assertEqual(evidence["resolved_config_sha256"],
                         runner.sha(self.fixture["run"] / "resolved_train_config.yaml"))
        self.assertEqual(evidence["best_checkpoint_sha256"], runner.sha(self.fixture["run"] / "weights/best.pt"))
        self.assertEqual(evidence["ordinal"]["lambda_ord"], 0.5)
        artifacts = evidence["training_artifacts"]
        self.assertEqual(artifacts["tensorboard"]["missing_tags"], [])
        self.assertEqual(artifacts["epochs_recorded"], 3)
        self.assertTrue(set(RESULT_COLUMNS).issubset(artifacts["columns"]))

    def test_guard_verifies_e01_immutability_and_run_isolation(self):
        guards = self.guards()
        self.assertEqual(guards["e01_immutability"]["status"], "verified")
        self.assertEqual(guards["e01_immutability"]["files_checked"], 1)
        self.assertEqual(guards["run_isolation"]["isolated"], True)
        self.assertNotIn("e01", self.fixture["run"].parts)

    def test_tampered_frozen_e01_artifact_stops_the_analysis(self):
        self.fixture["frozen"].write_text("# mutated frozen baseline\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            self.guards()
        write_predictions(self.fixture)
        with self.assertRaises(RuntimeError):
            self.analyze()
        self.assertFalse((self.fixture["run"] / "analysis").exists())

    def test_incomplete_manifest_is_refused(self):
        manifest = self.turntable(lambda item: item.pop("tensorboard"))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_wrong_experiment_identity_is_refused(self):
        manifest = self.turntable(lambda item: item.update(experiment_id="e01_yolo11n_seg"))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_validation_must_have_finished_before_analysis(self):
        manifest = self.turntable(lambda item: item.update(status="trained_not_validated"))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_missing_box_or_mask_validation_metric_is_refused(self):
        manifest = self.turntable(lambda item: item["validation_metrics_box"].update(mAP50=None))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)
        manifest = self.turntable(lambda item: item["validation_metrics_mask"].update(mAP50_95=""))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_config_digest_mismatch_is_a_hard_error(self):
        manifest = self.turntable(lambda item: item.update(experiment_config_sha256="0" * 64))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_resolved_config_and_checkpoint_mismatch_are_hard_errors(self):
        manifest = self.turntable(lambda item: item.update(resolved_config_sha256="0" * 64))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)
        manifest = self.turntable(lambda item: item.update(best_checkpoint_sha256="0" * 64))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)


    def test_non_pinned_training_seed_is_refused(self):
        manifest = self.turntable(lambda item: item.update(training_seed=1))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_lambda_ord_is_predeclared_and_never_re_tuned(self):
        manifest = self.turntable(lambda item: item["ordinal"].update(lambda_ord=0.3))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)
        manifest = self.turntable(lambda item: item["ordinal"].update(loss="a different ordinal loss"))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)
        self.fixture["config"]["ordinal"]["lambda_ord"] = 0.25
        with self.assertRaises(RuntimeError):
            self.evidence()

    def test_ordinal_criterion_must_have_run_with_a_recorded_scale(self):
        manifest = self.turntable(lambda item: item["ordinal_diagnostics"].update(criterion_class=None))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)
        # ``summarize_scale`` reports an empty history when no step carried instances; both
        # the empty dict and the ``None`` of a never-scored term must be refused.
        for empty in ([], {}, None):
            manifest = self.turntable(lambda item, value=empty: item["ordinal_diagnostics"].update(
                scale_history=value))
            with self.assertRaises(RuntimeError):
                self.evidence(manifest)

    def test_ordinal_loss_must_be_part_of_the_recorded_loss_vector(self):
        manifest = self.turntable(lambda item: item.update(loss_names=["box_loss", "seg_loss"]))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_a_second_tensorboard_writer_is_refused(self):
        manifest = self.turntable(lambda item: item["tensorboard"]["builtin_ultralytics_integration"]
                                  .update(neutralized=False))
        with self.assertRaises(RuntimeError):
            self.evidence(manifest)

    def test_run_artifacts_are_reverified_instead_of_trusted(self):
        write_results_csv(self.fixture["run"] / "results.csv",
                          columns=tuple(name for name in RESULT_COLUMNS if name != "train/ordinal_loss"))
        with self.assertRaises(RuntimeError):
            self.evidence()
        write_results_csv(self.fixture["run"] / "results.csv")
        shutil.rmtree(self.fixture["run"] / TENSORBOARD_DIRNAME)
        tensorboard_roundtrip(self.fixture["run"] / TENSORBOARD_DIRNAME, tags=("epoch", "train/ordinal_loss"))
        with self.assertRaises(RuntimeError):
            self.evidence()


def _digests(root):
    """sha256 of every file under ``root``; proves the analysis only reads its inputs."""
    return {str(path.relative_to(root)): a.digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


class AnalyzePipelineTest(SyntheticFixtureTestCase):
    """End-to-end E02 analysis on the synthetic run: report, figures, E01 isolation."""

    def test_analysis_runs_end_to_end_on_the_frozen_split(self):
        write_predictions(self.fixture)
        e01_before, run_before = _digests(self.fixture["e01_run"]), _digests(self.fixture["run"])
        report = self.analyze()
        self.assertEqual(report["schema_version"], "e02-analysis-v1")
        self.assertEqual(report["experiment_id"], EXPERIMENT)
        self.assertEqual(report["split"], "val")
        self.assertEqual(report["run_status"], STATUS)
        self.assertTrue(report["analysis_level"].startswith("E02 wrapper over the frozen E01 analysis primitives"))
        self.assertEqual(report["analysis_classification"], "official_frozen_operating_point")
        self.assertEqual(report["analysis_script"], "scripts/24_e02_analyze_results.py")
        self.assertEqual(report["test_guided_threshold_tuning"], False)
        self.assertEqual(report["sensitivity_only"], False)
        self.assertEqual(report["box_metrics"], BOX_METRICS)
        self.assertEqual(report["mask_metrics"], MASK_METRICS)
        self.assertEqual(report["prediction_export_conf"], 0.001)
        self.assertEqual(report["training_git_commit"], "c" * 40)
        self.assertEqual(report["e01_immutability"]["status"], "verified")
        self.assertEqual(report["run_isolation"]["isolated"], True)
        self.assertTrue(report["provenance_status"].startswith("verified"))
        # every E02 input is read-only: manifest, results.csv, exporter CSV and TensorBoard survive
        self.assertEqual(_digests(self.fixture["e01_run"]), e01_before)
        self.assertEqual({name: digest for name, digest in _digests(self.fixture["run"]).items()
                          if name in run_before}, run_before)
        self.assertFalse((self.fixture["e01_run"] / "analysis").exists())

    def test_output_directory_is_versioned_and_inside_the_e02_run(self):
        write_predictions(self.fixture)
        report = self.analyze()
        output = Path(report["analysis_directory"])
        self.assertEqual(output.name, f"val_conf_0p65_ordinal_v{analyzer.ANALYSIS_VERSION}")
        self.assertEqual(output.parent, self.fixture["run"] / "analysis")
        self.assertFalse(str(output).startswith(str(self.fixture["e01_run"])))
        self.assertTrue((output / "analysis_val.json").is_file())

    def test_an_existing_analysis_is_never_silently_overwritten(self):
        write_predictions(self.fixture)
        first = self.analyze()
        existing = Path(first["analysis_directory"]) / "analysis_val.json"
        digest_before = a.digest(existing)
        with self.assertRaises(FileExistsError):
            self.analyze()
        self.assertEqual(a.digest(existing), digest_before)

    def test_the_final_test_split_is_locked(self):
        write_predictions(self.fixture)
        with self.assertRaises(PermissionError):
            self.analyze(split="test")
        self.assertFalse((self.fixture["run"] / "analysis").exists())


    def test_the_operating_point_is_frozen_at_the_e01_confidence(self):
        write_predictions(self.fixture)
        with self.assertRaises(PermissionError):
            self.analyze(operating_confidence=0.5)
        report = self.analyze()
        self.assertEqual(report["operating_confidence_threshold"], 0.65)
        self.assertEqual(report["threshold_selection_split"], "val")
        self.assertEqual(report["official_operating_point"], True)
        self.assertEqual(report["operating_point"]["exported_predictions"], 2)
        self.assertEqual(report["operating_point"]["operational_predictions"], 1)
        self.assertEqual(report["operating_point"]["predictions_below_threshold_excluded"], 1)
        self.assertEqual(report["operating_point"]["prediction_export_modified"], False)
        self.assertEqual(report["export_detail"], {"frozen_split_images": 1, "images_with_any_export": 1,
                                                   "images_without_prediction": 0,
                                                   "images_with_operational_prediction": 1})
        self.assertIn("class-independent greedy box IoU>=0.5", report["matching"])

    def test_matched_instances_and_figures_are_written(self):
        write_predictions(self.fixture)
        report = self.analyze()
        output = Path(report["analysis_directory"])
        self.assertEqual(report["maturity"]["matched_instances"], 1)
        self.assertEqual(report["maturity"]["missed_detections"], 0)
        self.assertEqual(report["maturity"]["false_positives"], 0)
        self.assertEqual(report["maturity"]["gt_rows_pred_columns"], [[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.assertEqual(report["maturity"]["correct"], 1)
        rows = a.csv_rows(output / "matched_predictions_val.csv")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["image_id"], SPLIT_ROW["filename"])
        self.assertEqual(rows[0]["pred_instance_id"], "p0000")
        self.assertEqual(rows[0]["gt_instance_id"], "0")
        self.assertEqual(rows[0]["matched"], "true")
        self.assertEqual(rows[0]["failure_type"], "")
        self.assertEqual(rows[0]["human_review"], "not flagged")
        self.assertEqual(rows[0]["source_group_id"], SPLIT_ROW["source_group_id"])
        self.assertAlmostEqual(float(rows[0]["box_iou"]), 1.0, places=6)
        for figure in ("training_losses_val.png", "validation_metrics_val.png",
                       "maturity_confusion_val.png", "maturity_errors_val.png"):
            self.assertTrue((output / "figures" / figure).is_file(), figure)

    def test_adjacent_maturity_confusion_is_flagged_for_human_review(self):
        write_predictions(self.fixture, rows=((0.9, 1),))
        report = self.analyze()
        self.assertEqual(report["maturity"]["matched_instances"], 1)
        self.assertEqual(report["maturity"]["adjacent"], 1)
        self.assertEqual(report["maturity"]["correct"], 0)
        self.assertEqual(report["failure_case_counts"]["highest_confidence_wrong_class"], 1)
        cases = Path(report["analysis_directory"]) / "failure_cases/highest_confidence_wrong_class"
        self.assertTrue((cases / "00_synthetic.jpg").is_file())
        record = json.loads((cases / "00_synthetic.json").read_text(encoding="utf-8"))
        self.assertEqual(record["human_review"], "Needs Human Review")
        self.assertEqual(record["pred_instance_id"], "p0000")

    def test_a_missed_fruit_is_recorded_without_a_prediction(self):
        write_predictions(self.fixture, rows=())
        report = self.analyze()
        self.assertEqual(report["maturity"]["matched_instances"], 0)
        self.assertEqual(report["maturity"]["missed_detections"], 1)
        self.assertEqual(report["export_detail"]["images_with_any_export"], 0)
        self.assertEqual(report["export_detail"]["images_without_prediction"], 1)
        self.assertEqual(report["failure_case_counts"]["top_false_negatives"], 1)
        rows = a.csv_rows(Path(report["analysis_directory"]) / "matched_predictions_val.csv")
        self.assertEqual([(row["gt_instance_id"], row["pred_instance_id"], row["matched"]) for row in rows],
                         [("0", "", "false")])


    def test_ordinal_evidence_and_tensorboard_contract_are_surfaced(self):
        write_predictions(self.fixture)
        report = self.analyze()
        ordinal = report["ordinal"]
        self.assertEqual(ordinal["lambda_ord"], 0.5)
        self.assertEqual(ordinal["loss_definition"], self.fixture["config"]["ordinal"]["loss"])
        self.assertEqual(ordinal["loss_items"], list(LOSS_NAMES))
        self.assertEqual(ordinal["criterion_class"], "OrdinalSegmentationLoss")
        self.assertEqual(ordinal["assigner_calls"], 101)
        self.assertEqual(ordinal["criterion_steps"], 202)
        self.assertEqual(ordinal["scale_history"], SCALE_HISTORY)
        self.assertIn("NOT tuned", ordinal["scale_history"]["scope"])
        self.assertEqual(sorted(ordinal["recorded_ordinal_curves"]), ["train/ordinal_loss", "val/ordinal_loss"])
        self.assertEqual(ordinal["recorded_ordinal_curves"]["train/ordinal_loss"]["points"], 3)
        self.assertIn("predeclared", ordinal["note"])
        tensorboard = report["tensorboard"]
        self.assertEqual(tensorboard["missing_tags"], [])
        self.assertEqual(tensorboard["contract"]["ordinal_tag"], "ordinal_loss")
        self.assertIn("train/ordinal_loss", tensorboard["contract"]["required_tags"])
        self.assertEqual(tensorboard["builtin_ultralytics_integration"],
                         {"neutralized": True, "builtin_callbacks_registered": 1})

    def test_analysis_refuses_a_run_without_a_recorded_ordinal_loss(self):
        write_results_csv(self.fixture["run"] / "results.csv",
                          columns=tuple(name for name in RESULT_COLUMNS if name != "train/ordinal_loss"))
        write_predictions(self.fixture)
        # both the run-artifact gate and the report builder refuse the missing ordinal curve
        with self.assertRaises(RuntimeError):
            self.analyze()
        self.assertFalse((self.fixture["run"] / "analysis").exists())

    def test_soft_provenance_warnings_never_weaken_the_hard_verdict(self):
        write_predictions(self.fixture)
        report = self.analyze()
        self.assertEqual(report["provenance_status"], "verified_with_soft_warnings")
        self.assertTrue(report["provenance_warnings"])  # no raw archive inside the synthetic repo
        self.assertEqual(report["provenance_detail"]["raw_source"]["mode"], "original_zip")
        self.assertEqual(report["provenance_detail"]["provenance_mode"], "warning_only_E01_style_checks")
        self.assertEqual(report["e02_gates"]["status"], STATUS)
        self.assertEqual(report["input_sha256"]["results_csv"], a.digest(self.fixture["run"] / "results.csv"))
        self.assertEqual(report["input_sha256"]["run_manifest"],
                         a.digest(self.fixture["run"] / "run_manifest.yaml"))
        self.assertEqual(report["input_sha256"]["prediction_csv"],
                         a.digest(self.fixture["run"] / "predictions/predictions_val.csv"))
        self.assertEqual(report["input_sha256"]["resolved_train_config"],
                         a.digest(self.fixture["run"] / "resolved_train_config.yaml"))
        self.assertEqual(report["input_sha256"]["analysis_script"], a.digest(Path(analyzer.__file__)))


    def test_e01_comparison_uses_the_frozen_reference_only(self):
        write_predictions(self.fixture)
        report = self.analyze()
        comparison = report["e01_comparison"]
        self.assertEqual(comparison["e01_frozen_results"], FROZEN_E01_RESULTS)
        self.assertEqual(comparison["e01_reference_config"],
                         "configs/experiments/e02_yolo11n_seg_ordinal.yaml::e01_reference")
        self.assertEqual(comparison["e01_training_commit"], "a" * 40)
        self.assertEqual(comparison["e01_analysis_commit"], "b" * 40)
        self.assertEqual(set(comparison["expected_keys"]), set(FROZEN_E01_RESULTS))
        operational = comparison["e02_operational"]
        self.assertEqual(operational["operational_confidence"], 0.65)
        self.assertEqual(operational["gt_instances"], 1)
        self.assertEqual(operational["matched"], 1)
        self.assertEqual(operational["missed"], 0)
        self.assertEqual(operational["false_positives"], 0)
        self.assertEqual(operational["box_mAP50"], BOX_METRICS["mAP50"])
        self.assertEqual(operational["mask_mAP50"], MASK_METRICS["mAP50"])
        self.assertEqual(operational["maturity_percent"]["immature"], 100.0)
        self.assertIsNone(operational["maturity_percent"]["mature"])
        self.assertEqual(comparison["delta_e02_minus_e01"]["matched"], -89.0)
        self.assertEqual(comparison["delta_e02_minus_e01"]["box_mAP50"], 0.01)
        self.assertEqual(comparison["not_reproduced"], [])
        self.assertIn("never recomputed", comparison["note"])

    def test_compact_digest_keeps_every_audit_fact(self):
        write_predictions(self.fixture)
        report = self.analyze()
        digest = analyzer.compact(report)
        self.assertEqual(digest["schema_version"], "e02-analysis-v1")
        self.assertEqual(digest["experiment_id"], EXPERIMENT)
        self.assertEqual(digest["run_status"], STATUS)
        self.assertEqual(digest["split"], "val")
        self.assertEqual(digest["analysis_directory"], report["analysis_directory"])
        self.assertEqual(digest["operating_confidence_threshold"], 0.65)
        self.assertEqual(digest["analysis_classification"], "official_frozen_operating_point")
        self.assertEqual(digest["provenance_status"], "verified_with_soft_warnings")
        self.assertEqual(digest["provenance_warnings"], report["provenance_warnings"])
        self.assertEqual(digest["box"], {"precision": 0.90, "recall": 0.80, "mAP50": 0.97, "mAP50_95": 0.92})
        self.assertEqual(digest["mask"], {"precision": 0.89, "recall": 0.79, "mAP50": 0.96, "mAP50_95": 0.91})
        self.assertEqual(digest["e01_immutability"], "verified")
        self.assertEqual(digest["run_isolation"], True)
        self.assertEqual(digest["tensorboard_missing_tags"], [])
        self.assertEqual(digest["final_test_status"], runner.FINAL_TEST_POLICY)
        self.assertEqual(digest["ordinal"]["lambda_ord"], 0.5)
        self.assertEqual(sorted(digest["ordinal"]["curves"]), ["train/ordinal_loss", "val/ordinal_loss"])
        self.assertEqual(digest["operational"]["operational_confidence"], 0.65)
        self.assertIn("box_mAP50", digest["delta_e02_minus_e01"])


class TensorBoardEvidenceTest(unittest.TestCase):
    """The canonical E02 dashboard is proved with a real writer round-trip, no training."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="e02-tensorboard-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_the_contract_is_read_from_the_e02_logger(self):
        contract = analyzer.tensorboard_contract()
        self.assertEqual(contract["required_tags"], required_tag_list())
        self.assertEqual(contract["writer"], "torch.utils.tensorboard.SummaryWriter")
        self.assertEqual(contract["ordinal_tag"], "ordinal_loss")
        self.assertEqual(sorted(contract["groups"]), ["metrics", "optimization", "train", "validation"])
        self.assertIn("train/ordinal_loss", contract["groups"]["train"])
        self.assertIn("val/ordinal_loss", contract["groups"]["validation"])
        self.assertNotIn("ordinal_loss", " ".join(contract["groups"]["metrics"]))

    def test_the_analyzer_never_retypes_the_tensorboard_tags(self):
        source = Path(analyzer.__file__).read_text(encoding="utf-8")
        for literal in ("metrics/mAP50(B)", "metrics/mAP50-95(M)", "perf/gpu_memory_gb", "perf/elapsed_seconds",
                        "train/box_loss", "train/ordinal_loss", "val/box_loss"):
            self.assertNotIn(literal, source)

    def test_roundtrip_writes_and_reads_back_every_required_tag(self):
        log_dir = self.root / "runs" / EXPERIMENT / "seed_0" / TENSORBOARD_DIRNAME
        result = tensorboard_roundtrip(log_dir)
        self.assertTrue(result["ok"])
        self.assertEqual(set(result["observed_tags"]), set(required_tag_list()))
        self.assertEqual(result["missing_tags"], [])
        self.assertEqual(result["tags_written"], len(required_tag_list()))
        self.assertIn("EventAccumulator", result["read_back_backend"])
        self.assertTrue(result["event_files"])
        self.assertTrue(all(name.startswith("events.out.tfevents") for name in result["event_files"]))

    def test_the_writer_only_touches_its_own_log_directory(self):
        run = self.root / "runs" / EXPERIMENT / "seed_0"
        outside = self.root / "runs/e01_yolo11n_seg/seed_0"
        outside.mkdir(parents=True)
        (outside / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        before = _digests(outside)
        pre_existing = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
        result = tensorboard_roundtrip(run / TENSORBOARD_DIRNAME)
        self.assertTrue(result["ok"])
        self.assertEqual(_digests(outside), before)
        created = ({path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
                   - pre_existing)
        self.assertTrue(created)
        expected_prefix = f"runs/{EXPERIMENT}/seed_0/{TENSORBOARD_DIRNAME}/"
        self.assertTrue(all(name.startswith(expected_prefix) for name in created), created)

    def test_a_missing_backend_is_a_loud_failure(self):
        with patch.dict(sys.modules, {"torch.utils.tensorboard": None}):
            with self.assertRaises(RuntimeError):
                default_writer_factory(self.root / "no-backend")

    def test_the_builtin_ultralytics_writer_is_neutralized_off(self):
        from ultralytics.utils.callbacks import tensorboard as builtin
        original_callbacks = getattr(builtin, "callbacks", None)
        original_writer = getattr(builtin, "SummaryWriter", None)
        self.addCleanup(setattr, builtin, "callbacks", original_callbacks)
        self.addCleanup(setattr, builtin, "SummaryWriter", original_writer)
        result = neutralize_builtin_ultralytics_tensorboard()
        self.assertTrue(result["neutralized"])
        self.assertEqual(builtin.callbacks, {})
        self.assertIsNone(builtin.SummaryWriter)


if __name__ == "__main__":
    unittest.main()
