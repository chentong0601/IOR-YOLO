"""Synthetic-only analysis/config checks; no formal evaluation or model inference."""

import csv
import importlib.util
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def module(number, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{number}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


a = module("17_analyze_e01_results.py", "e01_analyzer_test")
r = module("16_resolve_e01_config.py", "e01_resolver_test")
runner = module("15_e01_run.py", "e01_runner_test_lock")


def gt(klass, id_, box=(0, 0, 10, 10)):
    return {"gt_class": klass, "gt_instance_id": id_, "box": list(box), "image_id": "synthetic.jpg"}


def pred(klass, id_, box=(0, 0, 10, 10), confidence=0.9):
    return {"pred_class": klass, "pred_instance_id": id_, "box": list(box),
            "confidence": confidence, "image_id": "synthetic.jpg"}


ZIP_SHA = "049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce"
SPLIT_ROW = {"filename": "synthetic.jpg", "final_split": "val",
             "source_group_id": "sg-test", "split_guard_cluster_id": "gc-test"}
RESULT_FIELDS = ["epoch", "train/box_loss", "metrics/precision(B)", "metrics/recall(B)",
                 "metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(M)",
                 "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"]


def source_fixture(root, *, mode="zip", detections=((0.9, (2, 2, 14, 14), 1),)):
    """Disposable synthetic repository + formal-run fixture; no real experiment.

    mode="zip"      -> the frozen archive is present locally (original ZIP provenance).
    mode="unpacked" -> byte-verified unpacked directory provenance; no local ZIP.
    Returns ``(run_directory, data_root, manifest)``.
    """
    run = root / "runs/e01_yolo11n_seg/seed_0"
    data = root / "processed"
    for path in (run / "predictions", run / "weights", root / "data/manifests", root / "configs/data",
                 root / "configs/experiments", data / "images/val", data / "labels/val"):
        path.mkdir(parents=True, exist_ok=True)
    pool = root / "data/manifests/d2_experiment_pool_frozen.csv"
    pool.write_text("filename,source_group_id\nsynthetic.jpg,sg-test\n", encoding="utf-8")
    split = root / "data/manifests/d2_split_frozen.csv"
    with split.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(SPLIT_ROW), lineterminator="\n")
        writer.writeheader()
        writer.writerow(SPLIT_ROW)
    experiment = root / "configs/experiments/e01_yolo11n_seg.yaml"
    experiment.write_text("model: yolo11n-seg.pt\n", encoding="utf-8")
    protocol = root / a.FROZEN_PROTOCOL_RELATIVE
    protocol.write_text(yaml.safe_dump({"zip_filename": "dataset-20260508.zip", "zip_sha256": ZIP_SHA}),
                        encoding="utf-8")
    files_manifest = root / a.FILES_SHA256_RELATIVE
    files_manifest.write_text("relative_path,sha256\nimages/val/synthetic.jpg,00\n", encoding="utf-8")
    identity_path = root / a.UNPACKED_IDENTITY_RELATIVE
    identity_path.write_text(json.dumps({"schema_version": a.UNPACKED_IDENTITY_SCHEMA,
                                         "source_zip_filename": "dataset-20260508.zip",
                                         "source_zip_sha256": ZIP_SHA, "expected_image_files": 1,
                                         "files_sha256_manifest_sha256": a.digest(files_manifest)}),
                             encoding="utf-8")
    Image.new("RGB", (20, 20), (80, 80, 80)).save(data / "images/val/synthetic.jpg")
    (data / "labels/val/synthetic.txt").write_text("0 0.1 0.1 0.7 0.1 0.7 0.7 0.1 0.7\n", encoding="utf-8")
    with (run / "results.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(dict(zip(RESULT_FIELDS, [1, .4, .7, .6, .5, .3, .6, .5, .4, .2])))
    (run / "weights/best.pt").write_bytes(b"synthetic checkpoint bytes")
    (run / "resolved_train_config.yaml").write_text("epochs: 100\n", encoding="utf-8")
    a.csv_write(run / "predictions/predictions_val.csv", a.PRED_FIELDS,
                [{"image_id": "synthetic.jpg", "pred_instance_id": f"p{index:04d}", "pred_class": klass,
                  "confidence": confidence, "box": json.dumps(list(box)), "mask_reference": "",
                  "source_group_id": "sg-test", "split_guard_cluster_id": "gc-test", "split": "val"}
                 for index, (confidence, box, klass) in enumerate(detections)])
    metrics = {"mAP50": .97, "mAP50_95": .94}
    manifest = {"run_id": "synthetic-analysis-fixture", "status": "trained_not_final_tested",
                "git_commit": "71357153fb19a7630e69d9fc9cbb243e5d2cd22d", "python": "3.11",
                "pytorch": "2.5.1", "ultralytics": "8.3.220", "training_seed": 0,
                "experiment_relevant_git_dirty": False, "dataset_zip_sha256": ZIP_SHA,
                "raw_source_kind": "original_zip", "raw_identity_status": "VERIFIED",
                "validation_metrics_box": metrics, "validation_metrics_mask": metrics,
                "frozen_pool_sha256": a.digest(pool), "frozen_split_sha256": a.digest(split),
                "protocol_yaml_sha256": a.digest(protocol), "experiment_config_sha256": a.digest(experiment),
                "best_checkpoint_sha256": a.digest(run / "weights/best.pt"),
                "resolved_config_sha256": a.digest(run / "resolved_train_config.yaml")}
    if mode == "unpacked":
        manifest.update(raw_source_kind="unpacked_directory", raw_source_path="/kaggle/input/d2-unpacked",
                        raw_identity_evidence_sha256=a.digest(identity_path),
                        files_sha256_manifest_sha256=a.digest(files_manifest))
    elif mode == "zip":
        archive = root / a.SOURCE_ZIP_RELATIVE
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"synthetic frozen archive bytes")
        manifest.update(raw_source_path=str(archive), dataset_zip_sha256=a.digest(archive))
    else:
        raise ValueError("unknown fixture mode")
    (run / "run_manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return run, data, manifest


def reporting(callable_, *args, **kwargs):
    """Run a provenance helper while swallowing the expected UserWarnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return callable_(*args, **kwargs)


class E01AnalysisTests(unittest.TestCase):
    def test_perfect_adjacent_severe_and_per_class(self):
        rows = a.match_image([gt(0, "g0"), gt(1, "g1", (20,0,30,10)),
                              gt(2, "g2", (40,0,50,10))],
                             [pred(0, "p0"), pred(2, "p1", (20,0,30,10)),
                              pred(0, "p2", (40,0,50,10))])
        summary = a.maturity_summary(rows)
        self.assertEqual(summary["gt_rows_pred_columns"], [[1,0,0], [0,0,1], [1,0,0]])
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["adjacent"], 1)
        self.assertEqual(summary["severe"], 1)
        self.assertEqual(summary["MASE_stage_matched_only"], 1)
        self.assertAlmostEqual(summary["off_by_one_accuracy_matched_only"], 2/3)
        self.assertEqual(summary["per_gt_class"][a.CLASSES[0]]["correct"], 1)
        self.assertEqual(summary["per_gt_class"][a.CLASSES[1]]["correct"], 0)
        self.assertEqual(a.maturity_summary(a.match_image([gt(2,"a")],[pred(2,"b")]))["MASE_stage_matched_only"], 0)

    def test_miss_fp_duplicate_prediction_and_deterministic_tie(self):
        truth = [gt(0,"g0"),gt(1,"g1",(20,0,30,10))]
        detections = [pred(0,"p0"),pred(2,"p1",confidence=.8),pred(2,"p2",(80,0,90,10))]
        rows = a.match_image(truth,detections)
        summary = a.maturity_summary(rows)
        self.assertEqual((summary["matched_instances"],summary["missed_detections"],summary["false_positives"]),(1,1,2))
        self.assertEqual(summary["gt_rows_pred_columns"],[[1,0,0],[0,0,0],[0,0,0]])
        self.assertEqual({x["category"] for x in a.failures(rows)}, {"", "F1 Missed Fruit", "F2 False Positive"})
        self.assertEqual(a.match_image(truth,detections),a.match_image(truth,detections))
        self.assertEqual([x["pred"]["pred_instance_id"] for x in rows if x["kind"]=="matched"],["p0"])
        self.assertIsNone(a.maturity_summary(a.match_image([gt(0,"g")],[]))["MASE_stage_matched_only"])

    def test_mask_and_failure_selection(self):
        square = [[.1,.1],[.7,.1],[.7,.7],[.1,.7]]
        self.assertEqual(a.mask_iou(square,square,20,20), 1)
        rows = a.match_image([{**gt(0,"g"),"polygon":square,"width":20,"height":20}],
                             [{**pred(2,"p"),"polygon":square}])
        self.assertEqual(a.failures(rows)[0]["category"],"F4 Severe Maturity Confusion")
        cases = a.select_cases(a.failures(rows))
        self.assertEqual(len(cases["severe_0_to_2"]),1)
        self.assertEqual(len(cases["lowest_mask_iou"]),1)

    def test_result_csv_parser_uses_distinct_box_and_mask_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/"results.csv"
            cols=["epoch","train/box_loss","metrics/precision(B)","metrics/recall(B)",
                  "metrics/mAP50(B)","metrics/mAP50-95(B)","metrics/precision(M)",
                  "metrics/recall(M)","metrics/mAP50(M)","metrics/mAP50-95(M)"]
            with file.open("w",newline="") as out:
                writer=csv.DictWriter(out,fieldnames=cols,lineterminator="\n")
                writer.writeheader()
                writer.writerow(dict(zip(cols,[1,.4,.7,.6,.5,.3,.6,.5,.4,.2])))
            parsed=a.parse_training_csv(file)
            self.assertEqual(parsed["last_epoch_validation"]["box"]["mAP50_95"],.3)
            self.assertEqual(parsed["last_epoch_validation"]["mask"]["mAP50_95"],.2)
            self.assertEqual(parsed["loss_curves"]["train/box_loss"],[.4])

    def test_test_lock_on_every_entrypoint(self):
        with self.assertRaises(PermissionError):
            a.checked_split("test",False,{"status":"final_test_completed","test_metrics_mask":{"mAP50":.2}})
        with self.assertRaises(PermissionError):
            a.checked_split("test",True,{"status":"trained_not_final_tested"})
        with self.assertRaises(PermissionError):
            runner.evaluate("test")  # must refuse BEFORE any preflight or CUDA interaction
        with self.assertRaises(ValueError):
            a.checked_split("val",True,{})
        a.checked_split("val",False,{})

    def test_provenance_warns_on_missing_and_mismatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/"data/manifests").mkdir(parents=True)
            (root/"data/manifests/d2_experiment_pool_frozen.csv").write_text("data",encoding="utf-8")
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                issues=a.provenance({"frozen_pool_sha256":"wrong"},root,repo_root=root)
            self.assertTrue(any("hash mismatch: frozen_pool" in item for item in issues))
            self.assertTrue(any("missing git_commit" in item for item in issues))
            self.assertTrue(any("E01 provenance" in str(w.message) for w in captured))

    def test_provenance_complete_synthetic_fixture(self):
        # The fixture is temporary; no synthetic scientific result is persisted.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"repo"
            run=root/"runs/seed_0"
            paths={
                "dataset_zip_sha256":root/"data/raw/multistage_apple_v4/dataset-20260508.zip",
                "frozen_pool_sha256":root/"data/manifests/d2_experiment_pool_frozen.csv",
                "frozen_split_sha256":root/"data/manifests/d2_split_frozen.csv",
                "protocol_yaml_sha256":root/"configs/data/d2_frozen_protocol.yaml",
                "experiment_config_sha256":root/"configs/experiments/e01_yolo11n_seg.yaml",
                "best_checkpoint_sha256":run/"weights/best.pt",
                "resolved_config_sha256":run/"resolved_train_config.yaml",
            }
            record={"run_id":"synthetic-test-only","git_commit":"012345", "git_dirty":False,
                    "experiment_relevant_git_dirty":False,
                    "python":"3.11","pytorch":"2.5.1","ultralytics":"8.3.220","training_seed":0}
            for field,path in paths.items():
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b"synthetic test bytes")
                record[field]=a.digest(path)
            self.assertEqual(a.provenance(record,run,repo_root=root),[])

    def test_synthetic_figure_generation_is_confined_to_temp(self):
        report={"split":"val", "training":{"loss_curves":{"train/box_loss":[.6,.4]},
                   "validation_metric_curves":{"metrics/mAP50(B)":[.1,.2]}},
                "maturity":{"gt_rows_pred_columns":[[1,0,0],[0,1,0],[0,0,1]],
                            "correct":3,"adjacent":0,"severe":0,
                            "missed_detections":0,"false_positives":0},
                "box_metrics":{"per_class":{"0":{"name":"immature", "mAP50_95":.2}}},
                "mask_metrics":{"per_class":{"0":{"name":"immature", "mAP50_95":.1}}}}
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp)/"run"
            run.mkdir()
            a.plot_actual_results(report,run,run/"analysis")
            self.assertEqual(len(list((run/"analysis/figures").glob("*.png"))),6)
            self.assertFalse((run/"figures").exists())

    def test_readonly_synthetic_pipeline_fixture(self):
        # This exercises parser -> matching -> report -> figures within a disposable directory.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"repo"
            run=root/"runs/e01_yolo11n_seg/seed_0"
            data=root/"processed"
            (root/"data/manifests").mkdir(parents=True)
            (run/"predictions").mkdir(parents=True)
            (data/"images/val").mkdir(parents=True)
            (data/"labels/val").mkdir(parents=True)
            with (root/"data/manifests/d2_split_frozen.csv").open("w",newline="") as out:
                writer=csv.DictWriter(out,fieldnames=["filename","final_split","source_group_id","split_guard_cluster_id"],lineterminator="\n")
                writer.writeheader()
                writer.writerow({"filename":"synthetic.jpg","final_split":"val","source_group_id":"sg-test","split_guard_cluster_id":"gc-test"})
            Image.new("RGB",(20,20),(80,80,80)).save(data/"images/val/synthetic.jpg")
            (data/"labels/val/synthetic.txt").write_text("0 0.1 0.1 0.7 0.1 0.7 0.7 0.1 0.7\n",encoding="utf-8")
            fields=["epoch","train/box_loss","metrics/precision(B)","metrics/recall(B)",
                    "metrics/mAP50(B)","metrics/mAP50-95(B)","metrics/precision(M)",
                    "metrics/recall(M)","metrics/mAP50(M)","metrics/mAP50-95(M)"]
            with (run/"results.csv").open("w",newline="") as out:
                writer=csv.DictWriter(out,fieldnames=fields,lineterminator="\n")
                writer.writeheader()
                writer.writerow(dict(zip(fields,[1,.4,.7,.6,.5,.3,.6,.5,.4,.2])))
            a.csv_write(run/"predictions/predictions_val.csv",a.PRED_FIELDS,[{
                "image_id":"synthetic.jpg","pred_instance_id":"p0000","pred_class":1,
                "confidence":.7,"box":json.dumps([2,2,14,14]),"mask_reference":"",
                "source_group_id":"sg-test","split_guard_cluster_id":"gc-test","split":"val"}])
            metrics={"mAP50_95":.3,"per_class":{"0":{"name":"immature", "mAP50_95":.2}}}
            (run/"run_manifest.yaml").write_text(yaml.safe_dump({
                "run_id":"synthetic-test","status":"trained_not_final_tested",
                "validation_metrics_box":metrics,"validation_metrics_mask":metrics}),encoding="utf-8")
            with patch.object(a,"ROOT",root), patch.object(a,"provenance_report",
                                                           return_value=([],{"status":"verified","raw_source":{}})):
                report=a.analyze(run,data)
            self.assertEqual(report["maturity"]["adjacent"],1)
            self.assertEqual(report["maturity"]["missed_detections"],0)
            self.assertTrue((run/"analysis/val_conf_0p65_v2/matched_predictions_val.csv").is_file())
            self.assertTrue((run/"analysis/val_conf_0p65_v2/figures/maturity_confusion_val.png").is_file())
            with self.assertRaises(FileExistsError):
                with patch.object(a,"ROOT",root), patch.object(a,"provenance_report",
                                                               return_value=([],{"status":"verified","raw_source":{}})):
                    a.analyze(run,data)

    def test_provenance_verified_unpacked_directory_without_local_zip(self):
        # The formal Kaggle run used the byte-verified unpacked directory, so the
        # absent local dataset-20260508.zip must NOT be reported as incomplete.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, _, manifest = source_fixture(root, mode="unpacked")
            self.assertFalse((root / a.SOURCE_ZIP_RELATIVE).exists())
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                issues, detail = a.provenance_report(manifest, run, repo_root=root)
            self.assertEqual(issues, [])
            self.assertEqual([str(item.message) for item in captured], [])
            self.assertEqual(detail["status"], "verified")
            self.assertEqual(detail["raw_source"]["mode"], "verified_unpacked_directory")
            self.assertFalse(detail["raw_source"]["local_source_zip_present"])
            self.assertEqual(detail["raw_source"]["dataset_zip_sha256"], ZIP_SHA)
            self.assertEqual(detail["raw_source"]["expected_source_zip_sha256"], ZIP_SHA)
            self.assertNotIn("missing provenance file: dataset-20260508.zip", issues)
            self.assertEqual(a.provenance(manifest, run, repo_root=root), [])

    def test_provenance_unpacked_directory_tampering_still_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, _, manifest = source_fixture(root, mode="unpacked")
            tampered = {
                "provenance hash mismatch: dataset_zip_sha256 (frozen source ZIP)":
                    {"dataset_zip_sha256": "0" * 64},
                "provenance hash mismatch: files_sha256_manifest_sha256":
                    {"files_sha256_manifest_sha256": "0" * 64},
                "provenance hash mismatch: raw_identity_evidence_sha256":
                    {"raw_identity_evidence_sha256": None},
                "missing raw_source_path for verified unpacked directory": {"raw_source_path": ""},
                "unpacked raw source is not identity-verified": {"raw_identity_status": "UNVERIFIED"},
                "missing provenance file: dataset-20260508.zip": {"raw_identity_status": "UNVERIFIED"},
            }
            for message, override in tampered.items():
                with self.subTest(message=message):
                    issues, _ = reporting(a.provenance_report, {**manifest, **override}, run, repo_root=root)
                    self.assertTrue(any(message in item for item in issues), issues)
            identity_path = root / a.UNPACKED_IDENTITY_RELATIVE
            files_manifest = root / a.FILES_SHA256_RELATIVE
            original = identity_path.read_text(encoding="utf-8")
            identity_path.write_text(json.dumps({**json.loads(original), "source_zip_sha256": "1" * 64}),
                                     encoding="utf-8")
            issues, _ = reporting(a.provenance_report, manifest, run, repo_root=root)
            self.assertTrue(any("frozen protocol and unpacked identity disagree" in item for item in issues))
            identity_path.write_text(original, encoding="utf-8")
            files_manifest.unlink()
            issues, _ = reporting(a.provenance_report, manifest, run, repo_root=root)
            self.assertTrue(any("missing provenance file: files_sha256.csv" in item for item in issues))
            identity_path.unlink()
            issues, _ = reporting(a.provenance_report, manifest, run, repo_root=root)
            self.assertTrue(any("missing provenance file: d2_unpacked_identity.json" in item for item in issues))

    def test_provenance_original_zip_source_still_requires_local_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, _, manifest = source_fixture(root, mode="zip")
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                issues, detail = a.provenance_report(manifest, run, repo_root=root)
            self.assertEqual(issues, [])
            self.assertEqual([str(item.message) for item in captured], [])
            self.assertEqual(detail["raw_source"]["mode"], "original_zip")
            self.assertTrue(detail["raw_source"]["local_source_zip_present"])
            issues, _ = reporting(a.provenance_report, {**manifest, "dataset_zip_sha256": "0" * 64},
                                  run, repo_root=root)
            self.assertIn("provenance hash mismatch: dataset_zip_sha256", issues)
            (root / a.SOURCE_ZIP_RELATIVE).unlink()
            issues, _ = reporting(a.provenance_report, manifest, run, repo_root=root)
            self.assertIn("missing provenance file: dataset-20260508.zip", issues)

    def test_operating_threshold_filters_operational_false_positives(self):
        detections = ((0.10, (2, 2, 14, 14), 1), (0.649, (2, 2, 14, 14), 1),
                      (0.65, (2, 2, 14, 14), 1), (0.90, (15, 15, 19, 19), 1))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, data, _ = source_fixture(root, mode="unpacked", detections=detections)
            with patch.object(a, "ROOT", root):
                report = a.analyze(run, data)
            self.assertEqual(report["prediction_export_conf"], 0.001)
            self.assertEqual(report["operating_confidence_threshold"], 0.65)
            self.assertEqual(report["threshold_selection_split"], "val")
            self.assertEqual(report["threshold_selection_basis"],
                             "validation F1 operating point; frozen before final test")
            self.assertFalse(report["test_guided_threshold_tuning"])
            self.assertEqual(report["operating_point"]["exported_predictions"], 4)
            self.assertEqual(report["operating_point"]["operational_predictions"], 2)
            self.assertEqual(report["operating_point"]["predictions_below_threshold_excluded"], 2)
            self.assertEqual(report["maturity"]["matched_instances"], 1)
            self.assertEqual(report["maturity"]["false_positives"], 1)
            self.assertEqual(report["maturity"]["missed_detections"], 0)
            self.assertEqual(report["maturity"]["adjacent"], 1)
            # the export itself is read in full and never rewritten
            self.assertEqual(len(list(a.csv_rows(run / "predictions/predictions_val.csv"))), 4)
            kept, stream = a.filter_predictions({"synthetic.jpg": [
                {"confidence": .649, "box": [2, 2, 14, 14]},
                {"confidence": .65, "box": [2, 2, 14, 14]}]}, a.OPERATING_CONFIDENCE_THRESHOLD)
            self.assertEqual([item["confidence"] for item in kept["synthetic.jpg"]], [.65])
            self.assertEqual(stream["predictions_below_threshold_excluded"], 1)
            # a non-default operating point is allowed for val and gets its own directory
            with patch.object(a, "ROOT", root):
                lower = a.analyze(run, data, operating_confidence=0.2)
            self.assertEqual(lower["operating_point"]["predictions_below_threshold_excluded"], 1)
            self.assertEqual(lower["maturity"]["false_positives"], 2)
            self.assertNotEqual(lower["analysis_directory"], report["analysis_directory"])

    def test_validation_override_is_marked_sensitivity_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, data, _ = source_fixture(root, mode="unpacked")
            with patch.object(a, "ROOT", root):
                official = a.analyze(run, data)
            self.assertTrue(official["official_operating_point"])
            self.assertEqual(official["analysis_classification"], "official_frozen_operating_point")
            self.assertFalse(official["sensitivity_only"])
            self.assertIsNone(official["requested_operating_confidence"])
            self.assertEqual(Path(official["analysis_directory"]).name, "val_conf_0p65_v2")
            with patch.object(a, "ROOT", root):
                with self.assertWarnsRegex(UserWarning, "SENSITIVITY-ONLY"):
                    other = a.analyze(run, data, operating_confidence=0.5)
            self.assertFalse(other["official_operating_point"])
            self.assertEqual(other["analysis_classification"], "sensitivity_only_non_official")
            self.assertTrue(other["sensitivity_only"])
            self.assertEqual(other["requested_operating_confidence"], 0.5)
            self.assertEqual(other["operating_confidence_threshold"], 0.5)
            self.assertEqual(Path(other["analysis_directory"]).name, "val_conf_0p5_v2")
            self.assertIn("SENSITIVITY-ONLY", other["interpretation"])
            self.assertEqual(other["provenance_status"], "verified")

    def test_final_test_threshold_locked_to_validation_value(self):
        self.assertEqual(a.OPERATING_CONFIDENCE_THRESHOLD, 0.65)
        self.assertEqual(a.operating_threshold("val"), 0.65)
        self.assertEqual(a.operating_threshold("val", 0.5), 0.5)
        self.assertEqual(a.operating_threshold("test"), 0.65)
        self.assertEqual(a.operating_threshold("test", 0.65), 0.65)
        for value in (0.5, 0.649, 0.7, 0.9):
            with self.assertRaises(PermissionError):
                a.operating_threshold("test", value)
        for value in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                a.operating_threshold("val", value)
        self.assertEqual(a.analysis_directory(Path("run"), "val", 0.65).name, "val_conf_0p65_v2")
        self.assertEqual(a.analysis_directory(Path("run"), "test", 0.65).name, "test_conf_0p65_v2")

    def test_final_test_analysis_entrypoint_refuses_its_own_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, data, manifest = source_fixture(root, mode="unpacked")
            finished = {**manifest, "status": "final_test_completed",
                        "test_metrics_box": {"mAP50": .97}, "test_metrics_mask": {"mAP50": .96}}
            (run / "run_manifest.yaml").write_text(yaml.safe_dump(finished), encoding="utf-8")
            with patch.object(a, "ROOT", root):
                with self.assertRaises(PermissionError):
                    a.analyze(run, data, split="test", final_test=True, operating_confidence=0.5)
            # the lock fires before any run file is read or any output is created
            self.assertFalse((run / "analysis").exists())

    def test_analysis_output_is_versioned_and_preserves_earlier_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            run, data, manifest = source_fixture(root, mode="unpacked")
            legacy_json = run / "analysis/analysis_val.json"
            legacy_csv = run / "analysis/matched_predictions_val.csv"
            legacy_json.parent.mkdir(parents=True)
            legacy_json.write_text(json.dumps({"status": "exploratory_superseded",
                                               "false_positives": 1418}), encoding="utf-8")
            legacy_csv.write_text("exploratory\n", encoding="utf-8")
            frozen_figure = run / "figures/maturity_confusion_val.png"
            frozen_figure.parent.mkdir(parents=True)
            frozen_figure.write_bytes(b"formal run figure bytes")
            with patch.object(a, "ROOT", root):
                report = a.analyze(run, data)
            versioned = run / "analysis/val_conf_0p65_v2"
            self.assertEqual(Path(report["analysis_directory"]), versioned)
            self.assertEqual(report["analysis_version"], 2)
            self.assertTrue((versioned / "analysis_val.json").is_file())
            self.assertTrue((versioned / "matched_predictions_val.csv").is_file())
            self.assertTrue((versioned / "figures/maturity_confusion_val.png").is_file())
            self.assertEqual(sorted(path.name for path in legacy_json.parent.iterdir()),
                             ["analysis_val.json", "matched_predictions_val.csv", "val_conf_0p65_v2"])
            self.assertEqual(json.loads(legacy_json.read_text(encoding="utf-8"))["status"],
                             "exploratory_superseded")
            self.assertEqual(legacy_csv.read_text(encoding="utf-8"), "exploratory\n")
            self.assertEqual(frozen_figure.read_bytes(), b"formal run figure bytes")
            self.assertEqual(report["superseded_analysis"], str(legacy_json))
            self.assertEqual(report["provenance_status"], "verified")
            self.assertEqual(report["training_git_commit"], manifest["git_commit"])
            self.assertEqual(report["analysis_script_sha256"], a.digest(Path(a.__file__)))
            self.assertEqual(report["analysis_git_commit"], a.git_state()["head"])
            with patch.object(a, "ROOT", root):
                with self.assertRaises(FileExistsError):
                    a.analyze(run, data)
            self.assertEqual(json.loads(legacy_json.read_text(encoding="utf-8"))["status"],
                             "exploratory_superseded")

    def test_template_na_and_auto_resolution(self):
        table=(ROOT.parent/"docs/templates/e01-results-table.md").read_text(encoding="utf-8")
        self.assertIn("| NA | NA | NA | NA |",table)
        cfg=r.resolve()
        self.assertEqual(cfg["defaults"]["optimizer"],"auto")
        self.assertEqual(cfg["defaults"]["batch"],16)
        self.assertEqual(cfg["resolved_pretrain"]["batch"],8)
        self.assertEqual(cfg["optimizer_auto_projection"]["projected_iterations"],1300)
        self.assertEqual(cfg["optimizer_auto_projection"]["projected_optimizer"],"AdamW")
        self.assertEqual(r.resolve(batch_override=4)["resolved_pretrain"]["batch"],4)
        with self.assertRaises(ValueError):
            r.resolve(batch_override=12)


if __name__ == "__main__":
    unittest.main()
