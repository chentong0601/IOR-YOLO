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
            self.assertEqual(len(list((run/"figures").glob("*.png"))),6)

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
            with patch.object(a,"ROOT",root), patch.object(a,"provenance",return_value=[]):
                report=a.analyze(run,data)
            self.assertEqual(report["maturity"]["adjacent"],1)
            self.assertEqual(report["maturity"]["missed_detections"],0)
            self.assertTrue((run/"analysis/matched_predictions_val.csv").is_file())
            self.assertTrue((run/"figures/maturity_confusion_val.png").is_file())
            with self.assertRaises(FileExistsError):
                with patch.object(a,"ROOT",root), patch.object(a,"provenance",return_value=[]):
                    a.analyze(run,data)

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
