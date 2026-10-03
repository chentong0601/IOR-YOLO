"""E02 ordinal training wiring: no training run, no CUDA and no dataset access.

Covers the injection points (criterion, model, trainer, loss items, dashboard),
the recorded assigner targets and the merge arithmetic of
``L_total = L_original_YOLO + lambda_ord * L_ord``.
"""

import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ior_yolo.losses.ordinal import OrdinalLossConfig, ordinal_component, ordinal_penalty  # noqa: E402
from ior_yolo.trainers import ordinal as wiring  # noqa: E402
from ultralytics.models.yolo.model import YOLO  # noqa: E402
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator  # noqa: E402
from ultralytics.nn.tasks import SegmentationModel  # noqa: E402
from ultralytics.utils.loss import v8SegmentationLoss  # noqa: E402
from ultralytics.utils.tal import TaskAlignedAssigner  # noqa: E402


class InjectionTest(unittest.TestCase):
    def test_classes_live_in_a_stable_importable_module(self):
        # Checkpoints pickle the model by module path; a different module name
        # would make every E02 checkpoint unloadable in the analysis scripts.
        for cls in (wiring.OrdinalSegmentationModel, wiring.OrdinalSegmentationLoss,
                    wiring.OrdinalSegmentationTrainer, wiring.OrdinalYOLO):
            self.assertEqual(cls.__module__, "ior_yolo.trainers.ordinal")

    def test_classes_extend_the_pinned_ultralytics_classes(self):
        self.assertTrue(issubclass(wiring.OrdinalSegmentationModel, SegmentationModel))
        self.assertTrue(issubclass(wiring.OrdinalSegmentationTrainer, SegmentationTrainer))
        self.assertTrue(issubclass(wiring.OrdinalSegmentationLoss, v8SegmentationLoss))
        self.assertTrue(issubclass(wiring.OrdinalYOLO, YOLO))

    def test_yolo_task_map_swaps_only_the_segment_model_and_trainer(self):
        stock, e02 = object.__new__(YOLO), object.__new__(wiring.OrdinalYOLO)
        stock_map, e02_map = stock.task_map, e02.task_map
        self.assertEqual(sorted(stock_map), sorted(e02_map))
        for task in ("classify", "detect", "pose", "obb"):
            self.assertEqual(e02_map[task], stock_map[task])
        self.assertEqual(e02_map["segment"]["model"], wiring.OrdinalSegmentationModel)
        self.assertEqual(e02_map["segment"]["trainer"], wiring.OrdinalSegmentationTrainer)
        self.assertEqual(e02_map["segment"]["validator"], SegmentationValidator)
        self.assertEqual(e02_map["segment"]["validator"], stock_map["segment"]["validator"])
        self.assertEqual(e02_map["segment"]["predictor"], stock_map["segment"]["predictor"])

    def test_model_init_criterion_uses_the_configured_ordinal_settings(self):
        model = object.__new__(wiring.OrdinalSegmentationModel)
        with patch.object(wiring, "OrdinalSegmentationLoss") as factory:
            model.init_criterion()
        factory.assert_called_once()
        self.assertIs(factory.call_args.args[0], model)
        self.assertEqual(factory.call_args.kwargs["loss_config"].lambda_ord, 0.5)
        self.assertEqual(factory.call_args.kwargs["loss_config"].num_stages, 3)

    def test_builtin_tensorboard_is_neutralized_before_the_trainer_base_runs(self):
        source = inspect.getsource(wiring.OrdinalSegmentationTrainer.__init__)
        self.assertLess(source.index("neutralize_builtin_ultralytics_tensorboard()"),
                        source.index("super().__init__("))

    def test_trainer_registers_the_dashboard_callbacks(self):
        source = inspect.getsource(wiring.OrdinalSegmentationTrainer.__init__)
        self.assertIn("E02TensorBoardLogger", source)
        self.assertIn("self.add_callback(event, callback)", source)
        self.assertIn("self.tensorboard.start()", source)


class SettingsTest(unittest.TestCase):
    def test_from_mapping_accepts_the_e02_config_block(self):
        settings = wiring.OrdinalRunSettings.from_mapping(
            {"lambda_ord": 0.5, "num_stages": 3, "stage_order": list(wiring.STAGE_ORDER),
             "log_batch_scalars": True})
        self.assertEqual(settings.lambda_ord, 0.5)
        self.assertEqual(settings.loss_config().lambda_ord, 0.5)
        self.assertEqual(settings.as_dict()["stage_order"], list(wiring.STAGE_ORDER))

    def test_from_mapping_rejects_unknown_missing_or_invalid_values(self):
        with self.assertRaises(ValueError):
            wiring.OrdinalRunSettings.from_mapping({"lambda_ord": 0.5, "num_stages": 3,
                                                    "stage_order": list(wiring.STAGE_ORDER), "extra": 1})
        with self.assertRaises(ValueError):
            wiring.OrdinalRunSettings.from_mapping({"lambda_ord": 0.5})
        with self.assertRaises(ValueError):
            wiring.OrdinalRunSettings.from_mapping({"lambda_ord": 0.5, "num_stages": 3,
                                                    "stage_order": ["mature apple", "semi-mature apple",
                                                                    "immature apple"]})
        with self.assertRaises(TypeError):
            wiring.OrdinalRunSettings.from_mapping(None)

    def test_pop_ordinal_settings_removes_the_key_from_the_ultralytics_overrides(self):
        overrides = {"data": "x.yaml",
                     wiring.ORDINAL_CONFIG_KEY: {"lambda_ord": 0.5, "num_stages": 3,
                                                 "stage_order": list(wiring.STAGE_ORDER)}}
        settings = wiring.pop_ordinal_settings(overrides)
        self.assertEqual(settings.num_stages, 3)
        self.assertEqual(sorted(overrides), ["data"])
        with self.assertRaises(ValueError):
            wiring.pop_ordinal_settings({"data": "x.yaml"})


class RecorderTest(unittest.TestCase):
    def test_recorder_forwards_arguments_and_records_outputs(self):
        calls = []

        def inner(*args, **kwargs):
            calls.append((args, kwargs))
            return ("sentinel", args[0])

        recorder = wiring.AssignerCallRecorder(inner)
        payload = torch.ones(1, 2, 3)
        result = recorder(payload, flag=7)
        self.assertEqual(result[0], "sentinel")
        self.assertEqual(calls, [((payload,), {"flag": 7})])
        self.assertEqual(recorder.calls, 1)
        self.assertIs(recorder.last_args[0], payload)
        self.assertIs(recorder.last_output, result)
        recorder.reset()
        self.assertIsNone(recorder.last_args)
        self.assertIsNone(recorder.last_output)


def build_criterion(batch_size=2, anchors=4, nc=3, reg_max=4, stages=(0, 2, 1)):
    """Build an ``OrdinalSegmentationLoss`` with a synthetic head output and assignment."""
    criterion = object.__new__(wiring.OrdinalSegmentationLoss)
    criterion.loss_config = OrdinalLossConfig()
    criterion.verify_logits = True
    criterion.nc = nc
    criterion.reg_max = reg_max
    criterion.no = reg_max * 4 + nc
    criterion.last_ordinal = None
    criterion.steps = 0
    # Mirrors ``OrdinalSegmentationLoss.__init__``: the engineering-only scale report the
    # run manifest is built from (``MAX_SCALE_RECORDS`` shaving included).
    criterion.scale_records = []
    feats = [torch.randn(batch_size, criterion.no, 2, 2)]
    logits = criterion.recompute_logits((feats, None, None))
    targets = torch.zeros(batch_size, anchors, nc)
    mask = torch.zeros(batch_size, anchors, dtype=torch.bool)
    for index, stage in enumerate(stages):
        targets[0, index, stage] = 1.0
        mask[0, index] = True
    recorder = wiring.AssignerCallRecorder(
        lambda assigned, *_args, **_kwargs: (None, None, targets, mask, None))
    recorder(logits.detach().sigmoid())  # emulate the pinned assigner call
    criterion.assigner = recorder
    return criterion, (feats, None, None), logits, targets, mask


class CriterionMergeTest(unittest.TestCase):
    def test_stock_items_are_preserved_and_the_ordinal_item_is_appended(self):
        criterion, preds, logits, targets, mask = build_criterion()
        expected = float(ordinal_component(logits, targets, mask)["loss"])
        with patch.object(v8SegmentationLoss, "__call__",
                          return_value=(torch.tensor(4.0), torch.tensor([1.0, 2.0, 3.0, 4.0]))):
            total, items = criterion(preds, {})
        self.assertEqual(items.shape, (5,))
        self.assertEqual([float(value) for value in items[:4]], [1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(float(items[4]), 0.5 * expected, places=6)
        self.assertAlmostEqual(float(total), 4.0 + 0.5 * expected * 2, places=6)

    def test_diagnostics_record_the_ordinal_contribution(self):
        criterion, preds, logits, targets, mask = build_criterion()
        with patch.object(v8SegmentationLoss, "__call__", return_value=(torch.tensor(1.0), torch.zeros(4))):
            criterion(preds, {})
        report = criterion.last_ordinal
        self.assertEqual(report["lambda_ord"], 0.5)
        self.assertEqual(report["batch_size"], 2)
        self.assertEqual(report["instances"], 3)
        self.assertEqual(report["step"], 0)
        self.assertGreater(report["penalty_max"], 0.0)
        self.assertAlmostEqual(report["weighted_loss"], 0.5 * report["loss"], places=8)
        self.assertEqual(criterion.steps, 1)

    def test_recorded_tensors_are_released_after_use(self):
        criterion, preds, *_ = build_criterion()
        with patch.object(v8SegmentationLoss, "__call__", return_value=(torch.tensor(1.0), torch.zeros(4))):
            criterion(preds, {})
        self.assertIsNone(criterion.assigner.last_output)
        self.assertEqual(criterion.assigner.calls, 1)

    def test_component_wise_stock_total_gains_the_ordinal_term_exactly_once(self):
        # Pinned 8.3.220 returns ``loss * batch_size`` as the vector (box, seg, cls, dfl) and
        # the trainer optimises ``loss.sum()``; the auxiliary term must appear exactly once.
        criterion, preds, logits, targets, mask = build_criterion()
        expected = float(ordinal_component(logits, targets, mask)["loss"])
        stock = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        with patch.object(v8SegmentationLoss, "__call__", return_value=(stock, stock.detach())):
            total, items = criterion(preds, {})
        self.assertEqual(total.shape, stock.shape)
        self.assertAlmostEqual(float(total.sum().detach()), 10.0 + 0.5 * expected * 2, places=5)
        self.assertAlmostEqual(float(total[0].detach()), 1.0, places=6)  # stock components untouched
        self.assertAlmostEqual(float(total[3].detach()), 4.0, places=6)
        self.assertAlmostEqual(float(total[2].detach()), 3.0 + 0.5 * expected * 2, places=5)
        record = criterion.last_ordinal["scale"]
        self.assertAlmostEqual(record["total_before_ordinal"], 10.0, places=5)
        self.assertAlmostEqual(record["lambda_scaled_ordinal_batch"], 0.5 * expected * 2, places=5)
        self.assertAlmostEqual(record["ordinal_share_of_total"],
                               0.5 * expected * 2 / float(total.sum().detach()), places=5)
        total.sum().backward()  # the merge must stay autograd-safe
        self.assertTrue(torch.equal(stock.grad, torch.ones(4)))

    def test_scale_record_refuses_a_term_applied_once_per_component(self):
        component = {"loss": torch.tensor(0.25), "instances": 3, "quality_mean": 0.5}
        stock = torch.tensor([1.0, 2.0, 3.0, 4.0])
        weighted = 0.5 * component["loss"]  # lambda_ord * L_ord, batch mean
        applied_once = stock.clone()
        applied_once[2] += weighted * 8  # slot 2 of (box, seg, cls, dfl): one single application
        record = wiring.scale_record(step=0, batch_size=8, stock_items=stock, component=component,
                                     total_before=stock, weighted=weighted, total_after=applied_once)
        self.assertAlmostEqual(record["lambda_scaled_ordinal_batch"], 1.0, places=6)
        self.assertAlmostEqual(record["total_after_ordinal"], 11.0, places=6)
        self.assertAlmostEqual(record["ordinal_share_of_total"], 1.0 / 11.0, places=6)
        with self.assertRaises(RuntimeError):  # a broadcasted merge would look like this
            wiring.scale_record(step=0, batch_size=8, stock_items=stock, component=component,
                                total_before=stock, weighted=weighted, total_after=stock + weighted * 8)

    def test_missing_assignment_is_refused(self):
        criterion, preds, *_ = build_criterion()
        criterion.assigner.reset()
        with patch.object(v8SegmentationLoss, "__call__", return_value=(torch.tensor(1.0), torch.zeros(4))):
            with self.assertRaises(RuntimeError):
                criterion(preds, {})

    def test_mismatched_logits_are_refused(self):
        criterion, preds, logits, targets, mask = build_criterion()
        recorder = wiring.AssignerCallRecorder(
            lambda assigned, *_args, **_kwargs: (None, None, targets, mask, None))
        recorder(torch.zeros_like(logits.detach()))  # a head layout change would look like this
        criterion.assigner = recorder
        with patch.object(v8SegmentationLoss, "__call__", return_value=(torch.tensor(1.0), torch.zeros(4))):
            with self.assertRaises(RuntimeError):
                criterion(preds, {})

    def test_unexpected_assigner_output_is_refused(self):
        criterion, preds, logits, targets, mask = build_criterion()
        recorder = wiring.AssignerCallRecorder(lambda assigned, *_args, **_kwargs: (None, None, targets))
        recorder(logits.detach().sigmoid())
        criterion.assigner = recorder
        with patch.object(v8SegmentationLoss, "__call__", return_value=(torch.tensor(1.0), torch.zeros(4))):
            with self.assertRaises(RuntimeError):
                criterion(preds, {})

    def test_recompute_logits_rejects_unknown_head_outputs(self):
        criterion, *_ = build_criterion()
        with self.assertRaises(RuntimeError):
            criterion.recompute_logits(())

    def test_logits_are_recovered_from_the_training_wrapping(self):
        criterion, (feats, _, _), logits, *_ = build_criterion()
        self.assertTrue(torch.equal(criterion.recompute_logits((feats, None, None)), logits))

    def test_logits_are_recovered_from_the_eval_mode_wrapping(self):
        # The validator feeds eval-mode outputs: (detections, (feats, mask_coeffs, proto)).
        criterion, (feats, _, _), logits, *_ = build_criterion()
        eval_mode = (torch.zeros(2, criterion.no + 32, 4),
                     (feats, torch.zeros(2, 32, 4), torch.zeros(2, 32, 8, 8)))
        self.assertTrue(torch.equal(criterion.recompute_logits(eval_mode), logits))

    def test_recompute_logits_rejects_levels_with_the_wrong_channel_count(self):
        criterion, *_ = build_criterion()
        wrong = [torch.zeros(2, criterion.no - 1, 2, 2)]
        with self.assertRaises(RuntimeError):
            criterion.recompute_logits(wrong)
        with self.assertRaises(RuntimeError):
            criterion.recompute_logits((torch.zeros(2, 7, 4), (torch.zeros(1),)))


class PinnedAssignerContractTest(unittest.TestCase):
    """The pinned 8.3.220 assignment contract the ordinal term reads ``w_i`` from.

    ``OrdinalSegmentationLoss.ordinal_term`` reads ``outputs[2]`` as ``target_scores`` and
    ``outputs[3]`` as the foreground mask. That mapping is only correct while the real
    ``TaskAlignedAssigner`` returns ``(target_labels, target_bboxes, target_scores,
    fg_mask.bool(), target_gt_idx)`` and while every matched anchor carries exactly one
    non-zero, quality-scaled target class. These tests run the real pinned assigner on CPU
    with deterministic synthetic anchors, so an Ultralytics change fails here instead of
    silently mis-weighting the auxiliary term.
    """

    @staticmethod
    def assign():
        """Run the real pinned ``TaskAlignedAssigner`` on deterministic synthetic anchors."""
        assigner = TaskAlignedAssigner(topk=10, num_classes=3, alpha=0.5, beta=6.0)
        grid = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij"), dim=-1)
        anchor_points = grid.reshape(-1, 2).float() * 8.0 + 4.0  # (16, 2) pixel centre grid
        gt_bboxes = torch.tensor([[[0.0, 0.0, 16.0, 32.0], [16.0, 0.0, 32.0, 32.0]]])
        # Anchors 0-7 fall inside the first ground truth, 8-15 inside the second; the widening
        # gives every anchor a different IoU, hence a different alignment quality.
        widen = (torch.arange(anchor_points.shape[0]) % 4).float() / 2.0
        pd_bboxes = gt_bboxes[0].repeat_interleave(8, dim=0)
        pd_bboxes[:, 0] -= widen
        pd_bboxes[:, 2] += widen
        pd_bboxes = pd_bboxes.unsqueeze(0)
        pd_scores = torch.linspace(0.1, 0.9, anchor_points.shape[0] * 3).reshape(1, -1, 3)
        gt_labels = torch.tensor([[[1.0], [2.0]]])
        mask_gt = torch.ones(1, 2, 1, dtype=torch.bool)
        return assigner(pd_scores, pd_bboxes, anchor_points, gt_bboxes=gt_bboxes, gt_labels=gt_labels,
                        mask_gt=mask_gt)

    def test_real_assigner_keeps_the_output_order_and_foreground_dtype_the_trainer_reads(self):
        outputs = self.assign()
        self.assertEqual(len(outputs), 5)
        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = outputs
        self.assertEqual(target_scores.shape, (1, 16, 3))
        self.assertEqual(fg_mask.shape, (1, 16))
        # outputs[3] must be the bool foreground mask: target_gt_idx is int64, and a scalar
        # target_scores_sum would not even carry the (B, A) shape the ordinal term validates.
        self.assertIs(fg_mask.dtype, torch.bool)
        self.assertIsNot(target_gt_idx.dtype, torch.bool)
        self.assertEqual(target_labels.shape, (1, 16))
        self.assertEqual(target_bboxes.shape, (1, 16, 4))
        self.assertGreater(int(fg_mask.sum()), 0)  # the contract test must not pass vacuously
        self.assertFalse(bool(target_scores.requires_grad))  # constant targets, no gradient path

    def test_real_matched_anchors_carry_exactly_one_quality_weighted_class(self):
        assigned, _, target_scores, fg_mask, _ = self.assign()
        weights = target_scores.sum(dim=-1)
        mass = (target_scores > 0).sum(dim=-1)
        self.assertEqual(int(fg_mask.sum()), 16)
        self.assertTrue(bool((mass[fg_mask] == 1).all()))  # one class per matched anchor
        self.assertTrue(bool((mass[~fg_mask] == 0).all()))  # nothing outside the matched anchors
        self.assertTrue(bool((weights[fg_mask] > 0).all()))
        self.assertTrue(bool((weights[fg_mask] <= 1.0 + 1e-6).all()))  # normalized alignment quality
        # The pinned target is NOT one-hot: the quality value is strictly below 1 somewhere.
        self.assertIs(target_scores.dtype, torch.float32)
        self.assertTrue(bool((weights[fg_mask] < 1.0).any()))
        # w_i is the mass at the assigned class, so it equals target_scores[i, assigned_class_i]
        # and target_scores[i].sum() at the same time.
        self.assertEqual(target_scores.argmax(dim=-1)[fg_mask].tolist(), assigned[fg_mask].tolist())
        self.assertAlmostEqual(float(weights.sum()), float(target_scores[fg_mask].sum()), places=5)

    def test_ordinal_component_matches_the_closed_form_on_the_real_assignment(self):
        _, _, target_scores, fg_mask, _ = self.assign()
        logits = torch.linspace(-0.5, 0.5, 16 * 3).reshape(1, 16, 3).requires_grad_(True)
        component = ordinal_component(logits, target_scores, fg_mask)
        stages = target_scores.argmax(dim=-1)[fg_mask]
        errors = ordinal_penalty(logits[fg_mask], stages)
        weights = target_scores.sum(dim=-1)[fg_mask]
        stock_denominator = max(float(target_scores.sum()), 1.0)  # pinned target_scores_sum
        self.assertAlmostEqual(component["normalization"], stock_denominator, places=5)
        self.assertAlmostEqual(component["weight_sum"], float(weights.sum()), places=4)
        self.assertAlmostEqual(float(component["loss"]),
                               float((weights * errors).sum()) / max(float(weights.sum()), 1.0), places=6)
        self.assertGreater(float(component["loss"]), 0.0)
        component["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


class TrainerHookTest(unittest.TestCase):
    def test_loss_names_gain_exactly_one_ordinal_item(self):
        stub = object.__new__(wiring.OrdinalSegmentationTrainer)
        stub.loss_names = ()

        def parent(self):
            self.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")
            return "validator"

        with patch.object(SegmentationTrainer, "get_validator", parent):
            validator = wiring.OrdinalSegmentationTrainer.get_validator(stub)
        self.assertEqual(validator, "validator")
        self.assertEqual(stub.loss_names, ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "ordinal_loss"))

    def test_loss_items_are_labelled_with_the_ordinal_key(self):
        # DetectionTrainer.label_loss_items is inherited unchanged; the fifth name
        # is what results.csv and TensorBoard record.
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        stub = object.__new__(wiring.OrdinalSegmentationTrainer)
        stub.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "ordinal_loss")
        labelled = DetectionTrainer.label_loss_items(stub, torch.tensor([1.0, 2.0, 3.0, 4.0, 0.25]))
        self.assertEqual(sorted(labelled), ["train/box_loss", "train/cls_loss", "train/dfl_loss",
                                            "train/ordinal_loss", "train/seg_loss"])
        self.assertEqual(labelled["train/ordinal_loss"], 0.25)

    def test_get_model_builds_the_ordinal_model_with_the_same_call_shape(self):
        stub = object.__new__(wiring.OrdinalSegmentationTrainer)
        stub.data = {"nc": 3, "channels": 3}
        stub.ordinal_settings = wiring.OrdinalRunSettings()
        with patch.object(wiring, "OrdinalSegmentationModel") as factory:
            model = wiring.OrdinalSegmentationTrainer.get_model(stub, cfg={"yaml": "stub"}, verbose=False)
        self.assertIs(model, factory.return_value)
        self.assertEqual(factory.call_args.args[0], {"yaml": "stub"})
        self.assertEqual(factory.call_args.kwargs["nc"], 3)
        self.assertEqual(factory.call_args.kwargs["ch"], 3)
        self.assertEqual(factory.return_value.ordinal_settings.lambda_ord, 0.5)
        factory.return_value.load.assert_not_called()

    def test_get_model_loads_pretrained_weights_when_given(self):
        stub = object.__new__(wiring.OrdinalSegmentationTrainer)
        stub.data = {"nc": 3, "channels": 3}
        stub.ordinal_settings = wiring.OrdinalRunSettings()
        with patch.object(wiring, "OrdinalSegmentationModel") as factory:
            wiring.OrdinalSegmentationTrainer.get_model(stub, cfg="yolo11n-seg.yaml", weights="yolo11n-seg.pt",
                                                        verbose=False)
        factory.return_value.load.assert_called_once_with("yolo11n-seg.pt")

    def test_ordinal_diagnostics_reports_the_configured_state(self):
        stub = object.__new__(wiring.OrdinalSegmentationTrainer)
        stub.model = None
        stub.loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "ordinal_loss")
        stub.ordinal_settings = wiring.OrdinalRunSettings()
        stub.tensorboard = SimpleNamespace(log_dir="runs/e02/tensorboard", log_batch_scalars=True)
        stub.builtin_tensorboard = {"neutralized": True}
        report = wiring.OrdinalSegmentationTrainer.ordinal_diagnostics(stub)
        self.assertEqual(report["configured"]["lambda_ord"], 0.5)
        self.assertIsNone(report["criterion_class"])
        self.assertEqual(report["loss_items"], list(stub.loss_names))
        self.assertEqual(report["tensorboard"]["log_dir"], "runs/e02/tensorboard")
        self.assertTrue(report["tensorboard"]["builtin_ultralytics_integration"]["neutralized"])
        self.assertIn("train/ordinal_loss", report["tensorboard"]["required_tags"]["train"])


if __name__ == "__main__":
    unittest.main()
