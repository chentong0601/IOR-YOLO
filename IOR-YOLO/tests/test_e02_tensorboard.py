"""E02 TensorBoard contract: tag coverage, callback wiring and a real writer smoke.

No training, no CUDA and no dataset access: the writer is either injected or the
real ``SummaryWriter`` writing synthetic values into a temporary directory.
"""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ior_yolo.utils import tensorboard_logger as tb  # noqa: E402


class FakeWriter:
    def __init__(self):
        self.scalars = []
        self.flushed = 0
        self.closed = False

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), int(step)))

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


class FakeTrainer:
    """Minimal stand-in for the pinned trainer surface the logger reads."""

    def __init__(self):
        self.loss_names = (*tb.STOCK_LOSS_NAMES, tb.ORDINAL_LOSS_NAME)
        self.tloss = torch.tensor([1.5, 2.5, 3.5, 4.5, 0.012])
        self.loss = torch.tensor(12.0)
        self.lr = {"pg0": 0.01, "pg1": 0.02, "pg2": 0.0}
        self.epoch = 4
        self.metrics = {"metrics/precision(B)": 0.9, "metrics/recall(B)": 0.91, "metrics/mAP50(B)": 0.92,
                        "metrics/mAP50-95(B)": 0.93, "metrics/precision(M)": 0.94, "metrics/recall(M)": 0.95,
                        "metrics/mAP50(M)": 0.96, "metrics/mAP50-95(M)": 0.97, "val/box_loss": 1.1,
                        "val/seg_loss": 1.2, "val/cls_loss": 1.3, "val/dfl_loss": 1.4,
                        "val/ordinal_loss": 0.02, "fitness": 0.93, "note": "not numeric", "flag": True}
        self.epoch_time = 3.25
        self.train_time_start = time.time() - 42.0

    def label_loss_items(self, loss_items=None, prefix="train"):
        keys = [f"{prefix}/{name}" for name in self.loss_names]
        if loss_items is None:
            return keys
        return dict(zip(keys, [round(float(x), 5) for x in loss_items]))

    def _get_memory(self):
        return 1.25


class RequiredTagContractTest(unittest.TestCase):
    def test_training_losses_include_the_ordinal_term(self):
        for name in (*tb.STOCK_LOSS_NAMES, tb.ORDINAL_LOSS_NAME):
            self.assertIn(f"train/{name}", tb.REQUIRED_TAGS["train"])

    def test_validation_losses_and_box_mask_metrics_are_required(self):
        for name in tf_val_names():
            self.assertIn(f"val/{name}", tb.REQUIRED_TAGS["validation"])
        for prefix in ("precision", "recall", "mAP50", "mAP50-95"):
            self.assertIn(f"metrics/{prefix}(B)", tb.REQUIRED_TAGS["metrics"])
            self.assertIn(f"metrics/{prefix}(M)", tb.REQUIRED_TAGS["metrics"])

    def test_optimization_tags_cover_epoch_lr_memory_and_runtime(self):
        for tag in ("epoch", "lr/pg0", "perf/gpu_memory_gb", "perf/epoch_time_seconds", "perf/elapsed_seconds",
                    "perf/iteration"):
            self.assertIn(tag, tb.REQUIRED_TAGS["optimization"])

    def test_flat_required_tag_list_is_unique_and_ordered(self):
        tags = tb.required_tag_list()
        self.assertEqual(len(tags), len(set(tags)))
        self.assertEqual(tags[:5], list(tb.REQUIRED_TAGS["train"]))
        self.assertIn("train/ordinal_loss", tags)


def tf_val_names():
    return (*tb.STOCK_LOSS_NAMES, tb.ORDINAL_LOSS_NAME)


class LoggerWiringTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="e02-tensorboard-test-")
        self.writer = FakeWriter()
        self.logger = tb.E02TensorBoardLogger(Path(self.temp.name) / tb.TENSORBOARD_DIRNAME,
                                              writer_factory=lambda path: self.writer)

    def tearDown(self):
        self.logger.close()
        self.temp.cleanup()

    def test_write_before_start_fails(self):
        with self.assertRaises(RuntimeError):
            self.logger.write({"train/box_loss": 1.0}, step=1)

    def test_callbacks_cover_the_required_events(self):
        self.assertEqual(sorted(self.logger.callbacks()),
                         ["on_fit_epoch_end", "on_train_batch_end", "on_train_end", "on_train_epoch_end"])

    def test_epoch_end_writes_train_losses_and_optimization_state(self):
        self.logger.start()
        self.logger.on_train_epoch_end(FakeTrainer())
        tags = {tag for tag, _, step in self.writer.scalars if step == 5}
        for tag in tb.REQUIRED_TAGS["train"]:
            self.assertIn(tag, tags)
        for tag in ("epoch", "lr/pg0", "perf/gpu_memory_gb", "perf/elapsed_seconds"):
            self.assertIn(tag, tags)

    def test_fit_epoch_end_writes_validation_losses_and_metrics(self):
        self.logger.start()
        self.logger.on_fit_epoch_end(FakeTrainer())
        tags = {tag for tag, _, _ in self.writer.scalars}
        for tag in (*tb.REQUIRED_TAGS["validation"], *tb.REQUIRED_TAGS["metrics"]):
            self.assertIn(tag, tags)

    def test_batch_end_counts_iterations_and_writes_the_total_loss(self):
        self.logger.start()
        trainer = FakeTrainer()
        self.logger.on_train_batch_end(trainer)
        self.logger.on_train_batch_end(trainer)
        iterations = [(tag, step) for tag, _, step in self.writer.scalars if tag == "perf/iteration"]
        self.assertEqual(iterations, [("perf/iteration", 1), ("perf/iteration", 2)])
        self.assertIn("train/total_loss", {tag for tag, _, _ in self.writer.scalars})

    def test_batch_logging_can_be_disabled(self):
        logger = tb.E02TensorBoardLogger(Path(self.temp.name) / "disabled", writer_factory=lambda path: self.writer,
                                         log_batch_scalars=False)
        logger.start()
        logger.on_train_batch_end(FakeTrainer())
        self.assertEqual(self.writer.scalars, [])
        logger.close()

    def test_non_finite_values_are_skipped(self):
        self.logger.start()
        written = self.logger.write({"a": float("nan"), "b": float("inf"), "c": "text", "d": 1.5,
                                     "e": True, "f": float("-inf")}, step=1)
        self.assertEqual(written, 1)
        self.assertEqual([tag for tag, _, _ in self.writer.scalars], ["d"])

    def test_train_end_closes_and_close_is_idempotent(self):
        self.logger.start()
        self.logger.on_train_end(FakeTrainer())
        self.assertTrue(self.writer.closed)
        self.assertGreaterEqual(self.writer.flushed, 1)
        self.logger.close()
        self.assertTrue(self.writer.closed)

    def test_default_writer_factory_fails_loudly_without_tensorboard(self):
        with patch.dict(sys.modules, {"torch.utils.tensorboard": None}):
            with self.assertRaises(RuntimeError) as caught:
                tb.default_writer_factory(Path(self.temp.name) / "missing")
        self.assertIn("requirements-e02.txt", str(caught.exception))


class RoundTripTest(unittest.TestCase):
    @unittest.skipUnless(tb.tensorboard_availability()["available"],
                         "TensorBoard runtime not installed (see IOR-YOLO/requirements-e02.txt)")
    def test_roundtrip_writes_and_reads_back_every_required_tag(self):
        with tempfile.TemporaryDirectory(prefix="e02-tensorboard-roundtrip-") as temporary:
            directory = Path(temporary) / tb.TENSORBOARD_DIRNAME
            report = tb.tensorboard_roundtrip(directory, tags=("train/ordinal_loss", "epoch", "lr/pg0"))
            self.assertTrue(report["event_files"])
            self.assertEqual(report["missing_tags"], [])
            self.assertTrue(report["ok"])
            self.assertIn("EventAccumulator", report["read_back_backend"])

    @unittest.skipUnless(tb.tensorboard_availability()["available"],
                         "TensorBoard runtime not installed (see IOR-YOLO/requirements-e02.txt)")
    def test_full_required_contract_roundtrips_with_the_real_writer(self):
        with tempfile.TemporaryDirectory(prefix="e02-tensorboard-contract-") as temporary:
            report = tb.tensorboard_roundtrip(Path(temporary) / tb.TENSORBOARD_DIRNAME)
        self.assertTrue(report["ok"])
        self.assertEqual(report["tags_written"], len(tb.required_tag_list()))
        self.assertEqual(report["missing_tags"], [])
        for tag in tb.required_tag_list():
            self.assertIn(tag, report["observed_tags"])
        self.assertGreater(report["tags_expected"], 20)

    def test_availability_reports_the_writer_description(self):
        availability = tb.tensorboard_availability()
        self.assertEqual(availability["writer"], tb.WRITER_DESCRIPTION)
        self.assertIn("available", availability)

    def test_builtin_ultralytics_integration_is_neutralized(self):
        from ultralytics.utils.callbacks import tensorboard as builtin
        registered = {"on_pretrain_routine_start": object(), "on_train_start": object()}
        with patch.object(builtin, "callbacks", registered), patch.object(builtin, "SummaryWriter", object()):
            record = tb.neutralize_builtin_ultralytics_tensorboard()
            self.assertTrue(record["neutralized"])
            self.assertTrue(record["builtin_summary_writer_available"])
            self.assertEqual(record["builtin_callbacks_registered"], len(registered))
            self.assertEqual(builtin.callbacks, {})
            self.assertIsNone(builtin.SummaryWriter)

    def test_roundtrip_tags_are_readable_without_the_full_contract(self):
        report = tb.tensorboard_roundtrip(Path(tempfile.mkdtemp(prefix="e02-tb-partial-")) / "tb",
                                          tags=("train/ordinal_loss",))
        self.assertEqual(report["tags_written"], 1)
        self.assertEqual(report["read_back_backend"] and report["ok"], True)


if __name__ == "__main__":
    unittest.main()


class ScalarBuilderTest(unittest.TestCase):
    def test_train_scalars_carry_the_ordinal_series(self):
        scalars = tb.train_scalars(FakeTrainer())
        self.assertEqual(sorted(scalars), sorted(f"train/{name}" for name in tf_val_names()))
        self.assertAlmostEqual(scalars["train/ordinal_loss"], 0.012, places=6)

    def test_train_scalars_refuses_a_drifted_loss_name_contract(self):
        trainer = FakeTrainer()
        trainer.loss_names = (*tb.STOCK_LOSS_NAMES,)
        with self.assertRaises(RuntimeError):
            tb.train_scalars(trainer)
        trainer = FakeTrainer()
        trainer.tloss = torch.tensor([1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(RuntimeError):
            tb.train_scalars(trainer)
        trainer = FakeTrainer()
        trainer.tloss = None
        self.assertEqual(tb.train_scalars(trainer), {})

    def test_validation_scalars_keep_only_numeric_metrics(self):
        scalars = tb.validation_scalars(FakeTrainer())
        self.assertIn("val/ordinal_loss", scalars)
        self.assertIn("metrics/mAP50(B)", scalars)
        self.assertIn("metrics/mAP50-95(M)", scalars)
        self.assertNotIn("note", scalars)
        self.assertNotIn("flag", scalars)

    def test_optimization_scalars_report_epoch_lr_memory_and_runtime(self):
        scalars = tb.optimization_scalars(FakeTrainer(), epoch=5, batch_iteration=17)
        self.assertEqual(scalars["epoch"], 5.0)
        self.assertEqual(scalars["perf/iteration"], 17.0)
        self.assertEqual(scalars["lr/pg0"], 0.01)
        self.assertEqual(scalars["lr/pg2"], 0.0)
        self.assertEqual(scalars["perf/gpu_memory_gb"], 1.25)
        self.assertEqual(scalars["perf/epoch_time_seconds"], 3.25)
        self.assertGreater(scalars["perf/elapsed_seconds"], 0.0)

    def test_batch_scalars_add_the_total_loss(self):
        scalars = tb.batch_scalars(FakeTrainer())
        self.assertEqual(scalars["train/total_loss"], 12.0)
        self.assertIn("train/ordinal_loss", scalars)
