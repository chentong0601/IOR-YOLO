"""Canonical E02 TensorBoard logging.

TensorBoard is a hard requirement for formal E02 training (see
``docs/e02-ordinal-protocol.md``): every E02 epoch must leave box/seg/cls/dfl and
ordinal training losses, the four validation loss items, box/mask
precision/recall/mAP50/mAP50-95 and the optimization state in the E02 run
directory. Nothing here writes outside the caller-supplied log directory, so the
frozen E01 run directory can never be touched.

Two independent log sources would make the dashboard ambiguous (and the
Ultralytics built-in integration cannot log ``ordinal_loss`` at all), so this
module owns the single canonical writer under ``<run_dir>/tensorboard`` and
:func:`neutralize_builtin_ultralytics_tensorboard` disables the Ultralytics
integration in-process for E02 runs only.

The scalars contract is produced by pure functions of a trainer-like object, so
it is unit-testable without training, CUDA or datasets.
"""

from __future__ import annotations

import time
from pathlib import Path

TENSORBOARD_DIRNAME = "tensorboard"
WRITER_DESCRIPTION = "torch.utils.tensorboard.SummaryWriter"

# Every tag below must appear in a formal E02 run (see the protocol document).
REQUIRED_TAGS: dict[str, tuple[str, ...]] = {
    "train": ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss", "train/ordinal_loss"),
    "validation": ("val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss", "val/ordinal_loss"),
    "metrics": ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
                "metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"),
    "optimization": ("epoch", "lr/pg0", "perf/gpu_memory_gb", "perf/epoch_time_seconds",
                     "perf/elapsed_seconds", "perf/iteration", "train/total_loss"),
}
# Tags the stock Ultralytics validators additionally produce; logged when present.
OPTIONAL_TAGS: tuple[str, ...] = ("fitness",)
# Stock Ultralytics loss names that must stay untouched next to ordinal_loss.
STOCK_LOSS_NAMES: tuple[str, ...] = ("box_loss", "seg_loss", "cls_loss", "dfl_loss")
ORDINAL_LOSS_NAME = "ordinal_loss"
# Round-to-5-decimals behaviour of Ultralytics' own label_loss_items is kept.
LOSS_ITEM_PREFIX = "train"


def required_tag_list() -> list[str]:
    """Flat, ordered list of every tag a formal E02 run must record."""
    return [tag for group in REQUIRED_TAGS.values() for tag in group]


def default_writer_factory(log_dir: Path):
    """Create the real TensorBoard writer, failing loudly when unavailable."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - exercised only without tensorboard
        raise RuntimeError(
            "TensorBoard logging is a hard requirement for E02: install the pinned E02 runtime "
            "(python -m pip install -r IOR-YOLO/requirements-e02.txt) and re-run; no E02 run may "
            f"silently train without a dashboard. Cause: {type(exc).__name__}: {exc}") from exc
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _as_float(value):
    """Convert a metric value to float, or return ``None`` when not numeric."""
    if isinstance(value, bool):
        return None
    detach = getattr(value, "detach", None)  # torch tensors (e.g. trainer.loss) must not warn/keep graph
    if callable(detach):
        value = detach()
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def train_scalars(trainer) -> dict[str, float]:
    """Training loss items of the current epoch, including ``train/ordinal_loss``.

    Raises ``RuntimeError`` instead of silently dropping the ordinal series: the
    auxiliary term must never disappear from the dashboard because a loss-items
    vector and the trainer's loss names drifted apart.
    """
    items = getattr(trainer, "tloss", None)
    if items is None:
        return {}
    names = tuple(getattr(trainer, "loss_names", ()) or ())
    if tuple(names) != (*STOCK_LOSS_NAMES, ORDINAL_LOSS_NAME):
        raise RuntimeError(f"E02 loss-name contract violated: expected {(*STOCK_LOSS_NAMES, ORDINAL_LOSS_NAME)}, "
                           f"got {names}")
    length = int(items.shape[0]) if hasattr(items, "shape") and len(items.shape) else 1
    if length != len(names):
        raise RuntimeError(f"E02 loss-item contract violated: {length} items vs {len(names)} loss names")
    labelled = dict(trainer.label_loss_items(items, prefix=LOSS_ITEM_PREFIX))
    if tuple(labelled) != tuple(f"{LOSS_ITEM_PREFIX}/{name}" for name in names):
        raise RuntimeError(f"E02 loss labelling contract violated: {tuple(labelled)}")
    return {key: float(value) for key, value in labelled.items()}


def validation_scalars(trainer) -> dict[str, float]:
    """Validation loss items and box/mask metrics of the last validation pass."""
    metrics = getattr(trainer, "metrics", None) or {}
    scalars = {}
    for key, value in metrics.items():
        number = _as_float(value)
        if number is not None:
            scalars[str(key)] = number
    return scalars


def _gpu_memory_gb(trainer) -> float | None:
    reader = getattr(trainer, "_get_memory", None)
    if not callable(reader):
        return None
    try:
        return float(reader())
    except Exception:  # pragma: no cover - depends on backend/driver state
        return None


def optimization_scalars(trainer, *, epoch: int | None = None, batch_iteration: int | None = None) -> dict[str, float]:
    """Epoch index, learning rate(s), GPU memory and runtime/iteration counters."""
    scalars: dict[str, float] = {}
    if epoch is not None:
        scalars["epoch"] = float(epoch)
    if batch_iteration is not None:
        scalars["perf/iteration"] = float(batch_iteration)
    for key, value in (getattr(trainer, "lr", None) or {}).items():
        number = _as_float(value)
        if number is not None:
            scalars[str(key) if str(key).startswith("lr/") else f"lr/{key}"] = number
    memory = _gpu_memory_gb(trainer)
    if memory is not None:
        scalars["perf/gpu_memory_gb"] = memory
    epoch_time = _as_float(getattr(trainer, "epoch_time", None))
    if epoch_time is not None:
        scalars["perf/epoch_time_seconds"] = epoch_time
    started = _as_float(getattr(trainer, "train_time_start", None))
    if started is not None:
        scalars["perf/elapsed_seconds"] = max(0.0, time.time() - started)
    return scalars


def batch_scalars(trainer) -> dict[str, float]:
    """Per-iteration scalars: the epoch loss items plus the current total loss."""
    scalars = train_scalars(trainer)
    total = _as_float(getattr(trainer, "loss", None))
    if total is not None:
        scalars["train/total_loss"] = total
    return scalars


class E02TensorBoardLogger:
    """E02-scoped TensorBoard writer driven by Ultralytics trainer callbacks.

    ``writer_factory`` is injectable so the callback wiring can be unit-tested
    without the TensorBoard package; the default factory is the real
    ``SummaryWriter`` and fails loudly when TensorBoard is missing.
    """

    def __init__(self, log_dir: Path | str, *, writer_factory=None, log_batch_scalars: bool = True):
        self.log_dir = Path(log_dir)
        self.log_batch_scalars = bool(log_batch_scalars)
        self._factory = writer_factory or default_writer_factory
        self._writer = None
        self._iteration = 0
        self.written: list[tuple[str, float, int]] = []

    @property
    def started(self) -> bool:
        return self._writer is not None

    def start(self) -> None:
        """Create the log directory and the writer (idempotent)."""
        if self._writer is not None:
            return
        self._writer = self._factory(self.log_dir)

    def write(self, scalars: dict, step: int) -> int:
        """Write finite scalars; returns the number of tags written."""
        if self._writer is None:
            raise RuntimeError("E02 TensorBoard logger used before start()")
        written = 0
        for tag, value in scalars.items():
            number = _as_float(value)
            if number is None or number != number or number in (float("inf"), float("-inf")):
                continue
            self._writer.add_scalar(str(tag), number, int(step))
            self.written.append((str(tag), number, int(step)))
            written += 1
        return written

    def close(self) -> None:
        """Flush and release the writer (idempotent)."""
        writer, self._writer = self._writer, None
        if writer is None:
            return
        flush = getattr(writer, "flush", None)
        if callable(flush):
            flush()
        close = getattr(writer, "close", None)
        if callable(close):
            close()

    def callbacks(self) -> dict[str, object]:
        """Trainer callback mapping to register with ``trainer.add_callback``."""
        return {"on_train_epoch_end": self.on_train_epoch_end,
                "on_fit_epoch_end": self.on_fit_epoch_end,
                "on_train_batch_end": self.on_train_batch_end,
                "on_train_end": self.on_train_end}

    def on_train_epoch_end(self, trainer) -> None:
        """Training loss items, learning rate, epoch and optimization state at the end of an epoch."""
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        self.write(train_scalars(trainer), epoch)
        self.write(optimization_scalars(trainer, epoch=epoch), epoch)

    def on_fit_epoch_end(self, trainer) -> None:
        """Validation losses and box/mask metrics, then optimization state."""
        step = int(getattr(trainer, "epoch", 0)) + 1
        self.write(validation_scalars(trainer), step)
        self.write(optimization_scalars(trainer, epoch=step), step)

    def on_train_batch_end(self, trainer) -> None:
        """Optional per-iteration trace of the losses including the ordinal term."""
        if not self.log_batch_scalars:
            return
        self._iteration += 1
        scalars = batch_scalars(trainer)
        scalars["perf/iteration"] = float(self._iteration)
        self.write(scalars, self._iteration)

    def on_train_end(self, trainer) -> None:
        """Close the writer so the event file is complete and readable."""
        self.close()


def neutralize_builtin_ultralytics_tensorboard() -> dict:
    """Disable the Ultralytics TensorBoard integration in this process only.

    Returns a provenance record. It is applied before the trainer is built, so
    the E02 dashboard has exactly one writer (``<run_dir>/tensorboard``) that
    also carries ``ordinal_loss``; nothing is written to disk, no Ultralytics
    setting file is modified, and non-E02 trainer paths are unaffected.
    """
    try:
        from ultralytics.utils.callbacks import tensorboard as builtin
    except Exception as exc:  # pragma: no cover - import failure is environment specific
        return {"neutralized": False, "reason": f"import failed: {type(exc).__name__}: {exc}"}
    registered = len(getattr(builtin, "callbacks", {}) or {})
    writer_available = getattr(builtin, "SummaryWriter", None) is not None
    if registered:
        builtin.callbacks = {}
    if writer_available:
        builtin.SummaryWriter = None
    return {"neutralized": True, "builtin_callbacks_registered": registered,
            "builtin_summary_writer_available": writer_available}


def tensorboard_availability() -> dict:
    """Report whether the real TensorBoard backend is importable."""
    try:
        import tensorboard
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "writer": WRITER_DESCRIPTION}
    return {"available": True, "version": str(getattr(tensorboard, "__version__", "unknown")),
            "writer": WRITER_DESCRIPTION}


def event_files(log_dir: Path | str) -> list[str]:
    """Names of TensorBoard event files inside ``log_dir``."""
    return sorted(path.name for path in Path(log_dir).glob("events.out.tfevents*"))


def read_written_tags(log_dir: Path | str) -> tuple[list[str], str]:
    """Read scalar tag names back from written event files, tolerating missing directories.

    Returns ``(sorted_tag_names, backend_description)``. A missing/empty log
    directory yields an empty list and a reason instead of an exception so
    callers can report their own contract violation.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as exc:
        return [], f"unavailable: {type(exc).__name__}: {exc}"
    directory = Path(log_dir)
    if not directory.is_dir() or not event_files(directory):
        return [], "no event files present"
    try:
        accumulator = EventAccumulator(str(directory))
        accumulator.Reload()
        return sorted(accumulator.Tags().get("scalars", [])), \
            "tensorboard.backend.event_processing.event_accumulator.EventAccumulator"
    except Exception as exc:  # unreadable/corrupt event files must not crash the gate
        return [], f"read failed: {type(exc).__name__}: {exc}"


def tensorboard_roundtrip(log_dir: Path | str, *, tags=None, step: int = 1) -> dict:
    """Write the tag contract with synthetic values and read every tag back.

    Used by the E02 preflight gate and by the disposable ``tensorboard-smoke``
    command: it proves the real writer works and that the dashboard will carry
    the required series, without training, CUDA or dataset access.
    """
    contract = tuple(tags) if tags is not None else tuple(required_tag_list())
    directory = Path(log_dir)
    logger = E02TensorBoardLogger(directory)
    logger.start()
    written = logger.write({tag: float(index + 1) for index, tag in enumerate(contract)}, step=step)
    logger.close()
    files = event_files(directory)
    observed, backend = read_written_tags(directory)
    missing = [tag for tag in contract if tag not in observed]
    return {"log_dir": str(directory), "writer": WRITER_DESCRIPTION, "event_files": files,
            "tags_expected": len(contract), "tags_written": written, "observed_tags": observed,
            "missing_tags": missing, "read_back_backend": backend,
            "ok": bool(files) and written == len(contract) and not missing}
