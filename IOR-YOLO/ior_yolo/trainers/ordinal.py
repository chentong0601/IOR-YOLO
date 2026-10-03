"""E02 ordinal-aware YOLO11n-seg criterion, model, trainer and YOLO entrypoint.

Implementation record (required by the E02 protocol):

1. **Where the ordinal loss is injected.** ``OrdinalSegmentationLoss`` (this
   module) subclasses the pinned ``ultralytics.utils.loss.v8SegmentationLoss``
   (8.3.220). The auxiliary term is computed inside ``__call__``, after the
   stock segmentation loss has been computed by ``super().__call__``, and is
   added to the total loss the trainer backpropagates.
2. **Form of the original Ultralytics classification loss.** In 8.3.220
   ``v8SegmentationLoss.__call__`` computes
   ``loss[2] = BCEWithLogitsLoss(reduction="none")(pred_scores, target_scores).sum() / target_scores_sum``
   where ``pred_scores`` are the raw per-anchor logits ``(B, A, nc)`` and
   ``target_scores`` are the task-aligned assigner targets. Those targets are
   *not* one-hot: the assigner multiplies the one-hot label scatter by its
   alignment metric, so ``target_scores[i, assigned_class_i] = w_i`` (the
   per-anchor quality) and everything else is 0. The stock classification loss
   divides by ``target_scores_sum = sum_i w_i`` and ``BboxLoss`` weights its
   per-instance terms by the same ``w_i``. It is scaled by ``hyp.cls`` and
   returned as ``loss * batch_size`` together with the detached item vector
   ``(box, seg, cls, dfl)``.
3. **How the ordinal loss is merged.** ``super().__call__`` is executed first and
   its return values are kept *unchanged*: no existing loss, gain, assignment or
   normalization is touched. The ordinal component is
   ``sum_i w_i * err_i / max(sum_i w_i, 1)`` - the same per-anchor quality weight
   and the same ``target_scores_sum`` denominator as the stock classification
   loss - then weighted by ``lambda_ord`` and scaled by ``batch_size``. Pinned
   8.3.220 returns the stock total as the per-component vector
   ``(box, seg, cls, dfl) * batch_size`` and the trainer optimises ``loss.sum()``,
   so the auxiliary term is added *exactly once*, to the classification slot, and
   the optimised objective is ``sum(L_original_YOLO) + lambda_ord * L_ord *
   batch_size`` with the effective gain ``lambda_ord = 0.5``. ``scale_record``
   re-derives that delta from the two totals and refuses to record a step where
   the term was applied any other number of times. The weighted component is
   appended to the item vector as a fifth entry so ``results.csv`` and TensorBoard
   record it as ``ordinal_loss``.
4. **Why the targets are not recomputed.** The task-aligned assigner is wrapped
   in a forwarding recorder (:class:`AssignerCallRecorder`). The ordinal term
   reuses the *same* matched anchors and stage targets the original
   classification loss was scored against, so no second assignment, no second
   target-building path and no divergent matching behaviour is introduced.
5. **Verification instead of trust.** The classification logits recovered from
   the raw head output are compared against the tensor the stock loss passed to
   the assigner; a mismatch raises instead of silently optimizing the wrong
   channels (see ``verify_logits``).
6. **Minimal invasion.** No Ultralytics source file is copied or patched; only
   ``init_criterion``, ``get_model``, ``get_validator`` and loss-item labelling
   are overridden. ``OrdinalYOLO`` keeps the stock validator and predictor, so
   validation, prediction export and metrics are byte-for-byte the E01 path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics.models.yolo.model import YOLO
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.loss import v8SegmentationLoss

from ior_yolo.losses.ordinal import (E02_LAMBDA_ORD, NUM_STAGES, STAGE_ORDER, OrdinalLossConfig,
                                     describe_component, ordinal_component, total_loss_with_ordinal)
from ior_yolo.utils.tensorboard_logger import (E02TensorBoardLogger, ORDINAL_LOSS_NAME, REQUIRED_TAGS,
                                              TENSORBOARD_DIRNAME, WRITER_DESCRIPTION,
                                              neutralize_builtin_ultralytics_tensorboard)

ORDINAL_CONFIG_KEY = "ordinal_config"
SETTINGS_KEYS = ("lambda_ord", "num_stages", "stage_order", "log_batch_scalars")
# How many plain-float scale records are kept for the engineering sanity report.
MAX_SCALE_RECORDS = 64


def scale_record(*, step: int, batch_size: int, stock_items: torch.Tensor, component: dict,
                 total_before: torch.Tensor, weighted: torch.Tensor, total_after: torch.Tensor) -> dict:
    """Plain-float snapshot of one step's loss scale (engineering sanity check only).

    ``weighted`` is the logged fifth item ``lambda_ord * L_ord`` (batch mean, the
    same scale as the stock item vector); ``total_before``/``total_after`` are the
    batch-scaled objectives the optimizer actually sees, which pinned 8.3.220 hands
    over as the per-component vector ``(box, seg, cls, dfl) * batch_size`` and the
    trainer reduces with ``loss.sum()``. Nothing here is a knob: ``lambda_ord`` is
    frozen by the protocol and must never be chosen from these numbers.
    """
    stock = [float(value) for value in stock_items.detach().flatten()[:4]]
    ordinal = float(weighted.detach())
    total_before_value = float(total_before.detach().sum())
    total_after_value = float(total_after.detach().sum())
    applied = total_after_value - total_before_value
    expected = ordinal * batch_size
    if not math.isclose(applied, expected, rel_tol=1e-4, abs_tol=1e-6):
        raise RuntimeError(f"the ordinal term was not applied exactly once: the objective grew by "
                           f"{applied:.6f} while lambda_ord * L_ord * batch_size = {expected:.6f}; a "
                           "component-wise total would apply it once per component")
    return {"step": int(step), "batch_size": int(batch_size), "instances": int(component["instances"]),
            "stock_box_loss": stock[0], "stock_seg_loss": stock[1], "stock_cls_loss": stock[2],
            "stock_dfl_loss": stock[3], "raw_ordinal_loss": float(component["loss"].detach()),
            "lambda_times_ordinal": ordinal, "lambda_scaled_ordinal_batch": expected,
            "total_before_ordinal": total_before_value, "total_after_ordinal": total_after_value,
            "ordinal_share_of_total": (applied / total_after_value) if total_after_value else None,
            "quality_mean": component["quality_mean"]}


def summarize_scale(records: list[dict]) -> dict | None:
    """Mean loss scale over the recorded steps; ``None`` when no step carried instances."""
    scored = [record for record in records if record["instances"] > 0]
    if not scored:
        return None

    def mean(key: str) -> float:
        return sum(record[key] for record in scored) / len(scored)

    return {"steps_recorded": len(records), "steps_with_instances": len(scored),
            "mean_stock_cls_loss": mean("stock_cls_loss"), "mean_raw_ordinal_loss": mean("raw_ordinal_loss"),
            "mean_lambda_times_ordinal": mean("lambda_times_ordinal"),
            "mean_lambda_scaled_ordinal_batch": mean("lambda_scaled_ordinal_batch"),
            "mean_ordinal_share_of_total": mean("ordinal_share_of_total"),
            "mean_instances_per_step": mean("instances"), "mean_quality": mean("quality_mean"),
            "scope": "engineering scale sanity only; lambda_ord stays frozen at 0.5 and was NOT tuned from these "
                     "numbers"}


@dataclass(frozen=True)
class OrdinalRunSettings:
    """Validated E02 ordinal settings travelling from the E02 config into training."""

    lambda_ord: float = E02_LAMBDA_ORD
    num_stages: int = NUM_STAGES
    stage_order: tuple[str, ...] = STAGE_ORDER
    log_batch_scalars: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict) -> "OrdinalRunSettings":
        """Build settings from the ``ordinal_config`` block, rejecting anything unknown."""
        if not isinstance(mapping, dict):
            raise TypeError(f"ordinal_config must be a mapping, got {type(mapping).__name__}")
        unknown = sorted(set(mapping) - set(SETTINGS_KEYS))
        if unknown:
            raise ValueError(f"unknown ordinal_config keys: {unknown}")
        missing = sorted({"lambda_ord", "num_stages", "stage_order"} - set(mapping))
        if missing:
            raise ValueError(f"missing ordinal_config keys: {missing}")
        return cls(lambda_ord=float(mapping["lambda_ord"]), num_stages=int(mapping["num_stages"]),
                   stage_order=tuple(str(stage) for stage in mapping["stage_order"]),
                   log_batch_scalars=bool(mapping.get("log_batch_scalars", True)))

    def __post_init__(self) -> None:
        """Validate immediately so a broken ordinal block fails before any trainer is built."""
        if not isinstance(self.log_batch_scalars, bool):
            raise ValueError(f"log_batch_scalars must be a bool, got {self.log_batch_scalars!r}")
        self.loss_config()  # raises for a bad stage count, stage order or lambda_ord

    def loss_config(self) -> OrdinalLossConfig:
        """Return the validated loss-level configuration."""
        return OrdinalLossConfig(num_stages=self.num_stages, lambda_ord=self.lambda_ord,
                                 stage_order=self.stage_order)

    def as_dict(self) -> dict:
        """YAML/JSON-safe description for run manifests."""
        return {"lambda_ord": self.lambda_ord, "num_stages": self.num_stages,
                "stage_order": list(self.stage_order), "log_batch_scalars": self.log_batch_scalars}


def pop_ordinal_settings(overrides: dict) -> OrdinalRunSettings:
    """Remove and validate ``ordinal_config`` before Ultralytics validates its own cfg keys."""
    if ORDINAL_CONFIG_KEY not in overrides:
        raise ValueError(f"E02 training requires an explicit {ORDINAL_CONFIG_KEY!r} block; refusing to train without "
                         "a recorded ordinal configuration")
    return OrdinalRunSettings.from_mapping(overrides.pop(ORDINAL_CONFIG_KEY))


class AssignerCallRecorder:
    """Forwarding proxy that records the last ``TaskAlignedAssigner`` call.

    The proxy adds no behaviour: arguments, keyword arguments and return values
    are forwarded untouched. Recording lets the ordinal term reuse the exact
    matched anchors and stage targets of the stock classification loss without
    re-running the assigner or duplicating its target construction. Only the most
    recent call is retained and it is dropped right after use, so no tensors stay
    alive across steps or inside saved checkpoints.
    """

    def __init__(self, assigner):
        self.assigner = assigner
        self.last_args: tuple | None = None
        self.last_output = None
        self.calls = 0

    def __call__(self, *args, **kwargs):
        output = self.assigner(*args, **kwargs)
        self.calls += 1
        self.last_args = args
        self.last_output = output
        return output

    def reset(self) -> None:
        """Drop the recorded references (called once the ordinal term consumed them)."""
        self.last_args = None
        self.last_output = None


class OrdinalSegmentationLoss(v8SegmentationLoss):
    """Pinned Ultralytics segmentation loss plus the E02 ordinal auxiliary term."""

    def __init__(self, model, *, loss_config: OrdinalLossConfig | None = None, verify_logits: bool = True):
        super().__init__(model)
        self.loss_config = loss_config or OrdinalLossConfig()
        if self.nc != self.loss_config.num_stages:
            raise ValueError(f"ordinal loss expects {self.loss_config.num_stages} stages but the model head has "
                             f"{self.nc} classes")
        self.verify_logits = bool(verify_logits)
        self.assigner = AssignerCallRecorder(self.assigner)
        self.last_ordinal: dict | None = None
        # Plain-float scale records of the most recent steps (no tensors, no graph);
        # used for the engineering scale sanity report only, never for tuning.
        self.scale_records: list[dict] = []
        self.steps = 0

    def recompute_logits(self, preds):
        """Recover the ``(B, A, nc)`` classification logits from the raw head output.

        Ultralytics wraps the per-level head outputs differently in training
        (``(feats, mask_coeffs, proto)``), in eval/validation
        (``(detections, (feats, mask_coeffs, proto))``) and for detection-style
        outputs, so the level list is located structurally and validated by
        channel count instead of assuming one wrapping. Anything else raises.
        """
        levels = []
        stack = [(preds, 0)]
        while stack:
            node, depth = stack.pop()
            if depth > 3 or isinstance(node, torch.Tensor):
                continue
            if isinstance(node, (list, tuple)):
                if node and all(isinstance(level, torch.Tensor) and level.dim() == 4 and level.shape[1] == self.no
                                for level in node):
                    levels.append(node)
                    continue
                stack.extend((child, depth + 1) for child in node)
        if not levels:
            raise RuntimeError("E02: no per-level head output with the pinned channel count was found in the model "
                               "output; refusing to add the ordinal term blindly")
        feats = levels[0]
        stacked = torch.cat([level.view(feats[0].shape[0], self.no, -1) for level in feats], 2)
        _, scores = stacked.split((self.reg_max * 4, self.nc), 1)
        return scores.permute(0, 2, 1).contiguous()

    def ordinal_term(self, preds) -> dict:
        """Ordinal component of this step, built from the recorded assignment."""
        outputs, args = self.assigner.last_output, self.assigner.last_args
        if outputs is None or not args:
            raise RuntimeError("E02: no task-aligned assignment recorded for this forward pass; refusing to guess the "
                               "ordinal targets")
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 4:
            raise RuntimeError("E02: unexpected task-aligned assigner output; the pinned assignment contract changed "
                               "and the ordinal term cannot be trusted")
        target_scores, foreground = outputs[2], outputs[3]
        logits = self.recompute_logits(preds)
        if self.verify_logits:
            assigned = args[0].detach()
            agreed = (logits.shape == assigned.shape and torch.allclose(
                logits.detach().sigmoid(), assigned.to(logits.dtype), atol=1e-4, rtol=1e-3))
            if not agreed:
                raise RuntimeError("E02: recovered classification logits disagree with the pinned Ultralytics head "
                                   "output; refusing to add the ordinal term to a mismatched objective")
        component = ordinal_component(logits, target_scores, foreground.bool(),
                                      num_stages=self.loss_config.num_stages)
        component["batch_size"] = int(logits.shape[0])
        return component

    def __call__(self, preds, batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Stock segmentation loss (unchanged) plus ``lambda_ord * L_ord`` on matched anchors."""
        total, items = super().__call__(preds, batch)
        component = self.ordinal_term(preds)
        self.assigner.reset()  # release the recorded tensors immediately
        batch_size = component["batch_size"]
        merged_total, weighted = total_loss_with_ordinal(total, component, self.loss_config, batch_size=batch_size)
        extended = torch.cat([items, weighted.reshape(1).to(items.dtype)])
        record = scale_record(step=self.steps, batch_size=batch_size, stock_items=items, component=component,
                              total_before=total, weighted=weighted, total_after=merged_total)
        self.scale_records.append(record)
        del self.scale_records[:-MAX_SCALE_RECORDS]
        self.last_ordinal = {**describe_component(component), "lambda_ord": self.loss_config.lambda_ord,
                             "weighted_loss": float(weighted), "batch_size": batch_size,
                             "step": self.steps, "scale": record}
        self.steps += 1
        return merged_total, extended


class OrdinalSegmentationModel(SegmentationModel):
    """Pinned YOLO11n-seg architecture; only the loss criterion class changes."""

    ordinal_settings: OrdinalRunSettings = OrdinalRunSettings()

    def init_criterion(self):
        """Return the E02 criterion; head, weights and forward path stay stock."""
        return OrdinalSegmentationLoss(self, loss_config=self.ordinal_settings.loss_config())


class OrdinalSegmentationTrainer(SegmentationTrainer):
    """Pinned ``SegmentationTrainer`` + E02 criterion, fifth loss item and dashboard."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        self.ordinal_settings = pop_ordinal_settings(overrides)
        # Must happen before BaseTrainer.__init__ registers its own integration callbacks.
        self.builtin_tensorboard = neutralize_builtin_ultralytics_tensorboard()
        super().__init__(cfg, overrides, _callbacks)
        self.tensorboard = E02TensorBoardLogger(Path(self.save_dir) / TENSORBOARD_DIRNAME,
                                                log_batch_scalars=self.ordinal_settings.log_batch_scalars)
        for event, callback in self.tensorboard.callbacks().items():
            self.add_callback(event, callback)
        self.tensorboard.start()

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the E02 model with the same signature and loading semantics as the stock trainer."""
        model = OrdinalSegmentationModel(cfg, nc=self.data["nc"], ch=self.data["channels"],
                                         verbose=verbose and RANK == -1)
        model.ordinal_settings = self.ordinal_settings
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Let ``ordinal_loss`` be logged as the fifth loss item, train and val alike."""
        validator = super().get_validator()
        self.loss_names = (*self.loss_names, ORDINAL_LOSS_NAME)
        return validator

    def tensorboard_provenance(self) -> dict:
        """Provenance of the canonical E02 dashboard."""
        return {"writer": WRITER_DESCRIPTION, "log_dir": str(self.tensorboard.log_dir),
                "log_batch_scalars": self.tensorboard.log_batch_scalars,
                "required_tags": {group: list(tags) for group, tags in REQUIRED_TAGS.items()},
                "builtin_ultralytics_integration": self.builtin_tensorboard}

    def ordinal_diagnostics(self) -> dict:
        """Ordinal state of the training criterion for the run manifest."""
        from ultralytics.utils.torch_utils import unwrap_model

        model = unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        criterion = getattr(model, "criterion", None) if model is not None else None
        recorder = getattr(criterion, "assigner", None)
        return {"configured": self.ordinal_settings.as_dict(),
                "criterion_class": type(criterion).__name__ if criterion is not None else None,
                "assigner_calls": int(getattr(recorder, "calls", 0)),
                "criterion_steps": int(getattr(criterion, "steps", 0)),
                "last_ordinal": getattr(criterion, "last_ordinal", None),
                "scale_history": summarize_scale(list(getattr(criterion, "scale_records", []) or [])),
                "loss_items": list(getattr(self, "loss_names", ()) or ()),
                "tensorboard": self.tensorboard_provenance()}


class OrdinalYOLO(YOLO):
    """Ultralytics ``YOLO`` entrypoint whose ``segment`` task uses the E02 classes.

    Every other task keeps the stock Ultralytics mapping, and the segment
    validator/predictor are untouched, so validation, prediction export and
    metric computation are the E01 code path.
    """

    @property
    def task_map(self) -> dict:
        """Return the stock task map with the segment model/trainer replaced."""
        task_map = dict(super().task_map)
        task_map["segment"] = {**task_map["segment"], "model": OrdinalSegmentationModel,
                               "trainer": OrdinalSegmentationTrainer}
        return task_map
