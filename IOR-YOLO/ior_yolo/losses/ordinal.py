"""Ordinal-aware auxiliary loss for the E02 apple-maturity experiment.

E02 hypothesis: the three D2 maturity stages are *ordered*
(immature < semi-mature < mature), while the stock Ultralytics classification
objective (BCEWithLogitsLoss over task-aligned targets) treats them as three
independent categories. This module contains the ordinal auxiliary term only.

Mathematical definition (K = 3, ascending order immature=0 < semi-mature=1 <
mature=2):

    p            = softmax(z)                       # classification logits z
    F_p(j)       = sum_{k <= j} p_k                 # predicted cumulative, j = 0 .. K-2
    F_y(j)       = 1[j >= y]                        # target cumulative mass at stage y
    err_i        = mean_j (F_p^(i)(j) - F_y^(i)(j))^2
    w_i          = target_scores[i].sum()           # task-aligned target score of anchor i
    L_ord        = sum_i w_i * err_i / max(sum_i w_i, 1)      # i over matched anchors

``F_y`` follows the frozen E02 encoding: immature -> (1, 1), semi-mature ->
(0, 1), mature -> (0, 0). A single instance penalty is bounded by 1: 0 for an
exact one-hot prediction, 0.5 for a fully confident adjacent error (K = 3) and
1.0 for a fully confident severe (off-by-two) error.

Quality weighting (pre-commit protocol audit, 2026-10-03). The pinned
Ultralytics 8.3.220 target is *not* one-hot on matched anchors. In
``TaskAlignedAssigner.forward`` the one-hot label scatter is multiplied by
``norm_align_metric``, so ``target_scores[i, assigned_class_i] = w_i`` with
``w_i = (sigmoid(score_i)^alpha * IoU_i^beta) / max_j align_metric_j``
(``ultralytics/utils/tal.py``). The stock classification loss divides by
``target_scores_sum = sum_i w_i``, and ``BboxLoss.forward`` weights each
per-instance term by exactly that same quantity
(``weight = target_scores.sum(-1)[fg_mask]``). The ordinal auxiliary term
therefore weights every matched anchor by the ``w_i`` it shares the denominator
with: numerator and denominator use one single per-anchor target mass, and the
term keeps the scale convention of the loss it extends. An unweighted numerator
over the weighted denominator would up-weight low-quality anchors and would not
be comparable to the term it extends.

This module deliberately imports no Ultralytics code and touches no model: it is
the reviewable mathematical core, separately unit-tested. The injection into the
pinned Ultralytics training loop lives in ``ior_yolo/trainers/ordinal.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Frozen D2 class order of the E01/E02 derived dataset (dataset.yaml names).
STAGE_ORDER: tuple[str, ...] = ("immature apple", "semi-mature apple", "mature apple")
NUM_STAGES: int = len(STAGE_ORDER)
# F_y(j) = 1[j >= y] for j = 0 .. K-2, keyed by the frozen class name.
CUMULATIVE_TARGETS: dict[str, tuple[float, ...]] = {
    "immature apple": (1.0, 1.0),
    "semi-mature apple": (0.0, 1.0),
    "mature apple": (0.0, 0.0),
}
# First-round E02 value. Predeclared before training; never tuned on val or test.
E02_LAMBDA_ORD: float = 0.5
# Per-anchor weighting of the auxiliary term. Frozen: it mirrors the pinned
# Ultralytics target semantics, so it is a property of the loss, not a knob.
ORDINAL_WEIGHTING = "task_aligned_target_score"
# Slot of the classification term in the pinned component-wise total (box, seg, cls, dfl).
# The auxiliary term is added here exactly once, because the trainer optimises the vector sum.
ORDINAL_LOSS_SLOT = 2
# Tolerance for the structural invariant "one non-zero target class per matched anchor".
SINGLE_TARGET_CLASS_ATOL = 1e-6


@dataclass(frozen=True)
class OrdinalLossConfig:
    """Validated E02 ordinal-loss settings.

    ``lambda_ord`` is the weight of the auxiliary term in
    ``L_total = L_original_YOLO + lambda_ord * L_ord``. It is part of the frozen
    experimental protocol: it must be taken from the E02 config, never inferred
    from validation or test results.
    """

    num_stages: int = NUM_STAGES
    lambda_ord: float = E02_LAMBDA_ORD
    stage_order: tuple[str, ...] = field(default=STAGE_ORDER)

    def __post_init__(self) -> None:
        if self.num_stages < 2:
            raise ValueError(f"ordinal loss needs at least 2 ordered stages, got {self.num_stages}")
        if len(self.stage_order) != self.num_stages:
            raise ValueError(
                f"stage_order has {len(self.stage_order)} entries but num_stages is {self.num_stages}")
        if self.num_stages != NUM_STAGES:
            raise ValueError(f"E02 pins {NUM_STAGES} maturity stages, got {self.num_stages}")
        if not isinstance(self.lambda_ord, float) or self.lambda_ord <= 0.0:
            raise ValueError(f"lambda_ord must be a positive float, got {self.lambda_ord!r}")
        if tuple(self.stage_order) != STAGE_ORDER:
            raise ValueError(f"stage order must stay {STAGE_ORDER}, got {tuple(self.stage_order)}")

    def as_dict(self) -> dict:
        """Return a YAML/JSON-safe description for configs, manifests and logs."""
        return {"loss": "cumulative_distribution_squared_error",
                "num_stages": self.num_stages,
                "stage_order": list(self.stage_order),
                "lambda_ord": self.lambda_ord,
                "weighting": ORDINAL_WEIGHTING,
                "normalization": "sum_i w_i * err_i / max(sum_i w_i, 1) with w_i = task-aligned target "
                                 "score; the denominator is the target_scores_sum of the original "
                                 "Ultralytics cls loss, the numerator uses the same per-anchor w_i",
                "injection_point": "ior_yolo/trainers/ordinal.py::OrdinalSegmentationLoss.__call__"}


def target_cumulative(num_stages: int, stage_index: torch.Tensor) -> torch.Tensor:
    """Return ``F_y(j) = 1[j >= y]`` with shape ``stage_index.shape + (num_stages - 1,)``."""
    if num_stages < 2:
        raise ValueError(f"num_stages must be >= 2, got {num_stages}")
    if stage_index.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"stage_index must be an integer tensor, got {stage_index.dtype}")
    if bool((stage_index < 0).any()) or bool((stage_index >= num_stages).any()):
        raise ValueError(f"stage_index out of range [0, {num_stages - 1}]")
    j = torch.arange(num_stages - 1, device=stage_index.device)
    return (j >= stage_index.unsqueeze(-1)).to(torch.float32)


def predicted_cumulative(logits: torch.Tensor) -> torch.Tensor:
    """Return ``F_p(j) = sum_{k <= j} softmax(logits)_k`` for ``j = 0 .. K-2``."""
    if logits.shape[-1] < 2:
        raise ValueError(f"last dimension must hold K >= 2 classes, got {logits.shape[-1]}")
    return logits.softmax(dim=-1).cumsum(dim=-1)[..., :-1]


def ordinal_penalty(logits: torch.Tensor, stage_index: torch.Tensor) -> torch.Tensor:
    """Per-instance ``mean_j (F_p(j) - F_y(j))^2`` for ``(N, K)`` logits and ``(N,)`` stages.

    The penalty is bounded by 1: it is 0 for an exact one-hot prediction of the
    true stage, 0.5 for a fully confident adjacent-stage error (K = 3) and 1.0
    for a fully confident severe (off-by-two) error.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (N, K), got {tuple(logits.shape)}")
    if stage_index.dim() != 1 or stage_index.shape[0] != logits.shape[0]:
        raise ValueError(f"stage_index must be (N,) matching {tuple(logits.shape)}, got {tuple(stage_index.shape)}")
    cumulative_error = predicted_cumulative(logits) - target_cumulative(logits.shape[-1], stage_index)
    return (cumulative_error ** 2).mean(dim=-1)


def quality_weights(target_scores: torch.Tensor,
                    positive_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-anchor task-aligned quality ``w_i`` and the anchors that carry it.

    ``w_i = target_scores[i].sum()`` is the target mass the pinned assigner left
    on anchor ``i``. Because exactly one class is scattered per matched anchor,
    this equals ``target_scores[i, assigned_class_i]``; that structural invariant
    is verified here (cheaply) so a different assignment contract fails loudly
    instead of silently mis-weighting the auxiliary term.

    Returns ``(weights, matched)`` with ``weights`` of shape ``(B, A)`` and
    ``matched = positive_mask & (weights > 0)``.
    """
    if target_scores.dim() != 3:
        raise ValueError(f"target_scores must be (B, A, K), got {tuple(target_scores.shape)}")
    if positive_mask.shape != target_scores.shape[:2] or positive_mask.dtype is not torch.bool:
        raise ValueError(f"positive_mask must be a bool tensor of shape {tuple(target_scores.shape[:2])}, "
                         f"got {positive_mask.dtype} {tuple(positive_mask.shape)}")
    weights = target_scores.detach().sum(dim=-1).to(torch.float32)
    if bool((weights < 0).any()):
        raise ValueError("target_scores must be non-negative task-aligned targets")
    matched = positive_mask & (weights > 0)
    if bool(matched.any()):
        non_zero_classes = (target_scores.detach() > 0).sum(dim=-1)
        if bool((non_zero_classes[matched] > 1).any()):
            raise ValueError("a matched anchor carries more than one non-zero target class; the pinned "
                             "task-aligned assignment contract changed, so w_i can no longer be read as "
                             "the assigned-class target score")
    return weights, matched


def ordinal_component(logits: torch.Tensor,
                      target_scores: torch.Tensor,
                      positive_mask: torch.Tensor,
                      *,
                      num_stages: int = NUM_STAGES,
                      normalization: float | torch.Tensor | None = None) -> dict:
    """Ordinal auxiliary component of one forward pass, before ``lambda_ord``.

    ``logits`` is the raw classification logits ``(B, A, K)`` of the segmentation
    head, ``target_scores`` the task-aligned assigner target ``(B, A, K)`` and
    ``positive_mask`` the assigner foreground mask ``(B, A)``. Only matched
    anchors carry a maturity label, so only they contribute - exactly the anchors
    the original classification loss is scored on.

    Each per-instance error is weighted by the task-aligned target score
    ``w_i = target_scores[i, assigned_class_i]`` - the same quantity the pinned
    classification loss and ``BboxLoss`` weight by - and the denominator is the
    ``target_scores_sum`` the stock classification loss divides by, so numerator
    and denominator share one single per-anchor mass. ``w_i`` is read from
    ``target_scores`` and never re-derived from ``logits``.

    Returns the normalised component ``loss`` (``L_ord``), the weighted
    ``penalty_sum`` (the numerator), its unweighted counterpart, ``instances``,
    per-anchor quality statistics and further float diagnostics. A batch without
    any matched anchor yields a differentiable zero tied to ``logits`` so the
    returned tensors keep the same shape in every step.
    """
    if logits.dim() != 3 or target_scores.dim() != 3:
        raise ValueError(f"logits and target_scores must be (B, A, K), got {tuple(logits.shape)} and "
                         f"{tuple(target_scores.shape)}")
    if logits.shape != target_scores.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != target_scores shape {tuple(target_scores.shape)}")
    if logits.shape[-1] != num_stages:
        raise ValueError(f"ordinal loss configured for {num_stages} stages but logits hold {logits.shape[-1]} classes")
    if positive_mask.shape != logits.shape[:2]:
        raise ValueError(f"positive_mask must be {tuple(logits.shape[:2])}, got {tuple(positive_mask.shape)}")
    if positive_mask.dtype is not torch.bool:
        raise ValueError(f"positive_mask must be a bool tensor, got {positive_mask.dtype}")

    weights, matched = quality_weights(target_scores, positive_mask)
    if normalization is None:
        scale = torch.as_tensor(target_scores.detach().sum(), dtype=torch.float32, device=logits.device)
        # The default denominator is the stock target_scores_sum, so every anchor
        # carrying target mass must be a matched anchor; otherwise numerator and
        # denominator would not describe the same population.
        matched_mass = weights[matched].sum()
        if not bool(torch.isclose(scale, matched_mass, rtol=1e-5, atol=1e-6)):
            raise ValueError("target mass found outside the matched anchors "
                             f"(sum(target_scores)={float(scale):.6f}, matched mass={float(matched_mass):.6f}); "
                             "the numerator/denominator populations must be identical")
    else:
        scale = torch.as_tensor(normalization, dtype=torch.float32, device=logits.device).detach()
    denominator = torch.clamp(scale, min=1.0)

    instances = int(matched.sum())
    if instances == 0:
        zero = logits.sum() * 0.0  # differentiable zero; keeps the graph shape stable
        return {"loss": zero, "penalty_sum": zero, "penalty_sum_unweighted": zero, "weight_sum": 0.0,
                "instances": 0, "normalization": float(denominator), "weighting": ORDINAL_WEIGHTING,
                "unscaled_denominator": float(scale), "penalty_mean": None, "penalty_max": None,
                "quality_mean": None, "quality_min": None, "quality_max": None}

    stages = target_scores.argmax(dim=-1)
    # float32 for numerical stability under AMP; gradients still reach the logits.
    penalty = ordinal_penalty(logits[matched].float(), stages[matched])
    anchor_weight = weights[matched]
    # Weighted numerator: one w_i per matched anchor, the same mass as the denominator.
    penalty_sum = (penalty * anchor_weight).sum()
    weight_sum = anchor_weight.sum()
    return {"loss": penalty_sum / denominator, "penalty_sum": penalty_sum,
            "penalty_sum_unweighted": penalty.detach().sum(), "weight_sum": float(weight_sum.detach()),
            "instances": instances, "normalization": float(denominator), "weighting": ORDINAL_WEIGHTING,
            "unscaled_denominator": float(scale),
            "penalty_mean": float((penalty_sum / weight_sum).detach()),
            "penalty_max": float(penalty.detach().max()),
            "quality_mean": float(anchor_weight.detach().mean()),
            "quality_min": float(anchor_weight.detach().min()),
            "quality_max": float(anchor_weight.detach().max())}


def describe_component(component: dict) -> dict:
    """Reduce :func:`ordinal_component` output to plain floats for logs/manifests."""
    return {"loss": float(component["loss"].detach()),
            "penalty_sum": float(component["penalty_sum"].detach()),
            "penalty_sum_unweighted": float(component["penalty_sum_unweighted"].detach()),
            "weight_sum": float(component["weight_sum"]), "instances": int(component["instances"]),
            "normalization": float(component["normalization"]), "weighting": component["weighting"],
            "unscaled_denominator": float(component["unscaled_denominator"]),
            "penalty_mean": component["penalty_mean"], "penalty_max": component["penalty_max"],
            "quality_mean": component["quality_mean"], "quality_min": component["quality_min"],
            "quality_max": component["quality_max"]}


def total_loss_with_ordinal(original_total: torch.Tensor, component: dict, config: OrdinalLossConfig,
                            *, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge the auxiliary term into the stock Ultralytics total loss.

    Ultralytics returns ``loss * batch_size`` with every component already
    normalised, so the auxiliary term is scaled identically to stay comparable
    across batch compositions. The returned logged item is ``lambda_ord * L_ord``
    (weight applied, batch scaling not), matching how the stock loss items are
    reported in ``results.csv`` and TensorBoard.

    In pinned 8.3.220 (``v8SegmentationLoss``) the returned total is *not* a scalar
    but the per-component vector ``(box, seg, cls, dfl) * batch_size``, and
    ``engine/trainer.py`` optimises its sum (``self.loss = loss.sum()`` before
    ``backward()``). Adding the scalar auxiliary term to that vector would apply it
    once per component - four times here - so it is added exactly once, to the
    classification slot it extends. The optimised objective remains
    ``sum(L_original_YOLO) + lambda_ord * L_ord * batch_size``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    weighted = config.lambda_ord * component["loss"]
    scaled = weighted * batch_size
    if original_total.dim() == 0:
        merged = original_total + scaled
    else:
        if original_total.numel() < ORDINAL_LOSS_SLOT + 1:
            raise ValueError(f"a component-wise total must hold at least {ORDINAL_LOSS_SLOT + 1} entries "
                             f"(box, seg, cls, dfl), got shape {tuple(original_total.shape)}")
        merged = original_total.clone()
        merged[ORDINAL_LOSS_SLOT] = merged[ORDINAL_LOSS_SLOT] + scaled
    return merged, weighted.detach()
