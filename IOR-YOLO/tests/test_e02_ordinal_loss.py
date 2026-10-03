"""E02 ordinal-loss mathematics: no Ultralytics, no training, no dataset access."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported as a package (not by file path) so the loss objects are the same
# module instances the E02 trainer uses and pickles reference.
from ior_yolo.losses import ordinal as loss  # noqa: E402


def confident_logits(stage, *, scale=8.0, count=1):
    """One-hot-ish logits that predict exactly ``stage``."""
    logits = torch.full((count, loss.NUM_STAGES), -scale)
    logits[:, stage] = scale
    return logits


class CumulativeEncodingTest(unittest.TestCase):
    def test_frozen_cumulative_targets_match_the_e02_spec(self):
        self.assertEqual(loss.NUM_STAGES, 3)
        self.assertEqual(loss.CUMULATIVE_TARGETS["immature apple"], (1.0, 1.0))
        self.assertEqual(loss.CUMULATIVE_TARGETS["semi-mature apple"], (0.0, 1.0))
        self.assertEqual(loss.CUMULATIVE_TARGETS["mature apple"], (0.0, 0.0))

    def test_target_cumulative_is_one_for_j_at_or_after_the_true_stage(self):
        stages = torch.tensor([0, 1, 2])
        self.assertEqual(loss.target_cumulative(3, stages).tolist(), [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

    def test_target_cumulative_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            loss.target_cumulative(1, torch.tensor([0]))
        with self.assertRaises(ValueError):
            loss.target_cumulative(3, torch.tensor([0.5]))
        with self.assertRaises(ValueError):
            loss.target_cumulative(3, torch.tensor([3]))
        with self.assertRaises(ValueError):
            loss.target_cumulative(3, torch.tensor([-1]))

    def test_predicted_cumulative_is_a_cdf_of_the_softmax(self):
        logits = torch.tensor([[2.0, 0.0, 0.0]])
        probabilities = logits.softmax(-1)[0]
        expected = [(probabilities[0]).item(), (probabilities[0] + probabilities[1]).item()]
        self.assertTrue(torch.allclose(loss.predicted_cumulative(logits)[0], torch.tensor(expected)))
        with self.assertRaises(ValueError):
            loss.predicted_cumulative(torch.zeros(1, 1))


class OrdinalPenaltyTest(unittest.TestCase):
    def test_perfect_prediction_is_zero_and_is_the_minimum(self):
        perfect = loss.ordinal_penalty(confident_logits(1), torch.tensor([1]))
        self.assertAlmostEqual(float(perfect[0]), 0.0, places=6)
        for perturbation in (torch.tensor([[8.0, 6.0, -8.0]]), torch.tensor([[0.0, 8.0, -8.0]]),
                             torch.tensor([[-8.0, 6.0, 8.0]])):
            self.assertGreater(float(loss.ordinal_penalty(perturbation, torch.tensor([1]))[0]), float(perfect[0]))

    def test_adjacent_error_is_cheaper_than_severe_error(self):
        adjacent = loss.ordinal_penalty(confident_logits(0), torch.tensor([1]))
        severe = loss.ordinal_penalty(confident_logits(0), torch.tensor([2]))
        self.assertAlmostEqual(float(adjacent[0]), 0.5, places=6)
        self.assertAlmostEqual(float(severe[0]), 1.0, places=6)
        self.assertLess(float(adjacent[0]), float(severe[0]))

    def test_penalty_decreases_as_correct_class_confidence_grows(self):
        stages = torch.tensor([2])
        penalties = [float(loss.ordinal_penalty(torch.tensor([[2.0, 1.0, value]]), stages)[0])
                     for value in (2.0, 6.0, 12.0)]
        self.assertGreater(penalties[0], penalties[1])
        self.assertGreater(penalties[1], penalties[2])
        self.assertGreater(penalties[1], 0.0)

    def test_penalty_is_bounded_and_finite_for_extreme_logits(self):
        stages = torch.tensor([0, 1, 2])
        for magnitude in (50.0, 200.0, 1000.0):
            logits = torch.tensor([[-magnitude, 0.0, magnitude], [magnitude, magnitude, magnitude],
                                   [0.0, 0.0, 0.0]])
            penalty = loss.ordinal_penalty(logits, stages)
            self.assertTrue(torch.isfinite(penalty).all())
            self.assertTrue(bool((penalty >= 0).all() and (penalty <= 1.0 + 1e-6).all()))

    def test_penalty_shape_validation(self):
        with self.assertRaises(ValueError):
            loss.ordinal_penalty(torch.zeros(3), torch.tensor([0]))
        with self.assertRaises(ValueError):
            loss.ordinal_penalty(torch.zeros(2, 3), torch.tensor([0]))

    def test_gradient_is_finite_and_reaches_every_class(self):
        logits = torch.tensor([[0.4, -0.2, 0.1]], requires_grad=True)
        loss.ordinal_penalty(logits, torch.tensor([2])).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(bool((logits.grad.abs() > 0).all()))


def one_hot_targets(stages, *, anchors):
    """Build ``(1, anchors, K)`` assigner-style targets for ``stages`` on the first anchors."""
    targets = torch.zeros(1, anchors, loss.NUM_STAGES)
    for index, stage in enumerate(stages):
        targets[0, index, stage] = 1.0
    return targets


def quality_targets(stages, qualities, *, anchors):
    """Pinned-assigner-style targets: one class scattered at the per-anchor quality ``w_i``."""
    targets = torch.zeros(1, anchors, loss.NUM_STAGES)
    for index, (stage, quality) in enumerate(zip(stages, qualities)):
        targets[0, index, stage] = quality
    return targets


class QualityWeightingTest(unittest.TestCase):
    """Audit A: ``L_ord = sum_i w_i * err_i / max(sum_i w_i, 1)`` with ``w_i = target_scores[i, y_i]``."""

    def test_closed_form_matches_the_implementation(self):
        logits = torch.randn(1, 4, 3)
        stages = [0, 1, 2, 1]
        qualities = [0.9, 0.4, 0.75, 0.2]
        targets = quality_targets(stages, qualities, anchors=4)
        mask = torch.ones(1, 4, dtype=torch.bool)
        component = loss.ordinal_component(logits, targets, mask)
        errors = loss.ordinal_penalty(logits[0], torch.tensor(stages))
        weights = torch.tensor(qualities)
        self.assertAlmostEqual(float(component["loss"]), float((weights * errors).sum() / weights.sum()), places=6)
        self.assertAlmostEqual(component["weight_sum"], float(weights.sum()), places=6)
        self.assertAlmostEqual(component["normalization"], float(weights.sum()), places=6)
        self.assertAlmostEqual(component["unscaled_denominator"], float(weights.sum()), places=6)
        self.assertAlmostEqual(float(component["penalty_sum_unweighted"]), float(errors.sum()), places=6)

    def test_denominator_and_numerator_share_the_same_weights(self):
        logits = torch.randn(2, 6, 3)
        stages = [2, 0, 1, 1, 0, 2, 2, 0, 1, 1, 0, 2]
        qualities = [0.9, 0.35, 0.5, 0.25, 0.8, 0.1, 0.9, 0.35, 0.5, 0.25, 0.8, 0.1]
        targets = torch.cat([quality_targets(stages[:6], qualities[:6], anchors=6),
                             quality_targets(stages[6:], qualities[6:], anchors=6)], dim=0)
        mask = torch.ones(2, 6, dtype=torch.bool)
        component = loss.ordinal_component(logits, targets, mask)
        weights = targets.sum(dim=-1)[mask]
        self.assertAlmostEqual(component["normalization"], float(weights.sum()), places=5)
        self.assertAlmostEqual(component["weight_sum"], float(weights.sum()), places=5)
        # The loss is the weighted numerator divided by the very same weight population.
        self.assertAlmostEqual(float(component["loss"]),
                               float(component["penalty_sum"]) / float(weights.sum()), places=6)

    def test_lowering_one_anchor_quality_lowers_its_own_contribution(self):
        logits = torch.cat([confident_logits(0), confident_logits(1)]).unsqueeze(0)
        mask = torch.ones(1, 2, dtype=torch.bool)
        high = loss.ordinal_component(logits, quality_targets([2, 1], [1.0, 1.0], anchors=2), mask)
        low = loss.ordinal_component(logits, quality_targets([2, 1], [0.1, 1.0], anchors=2), mask)
        errors = loss.ordinal_penalty(logits[0], torch.tensor([2, 1]))
        self.assertAlmostEqual(float(errors[0]), 1.0, places=6)  # severe error, the anchor under test
        self.assertGreater(1.0 * float(errors[0]) / high["normalization"],
                           0.1 * float(errors[0]) / low["normalization"])
        self.assertLess(float(low["loss"]), float(high["loss"]))
        self.assertLess(low["weight_sum"], high["weight_sum"])

    def test_equal_errors_with_different_quality_are_not_forced_to_equal_weight(self):
        logits = torch.cat([confident_logits(0), confident_logits(0), confident_logits(0)]).unsqueeze(0)
        targets = quality_targets([1, 1, 2], [1.0, 0.1, 0.5], anchors=3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        component = loss.ordinal_component(logits, targets, mask)
        errors = loss.ordinal_penalty(logits[0], torch.tensor([1, 1, 2]))
        self.assertAlmostEqual(float(errors[0]), float(errors[1]), places=6)  # identical ordinal errors
        weights = torch.tensor([1.0, 0.1, 0.5])
        contributions = weights * errors / component["normalization"]
        self.assertAlmostEqual(float(contributions[0] / contributions[1]), 10.0, places=5)
        forced_equal = float(errors.sum() / weights.sum())  # what an unweighted numerator would produce
        self.assertAlmostEqual(float(component["loss"]), float((weights * errors).sum() / weights.sum()), places=6)
        self.assertGreater(abs(float(component["loss"]) - forced_equal), 1e-3)
        self.assertNotAlmostEqual(float(component["penalty_sum"]), float(component["penalty_sum_unweighted"]), places=3)
    def test_perfect_prediction_stays_the_minimum_under_quality_weighting(self):
        stages = [0, 1, 2]
        targets = quality_targets(stages, [0.9, 0.4, 0.7], anchors=3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        perfect = loss.ordinal_component(torch.cat([confident_logits(stage) for stage in stages]).unsqueeze(0),
                                         targets, mask)
        self.assertAlmostEqual(float(perfect["loss"]), 0.0, places=6)
        for perturbed in (torch.tensor([[8.0, 6.0, -8.0]]), torch.tensor([[-8.0, 6.0, 8.0]]),
                          torch.tensor([[0.0, 0.0, 0.0]])):
            # Anchor 0 is deliberately mis-predicted, so any degraded batch exceeds the perfect one.
            degraded = loss.ordinal_component(torch.cat([confident_logits(1), perturbed]).unsqueeze(0),
                                              targets[:, :2], mask[:, :2])
            self.assertGreater(float(degraded["loss"]), float(perfect["loss"]))

    def test_adjacent_error_stays_cheaper_than_severe_error_under_quality_weighting(self):
        mask = torch.ones(1, 1, dtype=torch.bool)
        adjacent = loss.ordinal_component(confident_logits(0).unsqueeze(0),
                                          quality_targets([1], [1.0], anchors=1), mask)
        severe = loss.ordinal_component(confident_logits(0).unsqueeze(0),
                                        quality_targets([2], [1.0], anchors=1), mask)
        self.assertAlmostEqual(float(adjacent["loss"]), 0.5, places=6)
        self.assertAlmostEqual(float(severe["loss"]), 1.0, places=6)
        self.assertLess(float(adjacent["loss"]), float(severe["loss"]))
        adjacent_low = loss.ordinal_component(confident_logits(0).unsqueeze(0),
                                              quality_targets([1], [0.6], anchors=1), mask)
        severe_low = loss.ordinal_component(confident_logits(0).unsqueeze(0),
                                            quality_targets([2], [0.6], anchors=1), mask)
        self.assertLess(float(adjacent_low["loss"]), float(severe_low["loss"]))

    def test_batch_permutation_invariance_holds_with_quality_weights(self):
        logits = torch.randn(3, 4, 3)
        targets = torch.cat([quality_targets([0, 1], [0.9, 0.3], anchors=4),
                             quality_targets([2, 1], [0.5, 0.8], anchors=4),
                             quality_targets([1, 0], [0.2, 0.7], anchors=4)], dim=0)
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask[:, :2] = True
        first = loss.ordinal_component(logits, targets, mask)
        permutation = torch.tensor([2, 0, 1])
        second = loss.ordinal_component(logits[permutation], targets[permutation], mask[permutation])
        self.assertAlmostEqual(float(first["loss"]), float(second["loss"]), places=6)
        self.assertAlmostEqual(first["weight_sum"], second["weight_sum"], places=6)
        self.assertEqual(first["instances"], second["instances"])

    def test_loss_and_gradient_are_finite_with_quality_weights(self):
        logits = torch.randn(1, 3, 3, requires_grad=True)
        component = loss.ordinal_component(logits, quality_targets([0, 1, 2], [0.9, 0.05, 0.4], anchors=3),
                                           torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(torch.isfinite(component["loss"]).all())
        self.assertGreaterEqual(float(component["loss"]), 0.0)
        component["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_matched_quality_comes_from_the_targets_not_from_the_prediction(self):
        logits = torch.randn(1, 2, 3)
        mask = torch.ones(1, 2, dtype=torch.bool)
        low = loss.ordinal_component(logits, quality_targets([1, 1], [0.25, 0.75], anchors=2), mask)
        high = loss.ordinal_component(logits, quality_targets([1, 1], [0.95, 0.95], anchors=2), mask)
        self.assertAlmostEqual(low["quality_min"], 0.25, places=6)
        self.assertAlmostEqual(high["quality_max"], 0.95, places=6)
        self.assertAlmostEqual(low["weight_sum"], 1.0, places=6)
        self.assertAlmostEqual(high["weight_sum"], 1.9, places=6)
        # Identical logits -> identical unweighted errors, yet the weighted term still moves.
        self.assertAlmostEqual(float(low["penalty_sum_unweighted"]), float(high["penalty_sum_unweighted"]), places=6)
        self.assertNotAlmostEqual(float(low["loss"]), float(high["loss"]), places=4)

    def test_denominator_below_one_follows_the_stock_clamp(self):
        component = loss.ordinal_component(confident_logits(0).unsqueeze(0),
                                           quality_targets([1], [0.4], anchors=1),
                                           torch.ones(1, 1, dtype=torch.bool))
        self.assertAlmostEqual(component["normalization"], 1.0, places=6)
        self.assertAlmostEqual(component["weight_sum"], 0.4, places=6)
        self.assertAlmostEqual(float(component["loss"]), 0.4 * 0.5, places=6)  # numerator / max(0.4, 1)

    def test_zero_quality_anchor_is_not_a_matched_anchor(self):
        component = loss.ordinal_component(confident_logits(1).unsqueeze(0),
                                           quality_targets([1], [0.0], anchors=1),
                                           torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(component["instances"], 0)
        self.assertAlmostEqual(float(component["loss"]), 0.0, places=8)

    def test_several_non_zero_target_classes_are_refused(self):
        targets = torch.zeros(1, 1, loss.NUM_STAGES)
        targets[0, 0, 1] = 0.6
        targets[0, 0, 2] = 0.4
        with self.assertRaises(ValueError):
            loss.ordinal_component(torch.randn(1, 1, 3), targets, torch.ones(1, 1, dtype=torch.bool))

    def test_target_mass_outside_the_matched_mask_is_refused(self):
        targets = quality_targets([1], [0.6], anchors=2)  # mass sits on anchor 0 ...
        mask = torch.tensor([[False, True]])  # ... but anchor 1 is the foreground anchor
        with self.assertRaises(ValueError):
            loss.ordinal_component(torch.randn(1, 2, 3), targets, mask)

    def test_negative_quality_is_refused(self):
        with self.assertRaises(ValueError):
            loss.ordinal_component(torch.randn(1, 1, 3), quality_targets([1], [-0.5], anchors=1),
                                   torch.ones(1, 1, dtype=torch.bool))

    def test_quality_weights_helper_validates_input_and_reports_mass(self):
        targets = quality_targets([0, 2], [0.75, 0.25], anchors=3)
        mask = torch.tensor([[True, True, False]])
        weights, matched = loss.quality_weights(targets, mask)
        self.assertEqual(weights.shape, (1, 3))
        self.assertEqual(matched.tolist(), [[True, True, False]])
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        with self.assertRaises(ValueError):
            loss.quality_weights(targets[0], mask)
        with self.assertRaises(ValueError):
            loss.quality_weights(targets, mask.to(torch.int64))



class PinnedAssignerDenominatorTest(unittest.TestCase):
    """Audit B: the denominator is the stock ``target_scores_sum`` of the pinned cls loss.

    In pinned Ultralytics 8.3.220 (``ultralytics/utils/loss.py``, ``v8SegmentationLoss``)
    the classification objective is ``BCEWithLogits(pred_scores, target_scores).sum() /
    target_scores_sum`` with ``target_scores_sum = max(target_scores.sum(), 1)``, and
    ``BboxLoss`` weights its per-instance terms by ``target_scores.sum(-1)[fg_mask]``.
    The assigner writes ``target_scores[i, assigned_class_i] = w_i`` on matched anchors
    only, so the ordinal term must use that same per-anchor mass in the numerator and
    divide by that same ``target_scores_sum``.
    """

    def test_quality_weight_is_the_assigned_class_target_score(self):
        # w_i = target_scores[i, assigned_class_i]: read from the targets, never re-derived
        # from the prediction, and identical to target_scores[i].sum() because exactly one
        # class carries the mass per matched anchor.
        targets = quality_targets([0, 2, 1], [0.9, 0.25, 0.6], anchors=4)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, :3] = True
        weights, matched = loss.quality_weights(targets, mask)
        assigned = targets.argmax(dim=-1)
        self.assertEqual(assigned.tolist(), [[0, 2, 1, 0]])
        for index, quality in enumerate((0.9, 0.25, 0.6)):
            self.assertAlmostEqual(float(weights[0, index]),
                                   float(targets[0, index, assigned[0, index]]), places=6)
            self.assertAlmostEqual(float(weights[0, index]), quality, places=6)
            self.assertAlmostEqual(float(weights[0, index]), float(targets[0, index].sum()), places=6)
        self.assertEqual(matched.tolist(), [[True, True, True, False]])
        self.assertFalse(bool(matched.requires_grad))  # targets are constants, not a gradient path

    def test_default_denominator_is_the_stock_target_scores_sum(self):
        logits = torch.randn(2, 5, 3, requires_grad=True)
        targets = torch.cat([quality_targets([0, 1], [0.8, 0.3], anchors=5),
                             quality_targets([2], [0.5], anchors=5)], dim=0)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[0, :2] = True
        mask[1, 0] = True
        component = loss.ordinal_component(logits, targets, mask)
        stock_denominator = float(max(targets.sum(), torch.tensor(1.0)))  # pinned target_scores_sum
        self.assertAlmostEqual(component["unscaled_denominator"], float(targets.sum()), places=5)
        self.assertAlmostEqual(component["normalization"], stock_denominator, places=5)
        # The mass the numerator is weighted by is the mass the denominator is built from.
        self.assertAlmostEqual(component["normalization"], component["weight_sum"], places=5)

    def test_numerator_and_denominator_share_the_same_weight_population(self):
        # Closed form of the frozen principle, evaluated straight from the assigner targets:
        #   L_ord = sum_i w_i * err_i / max(sum_i w_i, 1)
        logits = torch.randn(1, 6, 3)
        stages = [0, 1, 2, 2, 1, 0]
        qualities = [0.9, 0.2, 0.55, 0.7, 0.35, 0.15]
        targets = quality_targets(stages, qualities, anchors=6)
        mask = torch.ones(1, 6, dtype=torch.bool)
        component = loss.ordinal_component(logits, targets, mask)
        weights = targets.sum(-1)[mask]  # the per-anchor mass used by numerator and denominator
        errors = loss.ordinal_penalty(logits[mask], torch.tensor(stages))
        stock_denominator = max(float(targets.sum()), 1.0)
        self.assertAlmostEqual(float(component["loss"]),
                               float((weights * errors).sum()) / stock_denominator, places=6)
        self.assertAlmostEqual(component["weight_sum"], float(weights.sum()), places=5)
        # An unweighted numerator over the same denominator is a different, wrong objective.
        self.assertNotAlmostEqual(float(component["loss"]), float(errors.sum()) / stock_denominator,
                                  places=3)

    def test_lower_quality_anchor_contributes_less_at_a_fixed_denominator(self):
        # Hold the denominator fixed so the comparison isolates the per-anchor w_i.
        logits = torch.cat([confident_logits(0), confident_logits(1)]).unsqueeze(0)
        mask = torch.ones(1, 2, dtype=torch.bool)
        errors = loss.ordinal_penalty(logits[0], torch.tensor([2, 1]))
        self.assertAlmostEqual(float(errors[0]), 1.0, places=6)  # severe error on the anchor under test
        self.assertAlmostEqual(float(errors[1]), 0.0, places=6)  # exact prediction
        high = loss.ordinal_component(logits, quality_targets([2, 1], [1.0, 1.0], anchors=2), mask,
                                      normalization=2.0)
        low = loss.ordinal_component(logits, quality_targets([2, 1], [0.1, 1.0], anchors=2), mask,
                                     normalization=2.0)
        self.assertAlmostEqual(high["normalization"], low["normalization"], places=6)
        self.assertAlmostEqual(float(high["loss"]), 1.0 * 1.0 / 2.0, places=6)
        self.assertAlmostEqual(float(low["loss"]), 0.1 * 1.0 / 2.0, places=6)
        self.assertLess(float(low["loss"]), float(high["loss"]))
        self.assertAlmostEqual(float(high["penalty_sum"] - low["penalty_sum"]), 0.9 * 1.0, places=6)
        # The same ordering must hold at the stock denominator.
        stock_high = loss.ordinal_component(logits, quality_targets([2, 1], [1.0, 1.0], anchors=2), mask)
        stock_low = loss.ordinal_component(logits, quality_targets([2, 1], [0.1, 1.0], anchors=2), mask)
        self.assertLess(float(stock_low["loss"]), float(stock_high["loss"]))
        self.assertLess(float(stock_low["penalty_sum"]), float(stock_high["penalty_sum"]))


class OrdinalComponentTest(unittest.TestCase):
    def test_normalization_matches_the_original_classification_loss(self):
        logits = torch.randn(1, 6, 3)
        targets = one_hot_targets([0, 1, 2], anchors=6)
        mask = torch.zeros(1, 6, dtype=torch.bool)
        mask[0, :3] = True
        component = loss.ordinal_component(logits, targets, mask)
        stages = targets.argmax(-1)[mask]
        expected = loss.ordinal_penalty(logits[mask], stages).sum() / float(targets.sum())
        self.assertAlmostEqual(float(component["loss"]), float(expected), places=6)
        self.assertEqual(component["instances"], 3)
        self.assertAlmostEqual(component["normalization"], 3.0, places=6)
        # One-hot targets carry w_i = 1, so the quality-weighted form and the
        # unweighted form coincide; the equality above is therefore only a
        # special case of the weighted definition (see QualityWeightingTest).
        self.assertAlmostEqual(component["weight_sum"], 3.0, places=6)
        self.assertAlmostEqual(float(component["penalty_sum"]), float(component["penalty_sum_unweighted"]), places=6)

    def test_unmatched_anchors_do_not_contribute(self):
        logits = torch.randn(1, 8, 3)
        targets = one_hot_targets([1], anchors=8)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 0] = True
        component = loss.ordinal_component(logits, targets, mask)
        expected = loss.ordinal_penalty(logits[0, :1], torch.tensor([1]))
        self.assertAlmostEqual(float(component["loss"]), float(expected.sum()), places=6)
        self.assertEqual(component["instances"], 1)

    def test_foreground_mask_without_targets_yields_a_differentiable_zero(self):
        logits = torch.randn(1, 4, 3, requires_grad=True)
        targets = torch.zeros(1, 4, 3)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, 0] = True
        component = loss.ordinal_component(logits, targets, mask)
        self.assertEqual(component["instances"], 0)
        self.assertAlmostEqual(float(component["loss"]), 0.0, places=8)
        component["loss"].backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_empty_batch_is_finite_and_zero(self):
        logits = torch.randn(2, 5, 3, requires_grad=True)
        targets = torch.zeros(2, 5, 3)
        mask = torch.zeros(2, 5, dtype=torch.bool)
        component = loss.ordinal_component(logits, targets, mask)
        self.assertEqual(component["instances"], 0)
        self.assertAlmostEqual(float(component["loss"]), 0.0, places=8)
        self.assertAlmostEqual(component["normalization"], 1.0, places=8)

    def test_batch_permutation_does_not_change_the_component(self):
        logits = torch.randn(3, 4, 3)
        targets = torch.zeros(3, 4, 3)
        for batch, stages in enumerate(((0, 1), (2, 1), (1, 0))):
            targets[batch, 0, stages[0]] = 1.0
            targets[batch, 1, stages[1]] = 1.0
        mask = torch.zeros(3, 4, dtype=torch.bool)
        mask[:, :2] = True
        first = loss.ordinal_component(logits, targets, mask)
        permutation = torch.tensor([2, 0, 1])
        second = loss.ordinal_component(logits[permutation], targets[permutation], mask[permutation])
        self.assertAlmostEqual(float(first["loss"]), float(second["loss"]), places=6)
        self.assertEqual(first["instances"], second["instances"])

    def test_gradient_is_finite_and_non_zero(self):
        logits = torch.randn(1, 4, 3, requires_grad=True)
        targets = one_hot_targets([0, 2], anchors=4)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        mask[0, :2] = True
        component = loss.ordinal_component(logits, targets, mask)
        component["loss"].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_shape_and_dtype_validation(self):
        logits = torch.zeros(1, 4, 3)
        targets = one_hot_targets([0], anchors=4)
        mask = torch.zeros(1, 4, dtype=torch.bool)
        with self.assertRaises(ValueError):
            loss.ordinal_component(logits[0], targets, mask)
        with self.assertRaises(ValueError):
            loss.ordinal_component(logits, targets[:, :2], mask)
        with self.assertRaises(ValueError):
            loss.ordinal_component(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4), mask)
        with self.assertRaises(ValueError):
            loss.ordinal_component(logits, targets, mask.to(torch.int64))
        with self.assertRaises(ValueError):
            loss.ordinal_component(logits, targets, torch.zeros(1, 3, dtype=torch.bool))

    def test_describe_component_returns_plain_floats(self):
        logits = torch.randn(1, 3, 3)
        targets = one_hot_targets([0, 1, 2], anchors=3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        described = loss.describe_component(loss.ordinal_component(logits, targets, mask))
        self.assertEqual(sorted(described), ["instances", "loss", "normalization", "penalty_max", "penalty_mean",
                                             "penalty_sum", "penalty_sum_unweighted", "quality_max", "quality_mean",
                                             "quality_min", "unscaled_denominator", "weight_sum", "weighting"])
        self.assertIsInstance(described["loss"], float)
        self.assertIsInstance(described["instances"], int)
        self.assertEqual(described["weighting"], loss.ORDINAL_WEIGHTING)


class MergeAndConfigTest(unittest.TestCase):
    def test_total_loss_adds_lambda_times_component_scaled_by_batch(self):
        config = loss.OrdinalLossConfig()
        total, item = loss.total_loss_with_ordinal(torch.tensor(2.0), {"loss": torch.tensor(0.4)}, config,
                                                   batch_size=8)
        self.assertAlmostEqual(float(total), 2.0 + 0.5 * 0.4 * 8, places=6)
        self.assertAlmostEqual(float(item), 0.5 * 0.4, places=6)
        self.assertFalse(item.requires_grad)

    def test_component_wise_total_gains_the_auxiliary_term_exactly_once(self):
        # Pinned 8.3.220 returns the batch-scaled total as (box, seg, cls, dfl) and the trainer
        # optimises ``loss.sum()``; a plain broadcast would apply the term once per component.
        stock = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        total, item = loss.total_loss_with_ordinal(stock, {"loss": torch.tensor(0.4)},
                                                   loss.OrdinalLossConfig(), batch_size=8)
        self.assertEqual(total.shape, stock.shape)
        self.assertAlmostEqual(float(total.sum().detach()), 10.0 + 0.5 * 0.4 * 8, places=6)
        self.assertAlmostEqual(float(total[loss.ORDINAL_LOSS_SLOT].detach()), 3.0 + 0.5 * 0.4 * 8, places=6)
        for index in (0, 1, 3):
            self.assertAlmostEqual(float(total[index].detach()), float(stock[index].detach()), places=6)
        self.assertAlmostEqual(float(item), 0.2, places=6)
        self.assertAlmostEqual(float(stock.sum().detach()), 10.0, places=6)  # stock vector untouched
        total.sum().backward()
        self.assertTrue(torch.equal(stock.grad, torch.ones(4)))

    def test_component_wise_total_rejects_an_unknown_component_layout(self):
        with self.assertRaises(ValueError):
            loss.total_loss_with_ordinal(torch.tensor([1.0, 2.0]), {"loss": torch.tensor(0.4)},
                                         loss.OrdinalLossConfig(), batch_size=8)

    def test_total_loss_rejects_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            loss.total_loss_with_ordinal(torch.tensor(1.0), {"loss": torch.tensor(0.0)},
                                         loss.OrdinalLossConfig(), batch_size=0)

    def test_lambda_is_the_frozen_first_round_value(self):
        self.assertEqual(loss.E02_LAMBDA_ORD, 0.5)
        self.assertEqual(loss.OrdinalLossConfig().lambda_ord, 0.5)
        self.assertEqual(len(loss.STAGE_ORDER), loss.NUM_STAGES)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            loss.OrdinalLossConfig(lambda_ord=0.0)
        with self.assertRaises(ValueError):
            loss.OrdinalLossConfig(lambda_ord=1)
        with self.assertRaises(ValueError):
            loss.OrdinalLossConfig(num_stages=2, stage_order=("immature apple", "mature apple"))
        with self.assertRaises(ValueError):
            loss.OrdinalLossConfig(stage_order=("mature apple", "semi-mature apple", "immature apple"))
        described = loss.OrdinalLossConfig().as_dict()
        self.assertEqual(described["lambda_ord"], 0.5)
        self.assertEqual(described["stage_order"], list(loss.STAGE_ORDER))
        self.assertIn("OrdinalSegmentationLoss", described["injection_point"])


if __name__ == "__main__":
    unittest.main()
