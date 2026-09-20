"""Synthetic counterexamples for the independent D2 verifier."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "08_verify_d2_groups.py"
spec = importlib.util.spec_from_file_location("d2_independent_verifier", SOURCE)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerificationRulesTest(unittest.TestCase):
    def test_conservative_current_and_worst_case_are_distinct(self):
        nodes = ["train/a.jpg", "val/a_brightness.jpg", "test/z.jpg", "val/w.jpg"]
        edges = [
            {"a": nodes[0], "b": nodes[1], "kind": "named", "corr64": .999,
             "same_regions": True},
            {"a": nodes[1], "b": nodes[2], "kind": "visual", "dhash": 4,
             "corr64": .996, "full_corr": .98},
            {"a": nodes[2], "b": nodes[3], "kind": "visual", "dhash": 5,
             "corr64": .994, "full_corr": .98},
        ]
        self.assertEqual(len(verifier.partition(nodes, edges, mode="conservative")), 3)
        self.assertEqual(len(verifier.partition(nodes, edges)), 2)
        self.assertEqual(len(verifier.partition(nodes, edges, mode="worst")), 1)
        self.assertEqual(len(verifier.partition(nodes, edges, visual_corr=.997)), 3)

    def test_exact_duplicate_merges_even_in_conservative_mode(self):
        nodes = ["train/x.jpg", "test/y.jpg"]
        edge = {"a": nodes[0], "b": nodes[1], "kind": "exact", "sha256": "same"}
        for mode in ("conservative", "current", "worst"):
            self.assertEqual(len(verifier.partition(nodes, [edge], mode=mode)), 1)

    def test_candidate_never_merges_in_current(self):
        nodes = ["train/x.jpg", "test/x_brightness.jpg"]
        edge = {"a": nodes[0], "b": nodes[1], "kind": "named", "corr64": .994,
                "same_regions": True}
        self.assertEqual(verifier.current_tier(edge), "Candidate")
        self.assertEqual(len(verifier.partition(nodes, [edge])), 2)

    def test_block_ssim_identical_is_one(self):
        from PIL import Image

        image = Image.new("RGB", (64, 64), (75, 100, 125))
        self.assertAlmostEqual(verifier.blocks_ssim(image, image), 1.0, places=12)
        self.assertAlmostEqual(verifier.hist_affinity(image, image), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
