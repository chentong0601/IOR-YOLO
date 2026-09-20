"""Low-cost Stage 2B-5P guard, pool and U01 policy checks."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze = load_script("10_prepare_d2_freeze_candidate")
probe = load_script("11_probe_u01_evaluator")


class FreezeCandidateTest(unittest.TestCase):
    def test_guard_closure_does_not_claim_same_source(self):
        graph = freeze.Components(("sg-a", "sg-b", "sg-c"))
        graph.union("sg-a", "sg-b")
        graph.union("sg-b", "sg-c")
        self.assertEqual(graph.find("sg-a"), graph.find("sg-c"))
        self.assertEqual(freeze.seed_from_archive_sha("00" * 32),
                         freeze.seed_from_archive_sha("00" * 32))
        self.assertNotEqual(freeze.seed_from_archive_sha("00" * 32),
                            freeze.seed_from_archive_sha("01" * 32))

    def test_u01_synthetic_matching_exposes_fp_sensitivity(self):
        target = [("mature apple", (0, 0, 10, 10))]
        predictions = [target[0], ("immature apple", (20, 20, 25, 25))]
        self.assertEqual(probe.evaluate(target, predictions),
                         {"tp": 1, "fp": 1, "fn": 0, "ignored_predictions": 0})
        self.assertEqual(probe.evaluate(target, predictions, ignore_boxes=[(20, 20, 25, 25)]),
                         {"tp": 1, "fp": 0, "fn": 0, "ignored_predictions": 1})

    def test_real_candidate_manifests_deterministic_if_available(self):
        archive = ROOT / "data" / "raw" / "multistage_apple_v4" / "dataset-20260508.zip"
        manifests = ROOT / "data" / "manifests"
        if not archive.is_file():
            self.skipTest("audited D2 archive absent")
        names = ("split_guard_clusters.csv", "d2_experiment_pool_candidate.csv")
        digests = []
        summaries = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                for name in ("source_groups.csv", "files_sha256.csv", "source_group_relations.csv"):
                    shutil.copyfile(manifests / name, target / name)
                summaries.append(freeze.build(archive, target))
                digests.append(tuple(hashlib.sha256((target / name).read_bytes()).hexdigest()
                                     for name in names))
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(digests[0], tuple(hashlib.sha256((manifests / name).read_bytes()).hexdigest()
                                          for name in names))
        summary = summaries[0]
        self.assertEqual(summary["guard"]["split_guard_clusters"], 1096)
        self.assertEqual(summary["guard"]["guarded_dhash_candidate_edges"], 23)
        self.assertEqual(summary["guard"]["guarded_other_candidate_edges"], 6)
        self.assertEqual(summary["guard"]["unguarded_residual_risk_edges"], 0)
        self.assertEqual(summary["pool"]["plain_base_representations"], 1119)
        self.assertEqual(summary["pool"]["included_candidate_images"], 1099)
        self.assertEqual(summary["pool"]["excluded_conflict_source_groups"], 7)
        self.assertEqual(len(summary["pool"]["groups_without_base"]), 6)
        self.assertEqual(summary["pool"]["cross_split_conflict_pairs"], 3)
        self.assertEqual(summary["pool"]["same_split_conflict_pairs"], 7)
        for scenario in summary["scenarios"].values():
            self.assertEqual(scenario["all_original_relation_crossings_total"], 0)
            self.assertEqual(scenario["totals"]["images"], 1099)
            self.assertEqual(scenario["totals"]["source_groups"], 1099)
            self.assertLess(scenario["max_abs_image_deviation_pp"], 1)
        with (manifests / "split_guard_clusters.csv").open(newline="", encoding="utf-8") as source:
            guards = list(csv.DictReader(source))
        with (manifests / "d2_experiment_pool_candidate.csv").open(newline="", encoding="utf-8") as source:
            pool = list(csv.DictReader(source))
        self.assertEqual(len(guards), 1406)
        self.assertEqual(len(pool), 2812)
        self.assertTrue(all(row["include_candidate"] == "false" for row in pool if row["variant"] == "resize"))
        self.assertTrue(all(b"\r" not in (manifests / name).read_bytes() for name in names))

    def test_real_u01_probe_if_archive_available(self):
        archive = ROOT / "data" / "raw" / "multistage_apple_v4" / "dataset-20260508.zip"
        if not archive.is_file():
            self.skipTest("audited D2 archive absent")
        result = probe.probe(archive)
        self.assertEqual(result["retained_valid_targets"], 3)
        self.assertEqual(result["standard_no_ignore"]["fp"], 1)
        self.assertEqual(result["optional_ignore_zone_sensitivity"]["ignored_predictions"], 1)


if __name__ == "__main__":
    unittest.main()
