"""Focused checks for D2 source grouping and simulation invariants."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


grouping = load_script("06_build_source_groups")
simulation = load_script("07_simulate_group_split")


class SourceGroupTest(unittest.TestCase):
    def test_deterministic_component_order_and_exact_duplicate_grouping(self):
        nodes = [("train", "a.jpg"), ("val", "b.jpg"), ("test", "c.jpg")]
        outputs = []
        for order in (nodes, list(reversed(nodes))):
            graph = grouping.Components(order)
            exact_row = {"relation_type": "exact_duplicate"}
            kind, confidence = grouping.classify_existing(exact_row, {}, {}, None, "")
            self.assertEqual((kind, confidence), ("exact_duplicate", "Confirmed"))
            graph.union(nodes[0], nodes[1])
            outputs.append(graph.groups())
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0]), 2)
        self.assertEqual(set(outputs[0][1]), set(nodes[:2]))

    def test_candidate_is_not_merged(self):
        row = {"relation_type": "same_annotation_geometry", "status": "Candidate"}
        kind, confidence = grouping.classify_existing(row, {}, {}, None, "")
        self.assertEqual((kind, confidence), ("same_annotation_geometry", "Candidate"))
        graph = grouping.Components([(0, "a"), (1, "a")])
        if confidence in ("Confirmed", "Strongly Supported", "Supported"):
            graph.union((0, "a"), (1, "a"))
        self.assertEqual(len(graph.groups()), 2)

    def test_original_only_counts_and_class_preservation(self):
        rows = [
            {"official_split": "train", "filename": "a.jpg", "variant": "original",
             "source_group_id": "sg-1", "instance_count": "2"},
            {"official_split": "train", "filename": "a.jpg", "variant": "resize",
             "source_group_id": "sg-1", "instance_count": "2"},
            {"official_split": "val", "filename": "b.jpg", "variant": "original",
             "source_group_id": "sg-2", "instance_count": "1"},
        ]
        annotation = {
            ("train", "a.jpg"): [{"region_attributes": {"name": "immature apple"}},
                                   {"region_attributes": {"name": "mature apple"}}],
            ("val", "b.jpg"): [{"region_attributes": {"name": "semi-mature apple"}}],
        }
        groups = simulation.group_data(rows, annotation)
        total = simulation.totals(groups)
        self.assertEqual(total["images"], 2)
        self.assertEqual(total["multi_class_images"], 1)
        self.assertEqual([total[c] for c in simulation.CLASSES], [1, 1, 1])

    def test_seed_reproducibility_and_group_integrity(self):
        groups = {f"g{i}": {"images": 1 + i % 3, "multi_class_images": i % 2,
                             "classes": Counter({simulation.CLASSES[i % 3]: 1 + i % 3}),
                             "members": [("train", f"{i}.jpg")]} for i in range(30)}
        target = (0.7, 0.15, 0.15)
        a, counts_a = simulation.simulate(groups, target, 7)
        b, counts_b = simulation.simulate(groups, target, 7)
        reversed_groups = dict(reversed(list(groups.items())))
        c, counts_c = simulation.simulate(reversed_groups, target, 7)
        self.assertEqual(a, b)
        self.assertEqual(counts_a, counts_b)
        self.assertEqual(a, c)
        self.assertEqual(counts_a, counts_c)
        self.assertEqual(len(a), len(groups))
        self.assertEqual(sum(v["images"] for v in counts_a.values()),
                         simulation.totals(groups)["images"])
        node_to_group = {member: gid for gid, group in groups.items() for member in group["members"]}
        edges = [{"variant_a": "original", "variant_b": "original",
                  "split_a": "train", "file_a": "0.jpg", "split_b": "train",
                  "file_b": "0.jpg", "confidence": "Confirmed"}]
        self.assertEqual(simulation.edge_cuts(edges, node_to_group, a), {})

    def test_csv_uses_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            grouping.write_csv(path, ("a",), [{"a": "x"}])
            self.assertEqual(path.read_bytes(), b"a\nx\n")

    def test_real_manifest_resize_exceptions_if_available(self):
        path = ROOT / "data" / "manifests" / "source_groups.csv"
        if not path.is_file():
            self.skipTest("generated D2 manifest absent")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({row["source_group_id"] for row in rows if row["variant"] == "original"}), 1112)
        self.assertEqual(Counter(row["variant"] for row in rows),
                         {"original": 1406, "resize": 1406})
        by_node = {(row["official_split"], row["filename"], row["variant"]): row for row in rows}
        for left, right in (
                (("train", "IMG_13960_brightness.jpg"), ("val", "IMG_13960.jpg")),
                (("train", "IMG_14040_brightness.jpg"), ("val", "IMG_14040.jpg")),
                (("test", "3100.jpg"), ("train", "1640.jpg")),
                (("test", "1270.jpg"), ("train", "2340.jpg"))):
            self.assertEqual(by_node[(*left, "original")]["source_group_id"],
                             by_node[(*right, "original")]["source_group_id"])
        for (split, name, variant), resized in by_node.items():
            if variant == "resize":
                self.assertEqual(resized["source_group_id"],
                                 by_node[(split, name, "original")]["source_group_id"])
        for left, right in (
                (("test", "138.jpg"), ("train", "251.jpg")),
                (("train", "138_brightness.jpg"), ("val", "251_brightness.jpg")),
                (("train", "398.jpg"), ("val", "166.jpg"))):
            self.assertEqual(by_node[(*left, "original")]["source_group_id"],
                             by_node[(*right, "original")]["source_group_id"])
        for split, name in (("train", "172_brightness.jpg"),
                            ("val", "172_noise.jpg"), ("train", "327.jpg")):
            original = by_node[(split, name, "original")]
            resized = by_node[(split, name, "resize")]
            self.assertEqual(original["source_group_id"], resized["source_group_id"])
            self.assertEqual(resized["confidence"], "Unresolved")
            self.assertEqual(resized["relation_type"], "same_source_companion")
        unknown = by_node[("test", "IMG_54350.jpg", "original")]
        self.assertEqual((unknown["instance_count"], unknown["unlabeled_count"]), ("3", "1"))

    def test_human_adjudicated_relation_tiers_if_available(self):
        path = ROOT / "data" / "manifests" / "source_group_relations.csv"
        if not path.is_file():
            self.skipTest("generated D2 relations absent")
        with path.open(newline="", encoding="utf-8") as stream:
            edges = list(csv.DictReader(stream))
        expected = grouping.load_human_adjudications(grouping.ADJUDICATIONS)
        for pair, decision in expected.items():
            matching = [edge for edge in edges if edge["variant_a"] == edge["variant_b"] == "original"
                        and frozenset((f"{edge['split_a']}/{edge['file_a']}",
                                       f"{edge['split_b']}/{edge['file_b']}")) == pair]
            self.assertEqual(len(matching), 1, decision["case_id"])
            self.assertEqual(matching[0]["confidence"], "Strongly Supported")
            self.assertEqual(matching[0]["merged"], "true")
            self.assertIn("physical acquisition ID unverified", matching[0]["evidence"])
        self.assertEqual(Counter(edge["confidence"] for edge in edges),
                         {"Supported": 1426, "Strongly Supported": 393,
                          "Candidate": 33, "Confirmed": 10, "Unresolved": 3})

    def test_full_source_group_rebuild_is_deterministic_if_archive_available(self):
        archive = ROOT / "data" / "raw" / "multistage_apple_v4" / "dataset-20260508.zip"
        manifest = ROOT / "data" / "manifests"
        if not archive.is_file():
            self.skipTest("local audited D2 ZIP absent")
        digests = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory)
                for name in ("files_sha256.csv", "duplicate_report.csv"):
                    shutil.copyfile(manifest / name, target / name)
                summary = grouping.build(archive, target)
                self.assertEqual(summary["source_groups"], 1112)
                self.assertEqual(summary["cross_official_split_groups"], 67)
                digests.append(tuple(hashlib.sha256((target / name).read_bytes()).hexdigest()
                                     for name in ("source_groups.csv", "source_group_relations.csv",
                                                  "source_family_review.csv")))
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(digests[0], tuple(hashlib.sha256((manifest / name).read_bytes()).hexdigest()
                                          for name in ("source_groups.csv", "source_group_relations.csv",
                                                       "source_family_review.csv")))


if __name__ == "__main__":
    unittest.main()
