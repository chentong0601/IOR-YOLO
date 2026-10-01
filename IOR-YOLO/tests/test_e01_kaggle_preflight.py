"""Low-cost orchestration tests; no CUDA work, model download or training."""

import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "e01_kaggle_preflight_test", ROOT / "scripts/20_kaggle_e01_preflight.py")
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class KagglePreflightTest(unittest.TestCase):
    def test_orchestration_stops_before_formal_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, derived, output = root / "raw", root / "derived", root / "output"
            source.mkdir()
            counts = {"train": (769, 646, 297, 512),
                      "val": (165, 143, 65, 115),
                      "test": (165, 126, 63, 105)}
            state = {"head": "abc123", "dirty": False, "status": [],
                     "experiment_relevant_dirty": False, "experiment_relevant_status": []}
            raw = {"status": "VERIFIED", "image_files": 2812, "json_files": 6}
            converted = {"status": "validated", "counts": counts,
                         "raw_identity_status": "VERIFIED"}
            hardware = {"python": "3.12.13", "torch": "2.10.0+cu128",
                        "ultralytics": "8.3.220", "cuda_runtime": "12.8",
                        "gpu_name": "Tesla T4"}
            weights = {"filename": "yolo11n-seg.pt", "sha256": "weight-sha"}
            with patch.object(preflight.runner, "git_state", return_value=state), \
                 patch.object(preflight.identity, "verify", return_value=raw), \
                 patch.object(preflight.dataset, "build", return_value={"statistics": {}}), \
                 patch.object(preflight.dataset, "validate_only", return_value=converted), \
                 patch.object(preflight.environment, "inspect", return_value=hardware), \
                 patch.object(preflight.runner, "load_config", return_value={}), \
                 patch.object(preflight.resolver, "resolve", return_value={"resolved": True}), \
                 patch.object(preflight.runner, "pretrained_weight_record",
                              return_value=(object(), weights)), \
                 patch.object(preflight.runner, "smoke",
                              return_value={"status": "CUDA smoke passed"}) as smoke:
                result = preflight.run(source=source, platform="kaggle", batch=8,
                                       derived=derived, output_dir=output,
                                       expected_commit="abc123")
            self.assertEqual(result["formal_training"], "NOT STARTED")
            self.assertEqual(result["e01_readiness"], "READY FOR HUMAN CONFIRMATION")
            self.assertEqual(result["split_image_counts"], {"train": 769, "val": 165, "test": 165})
            smoke.assert_called_once()
            self.assertNotIn("runner.train(", inspect.getsource(preflight))
            self.assertTrue((output / "e01_kaggle_preflight.json").is_file())

    def test_failed_identity_stops_before_build_and_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "raw"
            source.mkdir()
            state = {"head": "abc123", "dirty": False, "status": [],
                     "experiment_relevant_dirty": False, "experiment_relevant_status": []}
            with patch.object(preflight.runner, "git_state", return_value=state), \
                 patch.object(preflight.identity, "verify", side_effect=ValueError("hash mismatch")), \
                 patch.object(preflight.dataset, "build") as build, \
                 patch.object(preflight.runner, "smoke") as smoke:
                with self.assertRaisesRegex(preflight.GateFailure, "Raw identity.*hash mismatch"):
                    preflight.run(source=source, platform="kaggle", batch=8,
                                  derived=Path(temp) / "derived",
                                  output_dir=Path(temp) / "output")
            build.assert_not_called()
            smoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
