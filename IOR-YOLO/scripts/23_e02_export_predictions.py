"""E02 prediction export: the frozen E01 exporter behind the E02 provenance gates.

The export protocol itself is E01's and is never re-implemented: script 18 keeps
``conf = 0.001`` / ``iou = 0.7`` so the complete AP-consistent prediction stream
is preserved for the formal PR/AP evidence. That stream is read-only evidence for
script 24, which applies the frozen operational operating point ``conf = 0.65`` on
top of it.

Only two things are added here, both E02-specific:

* the hard E02 gates run *before* the model or any run file is touched (E01
  immutability, E02 run-directory isolation, the recorded config/resolved-config/
  best-checkpoint digests, the predeclared ordinal manipulation, the
  ``validated_not_final_tested`` status and the re-verified training artifacts);
* the split is pinned to the frozen validation split. Script 18 still exposes
  ``--split test``/``--final-test``, which is correct for the locked E01 final
  test, but the E02 wrapper neither passes nor accepts them.

An existing export is never overwritten: a re-export would silently change the
input of a completed analysis.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    """Import a sibling script by path; each script keeps its own module namespace."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load project script: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyzer = load_script("e02_ordinal_analysis", "24_e02_analyze_results.py")
exporter = load_script("e02_frozen_e01_export", "18_export_e01_predictions.py")

analysis = analyzer.analysis
runner = analyzer.runner
SPLIT = analyzer.SPLIT
FINAL_TEST = False


def export(run_dir: Path = runner.RUN_DIR, data_root: Path = analysis.DEFAULT_DATA, *, split: str = SPLIT,
           config: dict | None = None, repo_root: Path | None = None) -> dict:
    """Run the frozen E01 export for the E02 run, gated on E02 provenance.

    The order is deliberate: the split lock and every E02 gate run before the
    exporter is called, so an untrustworthy run can never produce a prediction
    stream at all, and the frozen E01 run directory is only ever read.
    """
    analyzer.assert_validation_split(split)
    run_dir, data_root = Path(run_dir), Path(data_root)
    repo_root = Path(repo_root) if repo_root is not None else runner.ROOT
    config = runner.load_config() if config is None else config
    guards = analyzer.guard(run_dir, config=config, repo_root=repo_root)
    manifest_path = run_dir / "run_manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"E02 run manifest absent: {manifest_path}")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    evidence = analyzer.required_manifest_evidence(manifest, config, run_dir)
    predictions = run_dir / "predictions" / f"predictions_{split}.csv"
    masks = run_dir / "predictions" / f"masks_{split}"
    if predictions.exists() or masks.exists():
        raise FileExistsError("E02 prediction export already exists; the input of a completed analysis "
                              "is never overwritten by a re-export")
    result = exporter.export(run_dir, data_root, split=split, final_test=FINAL_TEST)
    return {"experiment_id": manifest["experiment_id"], "run_id": manifest["run_id"], "split": split,
            "final_test": FINAL_TEST, "images": result["images"], "predictions": result["predictions"],
            "csv": result["csv"], "prediction_csv_sha256": analysis.digest(predictions),
            "mask_files": len(list(masks.glob("*.json"))) if masks.is_dir() else 0,
            "prediction_export_conf": analyzer.PREDICTION_EXPORT_CONF,
            "operating_confidence_threshold": analyzer.OPERATING_CONFIDENCE_THRESHOLD,
            "e01_immutability": guards["e01_immutability"]["status"],
            "run_isolation": guards["run_isolation"], "e02_gates": evidence,
            "export_protocol": "frozen E01 exporter (scripts/18) reused unchanged; conf=0.001, iou=0.7",
            "note": "analysed by scripts/24 at the frozen operating point; the export itself is never "
                    "threshold-filtered and the Final Test stays locked"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=runner.RUN_DIR)
    parser.add_argument("--data-root", type=Path, default=analysis.DEFAULT_DATA,
                        help="derived dataset root holding images/val and labels/val")
    args = parser.parse_args()
    print(json.dumps(export(args.run_dir, args.data_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
