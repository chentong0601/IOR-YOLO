"""Run every non-formal Kaggle E01 readiness gate, then stop.

This entrypoint may verify/download official initialization weights and run a
disposable batch-feasibility smoke test. It never calls formal train/val/test.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT = Path("/kaggle/working/ior-yolo-output/preflight")


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset = load_module("e01_preflight_dataset", "13_build_e01_ultralytics_dataset.py")
environment = load_module("e01_preflight_environment", "14_check_training_environment.py")
resolver = load_module("e01_preflight_resolver", "16_resolve_e01_config.py")
identity = load_module("e01_preflight_identity", "19_verify_d2_unpacked.py")
runner = load_module("e01_preflight_runner", "15_e01_run.py")


class GateFailure(RuntimeError):
    def __init__(self, gate_name: str, cause: Exception):
        self.gate_name = gate_name
        self.cause = cause
        super().__init__(f"{gate_name}: {type(cause).__name__}: {cause}")


def gate(name: str, detail: str) -> None:
    print(f"{name}: {detail}", flush=True)


def checked(gate_name: str, action):
    try:
        return action()
    except GateFailure:
        raise
    except Exception as exc:
        raise GateFailure(gate_name, exc) from exc


def run(*, source: Path | None, platform: str, batch: int,
        derived: Path | None = None, output_dir: Path = DEFAULT_OUTPUT,
        expected_commit: str | None = None) -> dict:
    if platform != "kaggle":
        raise ValueError("this dedicated entrypoint supports --platform kaggle only")
    if batch != 8:
        raise ValueError("one-command preflight must test frozen batch=8; batch=4 requires a separate OOM record")

    state = checked("Git provenance", runner.git_state)
    if expected_commit and state["head"] != expected_commit:
        raise GateFailure("Git provenance", RuntimeError(
            f"Git HEAD {state['head']} != expected {expected_commit}"))
    if state["experiment_relevant_dirty"]:
        raise GateFailure("Git provenance", RuntimeError(
            f"E01-relevant Git files are uncommitted: {state['experiment_relevant_status']}"))
    gate("Git provenance", f"PASS ({state['head']})")

    source = checked("D2 source", lambda: runner.resolve_source(source, expected_platform=platform))
    derived = checked("Derived path", lambda: runner.resolve_derived(derived, expected_platform=platform))
    gate("D2 source", str(source))
    if not source.is_dir():
        raise GateFailure("D2 source", ValueError(
            "Kaggle one-command preflight requires the verified unpacked D2 directory"))

    raw = checked("Raw identity", lambda: identity.verify(
        source, dataset.DEFAULT_MANIFESTS / "files_sha256.csv", dataset.DEFAULT_IDENTITY))
    gate("Raw identity", f"{raw['status']} ({raw['image_files']} JPEG + {raw['json_files']} JSON)")

    if derived.exists():
        built = None
        converted = checked("Derived dataset", lambda: dataset.validate_only(
            source, dataset.DEFAULT_MANIFESTS, dataset.DEFAULT_PROTOCOL, derived))
        derived_action = "existing disposable dataset validated"
    else:
        built = checked("Derived dataset build", lambda: dataset.build(
            source, dataset.DEFAULT_MANIFESTS, dataset.DEFAULT_PROTOCOL, derived))
        converted = checked("Derived dataset validation", lambda: dataset.validate_only(
            source, dataset.DEFAULT_MANIFESTS, dataset.DEFAULT_PROTOCOL, derived))
        derived_action = "disposable dataset rebuilt and validated"
    counts = converted["counts"]
    image_counts = {part: int(counts[part][0]) for part in dataset.SPLITS}
    if image_counts != {"train": 769, "val": 165, "test": 165}:
        raise GateFailure("Frozen split", RuntimeError(
            f"frozen split count mismatch: {image_counts}"))
    class_counts = {
        part: {"immature": int(counts[part][1]), "semi_mature": int(counts[part][2]),
               "mature": int(counts[part][3])}
        for part in dataset.SPLITS
    }
    gate("Derived dataset", f"PASS ({derived_action})")
    gate("Split", "769/165/165")
    gate("Class instances", json.dumps(class_counts, ensure_ascii=False, sort_keys=True))

    hardware = checked("Environment", lambda: environment.inspect(
        require_cuda=True, expected_platform=platform))
    gate("Environment", f"PASS ({hardware['python']}; torch {hardware['torch']})")
    gate("Ultralytics", hardware["ultralytics"])
    gate("CUDA", f"PASS ({hardware['cuda_runtime']}; {hardware['gpu_name']}; device cuda:0)")
    gate("Formal device", "cuda:0")

    checked("Scientific config", runner.load_config)
    resolved = checked("Resolved config", lambda: resolver.resolve(runner.CONFIG))
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_e01_config.yaml"
    resolver.save(resolved_path, resolved)
    gate("Resolved config", f"PASS ({resolved_path})")

    _, weights = checked("Weights", runner.pretrained_weight_record)
    gate("Weights", f"VERIFIED ({weights['filename']}; {weights['sha256']})")

    smoke = checked("Batch-8 CUDA smoke", lambda: runner.smoke(
        batch=batch, expected_platform=platform, source=source, derived=derived))
    gate("Batch-8 CUDA smoke", "PASS")

    summary = {
        "schema_version": "e01-kaggle-preflight-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": state,
        "expected_commit": expected_commit,
        "source": str(source),
        "raw_identity": raw,
        "derived_dataset": {"path": str(derived), "action": derived_action,
                            "validation": converted, "build": built},
        "split_image_counts": image_counts,
        "class_instance_counts": class_counts,
        "environment": hardware,
        "resolved_config_path": str(resolved_path),
        "weights": weights,
        "smoke": smoke,
        "formal_training": "NOT STARTED",
        "e01_readiness": "READY FOR HUMAN CONFIRMATION",
    }
    record = output_dir / "e01_kaggle_preflight.json"
    record.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8", newline="\n")
    gate("Formal training", "NOT STARTED")
    gate("E01 readiness", "READY FOR HUMAN CONFIRMATION")
    gate("Preflight record", str(record))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="verified unpacked D2 directory; takes precedence over environment/defaults")
    parser.add_argument("--platform", choices=("kaggle",), default="kaggle")
    parser.add_argument("--batch", type=int, choices=(8,), default=8)
    parser.add_argument("--derived", type=Path,
                        help="disposable derived dataset; defaults under /kaggle/working")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-commit", help="optional audited Git commit to require")
    args = parser.parse_args()
    try:
        run(source=args.source, platform=args.platform, batch=args.batch,
            derived=args.derived, output_dir=args.output_dir,
            expected_commit=args.expected_commit)
    except Exception as exc:
        failed_gate = exc.gate_name if isinstance(exc, GateFailure) else "Argument/policy validation"
        gate("Failed gate", f"{failed_gate}: {exc}")
        gate("Formal training", "NOT STARTED")
        gate("E01 readiness", "NOT READY")
        print(json.dumps({"status": "FAILED", "failed_gate": failed_gate,
                          "error_type": type(exc).__name__,
                          "error": str(exc), "formal_training": "NOT STARTED"},
                         ensure_ascii=False, indent=2))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
