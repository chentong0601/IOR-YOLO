"""E02 ordinal-aware YOLO11n-seg entrypoint: preflight, smoke, train and val.

E01 (``runs/e01_yolo11n_seg/seed_0``) is treated as read-only evidence: this
entrypoint verifies the recorded digests of every E01 experiment artifact before
it does anything, never writes inside the E01 run directory and never modifies
E01 code, configs, manifests or results. The only experimental difference from
E01 is the ordinal auxiliary loss (``lambda_ord = 0.5``), asserted the other way
round as a byte-identical ``train``/augmentation block.

The Final Test stays locked: this entrypoint exposes no test command, validates
the frozen validation split only, and refuses any other split.

Formal training must be started by the user (visible local/Kaggle execution); the
disposable ``smoke``/``tensorboard-smoke`` commands exist for engineering checks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
# Import ior_yolo.* as a real package: the E02 model class is pickled into the
# checkpoint by module path, so every process that loads an E02 checkpoint must
# see exactly this module name.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs/experiments/e02_yolo11n_seg_ordinal.yaml"
TEMPLATE = ROOT / "configs/experiments/e02_run_manifest_template.yaml"
E01_CONFIG = ROOT / "configs/experiments/e01_yolo11n_seg.yaml"
KAGGLE_SOURCE = Path("/kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508")
KAGGLE_DERIVED = Path("/kaggle/working/ior-yolo-derived/d2_e01_ultralytics")
DATASET = ROOT / "data/processed/d2_e01_ultralytics"
EXPERIMENT_ID = "e02_yolo11n_seg_ordinal"
EXPECTED_SPLIT_SEED = 13436313853456744620
# This entrypoint deliberately exposes no test command: Final Test stays locked in E01.
COMMANDS = ("preflight", "smoke", "tensorboard-smoke", "train", "val")
FINAL_TEST_POLICY = "locked_inherited_from_e01"
EVALUATION_SPLIT = "val"
RUNS_ROOT = Path(os.environ.get("E02_RUNS_ROOT", ROOT / "runs")).expanduser().resolve()
RUN_DIR = RUNS_ROOT / f"{EXPERIMENT_ID}/seed_0"
PREFLIGHT_DIR = RUNS_ROOT / f"{EXPERIMENT_ID}/preflight"
E01_RUNS_ROOT = Path(os.environ.get("E01_RUNS_ROOT", ROOT / "runs")).expanduser().resolve()
E01_RUN_DIR = E01_RUNS_ROOT / "e01_yolo11n_seg/seed_0"
# E02-owned tracked artifacts; E01 paths are added from the frozen E01 runner.
E02_GIT_PATHS = (
    "IOR-YOLO/configs/experiments/e02_yolo11n_seg_ordinal.yaml",
    "IOR-YOLO/configs/experiments/e02_run_manifest_template.yaml",
    "IOR-YOLO/ior_yolo/losses/ordinal.py",
    "IOR-YOLO/ior_yolo/trainers/ordinal.py",
    "IOR-YOLO/ior_yolo/utils/tensorboard_logger.py",
    "IOR-YOLO/requirements-e02.txt",
    "IOR-YOLO/scripts/22_e02_ordinal_run.py",
    "IOR-YOLO/scripts/23_e02_export_predictions.py",
    "IOR-YOLO/scripts/24_e02_analyze_results.py",
    "IOR-YOLO/tests/test_e02_analysis.py",
    "IOR-YOLO/tests/test_e02_ordinal_loss.py",
    "IOR-YOLO/tests/test_e02_ordinal_training.py",
    "IOR-YOLO/tests/test_e02_preflight.py",
    "IOR-YOLO/tests/test_e02_tensorboard.py",
)


def load_script(name: str, filename: str):
    """Load a frozen E01 script read-only (never imported as a package)."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load project script: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset = load_script("e02_dataset", "13_build_e01_ultralytics_dataset.py")
environment = load_script("e02_environment", "14_check_training_environment.py")
e01_resolver = load_script("e02_e01_resolver", "16_resolve_e01_config.py")
e01_runner = load_script("e02_e01_formal_runner", "15_e01_run.py")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    """Git provenance for the whole repo plus the E01 and E02 experiment paths."""
    def porcelain(*paths: str) -> list[str]:
        command = ["git", "status", "--porcelain=v1", "--untracked-files=all"]
        if paths:
            command += ["--", *paths]
        return subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                              check=True).stdout.splitlines()

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()
    status = porcelain()
    e01_status = porcelain(*e01_runner.E01_GIT_PATHS)
    e02_status = porcelain(*E02_GIT_PATHS)
    return {"head": head, "dirty": bool(status), "status": status,
            "e01_relevant_dirty": bool(e01_status), "e01_relevant_status": e01_status,
            "e02_relevant_dirty": bool(e02_status), "e02_relevant_status": e02_status,
            "experiment_relevant_dirty": bool(e01_status or e02_status),
            "experiment_relevant_status": [*e01_status, *e02_status]}


def assert_train_parity(e02_train: dict, e01_train: dict) -> None:
    """Fail unless the E02 training/augmentation block equals the frozen E01 block."""
    if e02_train == e01_train:
        return
    keys = sorted(set(e02_train) | set(e01_train))
    differing = [key for key in keys if e02_train.get(key) != e01_train.get(key)]
    raise ValueError("E02 must keep the E01 training/augmentation block byte-identical; differing keys: "
                     f"{differing}")


def ordinal_config_from_config(config: dict) -> dict:
    """Build and validate the trainer's ``ordinal_config`` block from the E02 config."""
    from ior_yolo.losses.ordinal import CUMULATIVE_TARGETS, E02_LAMBDA_ORD, STAGE_ORDER, OrdinalLossConfig

    ordinal = config["ordinal"]
    if list(ordinal["encoding"]) != [0, 1, 2]:
        raise ValueError(f"ordinal encoding must stay ascending [0, 1, 2], got {ordinal['encoding']}")
    if tuple(ordinal["class_order"]) != STAGE_ORDER:
        raise ValueError(f"ordinal class_order must stay {STAGE_ORDER}, got {ordinal['class_order']}")
    if ordinal["semantics"] != "ascending_maturity":
        raise ValueError("ordinal semantics must stay 'ascending_maturity'")
    if ordinal["loss"] != "cumulative_distribution_squared_error":
        raise ValueError(f"unexpected ordinal loss: {ordinal['loss']}")
    if ordinal["lambda_selection"] != "predeclared_before_training":
        raise ValueError("lambda_ord must stay predeclared, never selected on validation or test")
    declared = {name: [float(x) for x in values] for name, values in ordinal["cumulative_targets"].items()}
    expected = {"immature": list(CUMULATIVE_TARGETS["immature apple"]),
                "semi-mature": list(CUMULATIVE_TARGETS["semi-mature apple"]),
                "mature": list(CUMULATIVE_TARGETS["mature apple"])}
    if declared != expected:
        raise ValueError(f"declared cumulative targets {declared} differ from the implemented {expected}")
    mapping = {"lambda_ord": float(ordinal["lambda_ord"]), "num_stages": int(ordinal["num_stages"]),
               "stage_order": list(ordinal["class_order"]),
               "log_batch_scalars": bool(config["tensorboard"]["log_batch_scalars"])}
    if mapping["lambda_ord"] != E02_LAMBDA_ORD:
        raise ValueError(f"lambda_ord must stay {E02_LAMBDA_ORD} for round 1, got {mapping['lambda_ord']}")
    OrdinalLossConfig(num_stages=mapping["num_stages"], lambda_ord=mapping["lambda_ord"],
                      stage_order=tuple(mapping["stage_order"]))
    return mapping


def load_config(config_path: Path = CONFIG, *, e01_config_path: Path = E01_CONFIG) -> dict:
    """Load the E02 config and machine-check every frozen E01 invariant it inherits."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ValueError(f"unexpected experiment_id: {config['experiment_id']}")
    if (config["task"], config["model"], config["pretrained"], config["training_seed"]) != (
            "segment", "yolo11n-seg.pt", True, 0):
        raise ValueError("E02 identity changed")
    if config["dataset_protocol_version"] != "d2-v4-stage2b5-v1" or config["split_seed_64"] != EXPECTED_SPLIT_SEED:
        raise ValueError("E02 config differs from the frozen D2 protocol")
    if config["train"]["seed"] != config["training_seed"]:
        raise ValueError("training seed disagreement")
    e01 = yaml.safe_load(Path(e01_config_path).read_text(encoding="utf-8"))
    assert_train_parity(config["train"], e01["train"])
    if config["execution_environment"] != e01["execution_environment"]:
        raise ValueError("E02 execution environment must stay identical to E01")
    execution = config["execution_environment"]
    if (execution["formal_platform"], execution["formal_device"], execution["single_gpu_only"]) != (
            "Kaggle", "cuda:0", True):
        raise ValueError("E02 formal execution policy changed")
    if config["run"]["run_directory"] != f"{EXPERIMENT_ID}/seed_0":
        raise ValueError(f"unexpected E02 run directory: {config['run']['run_directory']}")
    if config["run"]["final_test"] != FINAL_TEST_POLICY:
        raise ValueError("E02 must keep the Final Test locked and inherited from E01")
    ordinal_config_from_config(config)
    return config


def e01_immutability_report(config: dict, *, repo_root: Path = ROOT) -> dict:
    """Verify every recorded E01 digest: the frozen baseline must stay read-only."""
    expected = config["e01_reference"]["files_sha256"]
    verified, mismatched, missing = {}, [], []
    for relative, digest in expected.items():
        path = repo_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = sha(path)
        if actual != digest:
            mismatched.append({"path": relative, "expected": digest, "actual": actual})
        else:
            verified[relative] = actual
    ok = not missing and not mismatched
    return {"status": "verified" if ok else "E01_MUTATED", "ok": ok, "files_checked": len(expected),
            "verified": verified, "missing": missing, "mismatched": mismatched}


def assert_e01_immutable(config: dict, *, repo_root: Path = ROOT) -> dict:
    """Raise when any frozen E01 artifact changed; otherwise return the verification report."""
    report = e01_immutability_report(config, repo_root=repo_root)
    if not report["ok"]:
        raise RuntimeError("frozen E01 artifacts changed; E02 refuses to run against a mutated baseline: "
                           f"missing={report['missing']} "
                           f"mismatched={[item['path'] for item in report['mismatched']]}")
    return report


def verify_derived_classes(derived_root: Path) -> dict:
    """Assert the derived dataset still encodes the frozen ascending stage order."""
    data_yaml = Path(derived_root) / "dataset.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"derived dataset yaml missing: {data_yaml}")
    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("names")
    expected = {index: name for index, name in enumerate(dataset.CLASSES)}
    if names != expected:
        raise ValueError(f"derived dataset class order changed: {names} != {expected}")
    return {"dataset_yaml": str(data_yaml.resolve()), "names": names, "nc": len(expected),
            "class_order": list(dataset.CLASSES), "ascending_encoding": [0, 1, 2]}


def assert_run_isolation(run_dir: Path = RUN_DIR, e01_run_dir: Path = E01_RUN_DIR) -> dict:
    """The E02 run directory must never overlap the frozen E01 run directory."""
    run_dir, e01_run_dir = Path(run_dir).resolve(), Path(e01_run_dir).resolve()
    if run_dir == e01_run_dir or run_dir.is_relative_to(e01_run_dir) or e01_run_dir.is_relative_to(run_dir):
        raise RuntimeError(f"E02 run directory must not overlap the frozen E01 run directory: {run_dir}")
    if (run_dir.name, run_dir.parent.name) != ("seed_0", EXPERIMENT_ID):
        raise RuntimeError(f"unexpected E02 run directory: {run_dir}")
    if any(part.startswith("e01") for part in run_dir.parts):
        raise RuntimeError(f"E02 run directory must not live inside an E01 experiment path: {run_dir}")
    return {"run_dir": str(run_dir), "e01_run_dir": str(e01_run_dir), "isolated": True,
            "e01_run_dir_exists": e01_run_dir.exists()}


def resolve_config() -> dict:
    """Resolve E02 with the frozen E01 resolver, then relabel and extend the record.

    The E01 resolver is reused read-only so the projection of ``optimizer=auto``
    and the effective Ultralytics defaults stay literally the same computation as
    the baseline; only labels that would be wrong for E02 are renamed.
    """
    config = load_config()
    resolved = e01_resolver.resolve(CONFIG)
    relabelled = {("experiment_config_sha256" if key == "e01_requested_config_sha256" else key): value
                  for key, value in resolved.items()}
    relabelled["schema_version"] = "e02-resolved-v1"
    relabelled["ordinal"] = config["ordinal"]
    relabelled["ordinal_config"] = ordinal_config_from_config(config)
    relabelled["ordinal_diagnostics"] = None
    relabelled["tensorboard"] = config["tensorboard"]
    relabelled["e01_reference"] = {"training_commit": config["e01_reference"]["training_commit"],
                                   "analysis_commit": config["e01_reference"]["analysis_commit"],
                                   "files_sha256": config["e01_reference"]["files_sha256"],
                                   "frozen_results": config["e01_reference"]["frozen_results"]}
    relabelled["final_test"] = config["run"]["final_test"]
    return relabelled


def tensorboard_gate(log_dir: Path = RUN_DIR / "tensorboard") -> dict:
    """Hard gate: the pinned TensorBoard runtime must write and read back every required tag."""
    from ior_yolo.utils.tensorboard_logger import (TENSORBOARD_DIRNAME, required_tag_list,
                                                  tensorboard_availability, tensorboard_roundtrip)

    expected_log_dir = (RUN_DIR / TENSORBOARD_DIRNAME).resolve()
    if Path(log_dir).resolve() != expected_log_dir:
        raise RuntimeError(f"TensorBoard log directory must stay inside the E02 run directory "
                           f"({expected_log_dir}), got {log_dir}")
    availability = tensorboard_availability()
    if not availability["available"]:
        raise RuntimeError("TensorBoard is a hard requirement for E02 and is not importable; install the pinned "
                           "runtime with 'python -m pip install -r IOR-YOLO/requirements-e02.txt'. "
                           f"Cause: {availability['error']}")
    with tempfile.TemporaryDirectory(prefix="e02-tensorboard-gate-") as temporary:
        roundtrip = tensorboard_roundtrip(Path(temporary) / "roundtrip")
    if not roundtrip["ok"]:
        raise RuntimeError(f"TensorBoard round-trip check failed: {roundtrip}")
    return {"status": "verified", "availability": availability, "log_dir": str(expected_log_dir),
            "required_tags": required_tag_list(), "roundtrip": roundtrip}


def pretrained_weight_record():
    """Load the official pretrained initialization through the E02 class map, never randomly."""
    from ior_yolo.trainers.ordinal import OrdinalYOLO

    model = OrdinalYOLO(e01_runner.formal_model_name())
    path = Path(getattr(model, "ckpt_path", "") or "")
    if not path.is_file() or path.name != "yolo11n-seg.pt":
        raise RuntimeError("official YOLO11n-seg pretrained checkpoint is unavailable; stop without random "
                           "initialization")
    return model, {"filename": path.name,
                   "source": "Ultralytics official yolo11n-seg.pt loaded by ultralytics==8.3.220 "
                             "(E02 ordinal class map)",
                   "sha256": sha(path),
                   "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                   "local_path": str(path.resolve())}


SMOKE_FRACTION = 0.02
SMOKE_EPOCHS = 1


def smoke_train_args(config: dict, dataset_yaml: str, *, batch: int, device: str) -> dict:
    """Disposable one-epoch/tiny-fraction arguments; never a formal result."""
    args = {key: value for key, value in config["train"].items() if key != "augmentation"}
    args.update(config["train"]["augmentation"])
    args.update(data=dataset_yaml, device=device, batch=batch, epochs=SMOKE_EPOCHS, fraction=SMOKE_FRACTION,
                workers=0, val=False, save=False, plots=False)
    return args


def verify_training_artifacts(run_dir: Path, *, require_tensorboard: bool = True) -> dict:
    """Assert the E02 loss-item contract, the dashboard and a finite ordinal loss after training."""
    from ior_yolo.utils.tensorboard_logger import (TENSORBOARD_DIRNAME, event_files, read_written_tags,
                                                  required_tag_list)

    run_dir = Path(run_dir)
    csv_path = run_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"training produced no results.csv: {csv_path}")
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"results.csv has no epoch row: {csv_path}")
    columns = list(rows[0])
    required_columns = ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss", "train/ordinal_loss",
                        "val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss", "val/ordinal_loss",
                        "metrics/precision(B)", "metrics/mAP50(B)", "metrics/precision(M)", "metrics/mAP50-95(M)",
                        "lr/pg0")
    missing_columns = [key for key in required_columns if key not in columns]
    if missing_columns:
        raise RuntimeError(f"results.csv is missing required E02 columns: {missing_columns}")
    ordinal_values = [float(row["train/ordinal_loss"]) for row in rows]
    if not all(math.isfinite(value) and value >= 0.0 for value in ordinal_values):
        raise RuntimeError(f"non-finite or negative ordinal loss in results.csv: {ordinal_values}")
    report = {"results_csv": str(csv_path), "epochs_recorded": len(rows), "columns": columns,
              "ordinal_loss_first_epoch": ordinal_values[0], "ordinal_loss_last_epoch": ordinal_values[-1],
              "ordinal_loss_min": min(ordinal_values), "ordinal_loss_max": max(ordinal_values)}
    log_dir = run_dir / TENSORBOARD_DIRNAME
    files = event_files(log_dir)
    observed, backend = read_written_tags(log_dir)
    missing_tags = [tag for tag in required_tag_list() if tag not in observed]
    report["tensorboard"] = {"log_dir": str(log_dir), "event_files": files, "read_back_backend": backend,
                             "tags_observed": len(observed), "missing_tags": missing_tags}
    if require_tensorboard:
        if not files:
            raise RuntimeError(f"TensorBoard event files were not written to {log_dir}; E02 forbids training "
                               "without a completed dashboard")
        if missing_tags:
            raise RuntimeError(f"TensorBoard is missing required E02 tags: {missing_tags}")
    return report


def smoke(*, device: str = "cuda:0", batch: int = 8, platform: str = "kaggle", source: Path | None = None,
          derived: Path | None = None, output_dir: Path | None = None, oom_note: str | None = None) -> dict:
    """Disposable one-epoch/tiny-fraction E02 run; never a formal result.

    ``cuda:0`` reuses the full formal preflight (raw D2 identity + CUDA gates).
    ``cpu``/``mps`` are engineering-only: the raw ZIP is not re-verified, but the
    existing derived dataset must still match the frozen split manifest, and the
    result is labelled as engineering evidence.
    """
    if device not in ("cuda:0", "cpu", "mps"):
        raise ValueError(f"smoke device must be cuda:0, cpu or mps, got {device}")
    if batch not in (8, 4) or (batch == 4) != bool(oom_note):
        raise ValueError("smoke uses batch 8; batch 4 requires the prior batch-8 CUDA OOM note")
    config = load_config()
    immutable = assert_e01_immutable(config)
    state = git_state()
    if state["e01_relevant_dirty"]:
        raise RuntimeError("E01-relevant Git files are uncommitted; the frozen baseline must stay read-only: "
                           f"{state['e01_relevant_status']}")
    isolation = assert_run_isolation()
    if device == "cuda:0":
        ready = preflight(require_cuda=True, platform=platform, source=source, derived=derived)
        derived = Path(ready["derived_dataset"])
        dataset_yaml = ready["dataset_yaml"]
        source_verification = {"status": "full formal chain: raw D2 identity + CUDA gate verified",
                               "raw_source": ready["raw_source"]}
    else:
        local = load_script("e02_local_engineering", "21_e01_local_engineering.py")
        derived = Path(derived) if derived else DATASET
        classes = verify_derived_classes(derived)
        dataset_yaml = classes["dataset_yaml"]
        source_verification = {"status": "engineering device: raw D2 ZIP not re-verified",
                               "derived_dataset": local.validate_derived_train_val(
                                   Path(derived), dataset.DEFAULT_MANIFESTS),
                               "dataset_classes": classes}
    model, weights = pretrained_weight_record()
    args = smoke_train_args(config, dataset_yaml, batch=batch, device=device)
    temporary = tempfile.TemporaryDirectory(prefix="e02-ordinal-smoke-") if output_dir is None else None
    parent = Path(temporary.name) if temporary is not None else Path(output_dir).expanduser().resolve()
    if temporary is None:
        parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = parent / "smoke_seed_0"
        if run_dir.exists():
            raise FileExistsError(f"smoke output already exists; pass a new --output-dir: {run_dir}")
        started = datetime.now(timezone.utc).isoformat()
        model.train(**args, project=str(parent), name=run_dir.name, exist_ok=False,
                    ordinal_config=ordinal_config_from_config(config))
        artifacts = verify_training_artifacts(run_dir)
        diagnostics = model.trainer.ordinal_diagnostics()
        result = {"status": "E02 engineering smoke passed; not a formal experiment",
                  "experiment_id": EXPERIMENT_ID, "device": device, "batch": batch, "oom_note": oom_note,
                  "smoke_budget": {"epochs": SMOKE_EPOCHS, "fraction": SMOKE_FRACTION},
                  "git": state, "e01_immutability": immutable, "run_isolation": isolation,
                  "source_verification": source_verification, "dataset_yaml": dataset_yaml,
                  "ordinal_config": ordinal_config_from_config(config), "pretrained_weight": weights,
                  "run_directory": str(run_dir), "artifacts_retained": temporary is None,
                  "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "ordinal_diagnostics": diagnostics, "artifacts": artifacts,
                  "formal_training": "NOT STARTED"}
        if temporary is None:
            (parent / "smoke_result.json").write_text(
                json.dumps(e01_runner.yaml_safe(result), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8", newline="\n")
        return result
    finally:
        if temporary is not None:
            temporary.cleanup()


def tensorboard_smoke(*, output_dir: Path | None = None) -> dict:
    """Disposable dashboard check: write the required tag contract and read it back."""
    from ior_yolo.utils.tensorboard_logger import (TENSORBOARD_DIRNAME, required_tag_list,
                                                  tensorboard_availability, tensorboard_roundtrip)

    config = load_config()
    availability = tensorboard_availability()
    if not availability["available"]:
        raise RuntimeError("TensorBoard is a hard requirement for E02 and is not importable; install the pinned "
                           "runtime with 'python -m pip install -r IOR-YOLO/requirements-e02.txt'. "
                           f"Cause: {availability['error']}")
    retained = None
    if output_dir is None:
        with tempfile.TemporaryDirectory(prefix="e02-tensorboard-smoke-") as temporary:
            report = tensorboard_roundtrip(Path(temporary) / TENSORBOARD_DIRNAME)
    else:
        target = Path(output_dir).expanduser().resolve() / TENSORBOARD_DIRNAME
        if target.exists():
            raise FileExistsError(f"TensorBoard smoke output already exists: {target}")
        report = tensorboard_roundtrip(target)
        retained = str(target)
    if not report["ok"]:
        raise RuntimeError(f"TensorBoard smoke failed: {report}")
    return {"status": "TensorBoard smoke passed; not a formal experiment", "availability": availability,
            "required_tags": required_tag_list(), "tensorboard_config": config["tensorboard"],
            "report": report, "artifacts_retained": retained,
            "canonical_logdir": str(RUN_DIR / "tensorboard"), "formal_training": "NOT STARTED"}


def write_manifest(value: dict) -> None:
    """Write the E02 run manifest with the same YAML-safety rules as E01."""
    path = RUN_DIR / "run_manifest.yaml"
    path.write_text(yaml.safe_dump(e01_runner.yaml_safe(value), sort_keys=False, allow_unicode=True),
                    encoding="utf-8", newline="\n")


def read_manifest() -> dict:
    path = RUN_DIR / "run_manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"E02 run manifest absent: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_train_args(config: dict, dataset_yaml: str, *, batch_override: int | None) -> dict:
    """Formal training arguments: the frozen E01 recipe plus the E02 ordinal_config block."""
    args = {key: value for key, value in config["train"].items() if key != "augmentation"}
    args.update(config["train"]["augmentation"])
    if batch_override == 4:
        args["batch"] = 4
    args.update(data=dataset_yaml, device=0, project=str(RUN_DIR.parent), name=RUN_DIR.name,
                exist_ok=False, pretrained=True)
    return args


def train(*, batch_override: int | None = None, oom_note: str | None = None, platform: str = "kaggle",
          source: Path | None = None, derived: Path | None = None) -> dict:
    """Formal E02 training (user-launched, visible): ordinal term on top of the frozen E01 recipe."""
    ready = preflight(require_cuda=True, platform=platform, source=source, derived=derived)
    config = load_config()
    if (batch_override == 4) != bool(oom_note):
        raise ValueError("batch 4 requires a pretraining CUDA OOM note; no note permitted without fallback")
    state = git_state()
    if state["experiment_relevant_dirty"]:
        raise RuntimeError("E01/E02-relevant Git files are uncommitted; refusing to start a formal E02 run: "
                           f"{state['experiment_relevant_status']}")
    isolation = assert_run_isolation()
    if RUN_DIR.exists():
        raise FileExistsError(f"formal E02 run already exists; no overwrite: {RUN_DIR}")
    model, weight_record = pretrained_weight_record()
    pretrain_path = RUN_DIR.parent / "seed_0_resolved_train_config.yaml"
    if pretrain_path.exists():
        raise FileExistsError(f"pretrain snapshot already exists: {pretrain_path}")
    resolved = resolve_config()
    if batch_override == 4:
        resolved["hardware_oom_note"] = oom_note
    e01_resolver.save(pretrain_path, resolved)
    started = datetime.now(timezone.utc).isoformat()
    args = resolve_train_args(config, ready["dataset_yaml"], batch_override=batch_override)
    model.train(**args, ordinal_config=ordinal_config_from_config(config))
    trainer = model.trainer
    actual_dir = Path(trainer.save_dir).resolve()
    if actual_dir != RUN_DIR.resolve():
        raise RuntimeError(f"run directory unexpectedly changed: {actual_dir}")
    artifacts = verify_training_artifacts(RUN_DIR)
    diagnostics = trainer.ordinal_diagnostics()
    optimizer = trainer.optimizer
    runtime = {
        "optimizer": type(optimizer).__name__,
        "initial_optimizer_lr": float(optimizer.defaults["lr"]),
        "optimizer_defaults": {str(k): str(v) for k, v in optimizer.defaults.items()},
        "warmup_bias_lr": float(trainer.args.warmup_bias_lr),
        "train_loader_images": len(trainer.train_loader.dataset),
        "trainer_batch_size": trainer.batch_size,
        "trainer_epochs_budget": trainer.epochs,
        "actual_epochs_from_results_csv": artifacts["epochs_recorded"],
        "loss_names": list(trainer.loss_names),
        "ultralytics_args_yaml": "args.yaml",
    }
    resolved["status"] = "runtime_observed"
    resolved["ordinal_diagnostics"] = diagnostics
    resolved["actual_runtime"] = runtime
    e01_resolver.save(RUN_DIR / "resolved_train_config.yaml", resolved)
    for folder in ("metrics", "predictions", "figures"):
        (RUN_DIR / folder).mkdir(exist_ok=True)
    if (RUN_DIR / "results.csv").is_file():
        shutil.copyfile(RUN_DIR / "results.csv", RUN_DIR / "metrics/results.csv")
    for figure in RUN_DIR.glob("*.png"):
        shutil.copyfile(figure, RUN_DIR / "figures" / figure.name)
    write_run_manifest(ready, config, state, args, weight_record, artifacts, diagnostics, runtime, started,
                       oom_note)
    manifest = read_manifest()
    return {"status": manifest["status"], "run_directory": str(RUN_DIR), "run_id": manifest["run_id"],
            "ordinal": config["ordinal"], "ordinal_loss_first_epoch": artifacts["ordinal_loss_first_epoch"],
            "ordinal_loss_last_train_epoch": artifacts["ordinal_loss_last_epoch"],
            "tensorboard": diagnostics["tensorboard"], "run_isolation": isolation,
            "final_test": config["run"]["final_test"], "formal_training": "COMPLETED"}


def write_run_manifest(ready: dict, config: dict, state: dict, args: dict, weight_record: dict,
                       artifacts: dict, diagnostics: dict, runtime: dict, started: str,
                       oom_note: str | None) -> dict:
    """Build and persist the E02 run manifest (E01 field names kept for comparability)."""
    env = ready["environment"]
    manifest = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    freeze_text = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                 text=True, check=True).stdout
    (RUN_DIR / "pip-freeze.txt").write_text(freeze_text, encoding="utf-8", newline="\n")
    manifest.update(
        run_id=f"e02_seed0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        status="trained_not_final_tested", git_commit=state["head"], git_dirty=state["dirty"],
        experiment_relevant_git_dirty=state["experiment_relevant_dirty"],
        e01_relevant_git_dirty=state["e01_relevant_dirty"], e02_relevant_git_dirty=state["e02_relevant_dirty"],
        git_status_porcelain=state["status"], experiment_config_sha256=sha(CONFIG),
        resolved_config_sha256=sha(RUN_DIR / "resolved_train_config.yaml"),
        timestamp_start_utc=started, timestamp_end_utc=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(), os=platform.platform(), python=env["python"],
        python_executable=env["python_executable"], platform=env["execution_platform"],
        execution_platform=env["execution_platform"], cloud_provider=env["cloud_provider"],
        cloud_session_type=env["cloud_session_type"],
        container_image_git_commit=env["container_image_git_commit"],
        container_image_build_date=env["container_image_build_date"],
        pytorch=env["torch"], torchvision=env["torchvision"], ultralytics=env["ultralytics"],
        numpy=env["numpy"], opencv=env["opencv"], opencv_python=env["opencv_python"],
        cuda_runtime=env["cuda_runtime"], gpu=env["gpu_name"], gpu_memory_bytes=env["gpu_memory_bytes"],
        gpu_count=env["gpu_count"], visible_gpu_count=env["visible_gpu_count"],
        gpu_inventory=env["gpu_inventory"], selected_formal_device=env["selected_formal_device"],
        raw_source_kind=ready["dataset"]["raw_source_kind"], raw_source_path=ready["raw_source"],
        raw_identity_status=ready["dataset"]["raw_identity_status"],
        raw_identity_evidence_sha256=ready["dataset"].get("raw_identity_evidence_sha256"),
        derived_dataset_yaml=ready["dataset_yaml"], pretrained_weights_sha256=weight_record["sha256"],
        pretrained_weights_filename=weight_record["filename"],
        pretrained_weights_source=weight_record["source"],
        pretrained_weights_mtime_utc=weight_record["mtime_utc"],
        best_checkpoint_sha256=sha(RUN_DIR / "weights/best.pt"),
        last_checkpoint_sha256=sha(RUN_DIR / "weights/last.pt"),
        effective_optimizer=runtime["optimizer"],
        learning_rate_initial=runtime["initial_optimizer_lr"],
        effective_warmup_bias_lr=runtime["warmup_bias_lr"],
        batch=args["batch"], notes=oom_note, augmentation=config["train"]["augmentation"],
        actual_args_yaml=str(RUN_DIR / "args.yaml"), loss_names=diagnostics.get("loss_items"),
        best_checkpoint=str(RUN_DIR / "weights/best.pt"), last_checkpoint=str(RUN_DIR / "weights/last.pt"),
        results_directory=str(RUN_DIR), output_root=str(RUNS_ROOT), ordinal=config["ordinal"],
        ordinal_diagnostics=diagnostics, ordinal_loss_first_epoch=artifacts["ordinal_loss_first_epoch"],
        ordinal_loss_last_train_epoch=artifacts["ordinal_loss_last_epoch"],
        tensorboard=diagnostics["tensorboard"],
        tensorboard_event_files=artifacts["tensorboard"]["event_files"],
        e01_reference={"training_commit": config["e01_reference"]["training_commit"],
                       "analysis_commit": config["e01_reference"]["analysis_commit"],
                       "frozen_results": config["e01_reference"]["frozen_results"],
                       "files_sha256": config["e01_reference"]["files_sha256"]},
        e01_immutability=ready["e01_immutability"], train_e01_parity=ready["train_e01_parity"],
        test_metrics_box=None, test_metrics_mask=None,
        final_test={"status": "locked_inherited_from_e01", "command": None,
                    "note": "E02 exposes validation only; the single Final Test stays in the E01 protocol"})
    write_manifest(manifest)
    return manifest


def preflight(*, require_cuda: bool = False, platform: str = "kaggle", source: Path | None = None,
              derived: Path | None = None, output_dir: Path | None = None,
              allow_existing_run: bool = False) -> dict:
    """Run every non-training E02 readiness gate and stop before any formal training.

    ``allow_existing_run`` is used by the ``val`` command only. Scoring the best
    checkpoint needs the full readiness chain again (raw D2 identity, mandatory
    split columns, CUDA gate, pinned Ultralytics version), but the formal run
    directory must already exist by then. The default stays a strict
    pre-training gate so a formal E02 run can never be overwritten, and
    ``train`` keeps its own unconditional refusal.
    """
    config = load_config()
    immutable = assert_e01_immutable(config)
    parity = {"status": "byte-identical to the frozen E01 train/augmentation block",
              "train_keys": sorted(config["train"]), "execution_environment": "identical to E01"}
    state = git_state()
    isolation = assert_run_isolation()
    existing_run = RUN_DIR.exists()
    if existing_run and not allow_existing_run:
        raise FileExistsError(f"E02 run directory already exists; do not overwrite: {RUN_DIR}")
    source = e01_runner.resolve_source(source, expected_platform=platform)
    derived = e01_runner.resolve_derived(derived, expected_platform=platform)
    converted = dataset.validate_only(source, dataset.DEFAULT_MANIFESTS, dataset.DEFAULT_PROTOCOL, derived)
    classes = verify_derived_classes(derived)
    hardware = environment.inspect(require_cuda=require_cuda, expected_platform=platform)
    if "error" in hardware:
        raise RuntimeError(hardware["error"])
    expected_ultralytics = config["execution_environment"]["compatibility"]["ultralytics"].removeprefix("==")
    if hardware["ultralytics"] != expected_ultralytics:
        raise RuntimeError(f"Ultralytics version mismatch: {hardware['ultralytics']}")
    if require_cuda and hardware["selected_accelerator"] != "cuda:0":
        raise RuntimeError("formal E02 requires cuda:0")
    resolved = resolve_config()
    tensorboard = tensorboard_gate()
    explicit_output = output_dir is not None
    output_dir = (Path(output_dir) if explicit_output else PREFLIGHT_DIR).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_e02_config.yaml"
    record = output_dir / "e02_preflight.json"
    if not explicit_output and (record.exists() or resolved_path.exists()):
        # Never silently rewrite an existing preflight record; rerun with an explicit --preflight-dir.
        written = {"status": "existing preflight record preserved (pass --preflight-dir to refresh)",
                   "resolved_config_path": str(resolved_path), "record": str(record)}
    else:
        e01_resolver.save(resolved_path, resolved)
        written = {"status": "written", "resolved_config_path": str(resolved_path), "record": str(record)}
    summary = {"schema_version": "e02-preflight-v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "experiment_id": EXPERIMENT_ID, "git": state, "e01_immutability": immutable,
               "train_e01_parity": parity, "run_isolation": isolation,
               "run_directory_state": ("existing_formal_run_adopted_for_validation" if existing_run else
                                       "absent_before_formal_run"), "dataset": converted,
               "dataset_classes": classes, "dataset_yaml": str((Path(derived) / "dataset.yaml").resolve()),
               "derived_dataset": str(derived), "raw_source": str(source), "environment": hardware,
               "ordinal": config["ordinal"], "tensorboard": tensorboard, "preflight_output": written,
               "final_test": config["run"]["final_test"], "formal_training": "NOT STARTED",
               "e02_readiness": "READY FOR HUMAN CONFIRMATION"}
    if written["status"] == "written":
        record.write_text(json.dumps(e01_runner.yaml_safe(summary), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")
    return summary


def evaluate(*, platform: str = "kaggle", source: Path | None = None, derived: Path | None = None) -> dict:
    """Independent validation of the E02 best checkpoint; the only split this entrypoint can score."""
    ready = preflight(require_cuda=True, platform=platform, source=source, derived=derived,
                      allow_existing_run=True)
    manifest = read_manifest()
    if sha(RUN_DIR / "resolved_train_config.yaml") != manifest.get("resolved_config_sha256"):
        raise RuntimeError("resolved training config changed after the formal run")
    if manifest.get("validation_metrics_mask") is not None:
        raise RuntimeError("validation already recorded; do not overwrite")
    evaluation_dir = RUN_DIR / "validation"
    if evaluation_dir.exists():
        raise FileExistsError(f"validation output already exists; do not rerun silently: {evaluation_dir}")
    best = RUN_DIR / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    if sha(best) != manifest.get("best_checkpoint_sha256"):
        raise RuntimeError("best checkpoint differs from the run provenance")
    from ior_yolo.trainers.ordinal import OrdinalYOLO

    metrics = OrdinalYOLO(str(best)).val(data=ready["dataset_yaml"], split=EVALUATION_SPLIT, device=0,
                                         project=str(RUN_DIR), name=evaluation_dir.name, exist_ok=False,
                                         plots=True)
    manifest["validation_metrics_box"] = e01_runner.metric_record(metrics.box)
    manifest["validation_metrics_mask"] = e01_runner.metric_record(metrics.seg)
    manifest["status"] = "validated_not_final_tested"
    write_manifest(manifest)
    return {"status": manifest["status"], "split": EVALUATION_SPLIT, "run_directory": str(RUN_DIR),
            "box": manifest["validation_metrics_box"], "mask": manifest["validation_metrics_mask"],
            "final_test": manifest.get("final_test"),
            "note": "Final Test is inherited-locked from E01; this command can only score the frozen val split"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--require-cuda", action="store_true", help="enforce the pinned CUDA stack for preflight")
    parser.add_argument("--platform", choices=tuple(environment.PLATFORM_LABELS), default="kaggle",
                        help="formal execution platform; default is the primary Kaggle plan")
    parser.add_argument("--source", type=Path, help="D2 original ZIP or verified unpacked directory")
    parser.add_argument("--derived", type=Path, help="disposable derived E01 dataset directory")
    parser.add_argument("--batch", type=int, choices=(4, 8), default=8,
                        help="smoke/train: 4 requires documented prior batch-8 CUDA OOM")
    parser.add_argument("--oom-note", help="pretraining OOM evidence/reason when using batch 4")
    parser.add_argument("--device", choices=("cuda:0", "cpu", "mps"), default="cuda:0",
                        help="smoke only; formal E02 train/val always use cuda:0")
    parser.add_argument("--output-dir", type=Path, help="smoke/tensorboard-smoke output root (disposable)")
    parser.add_argument("--preflight-dir", type=Path, help="preflight record directory")
    args = parser.parse_args()
    if args.command in ("train", "val") and args.device != "cuda:0":
        parser.error("formal E02 train/val always use cuda:0; use the smoke command for engineering devices")
    if args.command in ("preflight", "tensorboard-smoke") and (args.oom_note or args.batch != 8):
        parser.error("--batch/--oom-note only apply to smoke/train")
    result = (preflight(require_cuda=args.require_cuda, platform=args.platform, source=args.source,
                        derived=args.derived, output_dir=args.preflight_dir)
              if args.command == "preflight" else
              smoke(device=args.device, batch=args.batch, platform=args.platform, source=args.source,
                    derived=args.derived, output_dir=args.output_dir, oom_note=args.oom_note)
              if args.command == "smoke" else
              tensorboard_smoke(output_dir=args.output_dir)
              if args.command == "tensorboard-smoke" else
              train(batch_override=4 if args.batch == 4 else None, oom_note=args.oom_note,
                    platform=args.platform, source=args.source, derived=args.derived)
              if args.command == "train" else
              evaluate(platform=args.platform, source=args.source, derived=args.derived))
    print(json.dumps(e01_runner.yaml_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
