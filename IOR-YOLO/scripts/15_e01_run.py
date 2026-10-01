"""Pinned E01 preflight and user-visible Windows train/val/final-test entrypoint.

Never run train/val/test on macOS; preflight is read-only on either platform.
Final test is a separate explicit command after validation is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
CONFIG = ROOT / "configs/experiments/e01_yolo11n_seg.yaml"
TEMPLATE = ROOT / "configs/experiments/run_manifest_template.yaml"
DATASET = ROOT / "data/processed/d2_e01_ultralytics"
DEFAULT_ARCHIVE = ROOT / "data/raw/multistage_apple_v4/dataset-20260508.zip"
DEFAULT_MANIFESTS = ROOT / "data/manifests"
DEFAULT_PROTOCOL = ROOT / "configs/data/d2_frozen_protocol.yaml"
RUN_DIR = ROOT / "runs/e01_yolo11n_seg/seed_0"
E01_GIT_PATHS = (
    "IOR-YOLO/configs/data/d2_frozen_protocol.yaml",
    "IOR-YOLO/configs/experiments/e01_yolo11n_seg.yaml",
    "IOR-YOLO/configs/experiments/run_manifest_template.yaml",
    "IOR-YOLO/data/manifests/d2_experiment_pool_frozen.csv",
    "IOR-YOLO/data/manifests/d2_split_frozen.csv",
    "IOR-YOLO/data/manifests/d2_exclusions_frozen.csv",
    "IOR-YOLO/requirements-e01.txt",
    "IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py",
    "IOR-YOLO/scripts/14_check_training_environment.py",
    "IOR-YOLO/scripts/15_e01_run.py",
    "IOR-YOLO/scripts/16_resolve_e01_config.py",
    "IOR-YOLO/scripts/17_analyze_e01_results.py",
    "IOR-YOLO/scripts/18_export_e01_predictions.py",
    "IOR-YOLO/configs/development/e01_local_engineering.yaml",
    "IOR-YOLO/scripts/19_e01_local_engineering.py",
)
FORMAL_DEVICE = "cuda:0"
spec = importlib.util.spec_from_file_location("e01_dataset", Path(__file__).with_name("13_build_e01_ultralytics_dataset.py"))
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)
spec = importlib.util.spec_from_file_location("e01_environment", Path(__file__).with_name("14_check_training_environment.py"))
environment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(environment)
spec = importlib.util.spec_from_file_location("e01_resolver", Path(__file__).with_name("16_resolve_e01_config.py"))
resolver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resolver)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if (config["experiment_id"], config["task"], config["model"], config["training_seed"]) != (
            "e01_yolo11n_seg", "segment", "yolo11n-seg.pt", 0):
        raise ValueError("E01 identity changed")
    checks = {"experiment_pool_sha256": ROOT / "data/manifests/d2_experiment_pool_frozen.csv",
              "dataset_split_sha256": ROOT / "data/manifests/d2_split_frozen.csv",
              "protocol_yaml_sha256": ROOT / "configs/data/d2_frozen_protocol.yaml"}
    for key, path in checks.items():
        if sha(path) != config[key]:
            raise ValueError(f"frozen artifact changed: {key}")
    if config["dataset_protocol_version"] != "d2-v4-stage2b5-v1" or config["split_seed_64"] != 13436313853456744620:
        raise ValueError("E01 config differs from D2 frozen protocol")
    if config["train"]["seed"] != config["training_seed"]:
        raise ValueError("training seed disagreement")
    return config


def preflight(
    *,
    require_cuda: bool = False,
    archive: Path = DEFAULT_ARCHIVE,
    manifest_dir: Path = DEFAULT_MANIFESTS,
    protocol: Path = DEFAULT_PROTOCOL,
    derived_root: Path = DATASET,
) -> dict:
    config = load_config()
    resolved = resolver.resolve(CONFIG)
    converted = dataset.validate_only(archive, manifest_dir, protocol, derived_root)
    hardware = environment.inspect(require_cuda=require_cuda)
    if "error" in hardware:
        raise RuntimeError(hardware["error"])
    if hardware["ultralytics"] != config["ultralytics_version"]:
        raise RuntimeError("Ultralytics version mismatch")
    if require_cuda and hardware["selected_accelerator"] != "cuda:0":
        raise RuntimeError("formal E01 requires cuda:0")
    return {"dataset": converted, "environment": hardware, "experiment_id": config["experiment_id"],
            "optimizer_auto_projection": resolved["optimizer_auto_projection"],
            "dataset_yaml": str((derived_root / "dataset.yaml").resolve()),
            "archive": str(archive.resolve()), "derived_root": str(derived_root.resolve())}


def git_state() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT.parent,
                            capture_output=True, text=True, check=True).stdout.splitlines()
    relevant = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all", "--", *E01_GIT_PATHS],
                              cwd=ROOT.parent, capture_output=True, text=True, check=True).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status,
            "experiment_relevant_dirty": bool(relevant), "experiment_relevant_status": relevant}


def pretrained_weight_record(model=None) -> tuple[object, dict]:
    from ultralytics import YOLO
    model = model or YOLO(formal_model_name())
    path = Path(getattr(model, "ckpt_path", "") or "")
    if not path.is_file() or path.name != "yolo11n-seg.pt":
        raise RuntimeError("official YOLO11n-seg pretrained checkpoint is unavailable; stop without random initialization")
    return model, {"filename": path.name,
                   "source": "Ultralytics official yolo11n-seg.pt loaded by ultralytics==8.3.220",
                   "sha256": sha(path),
                   "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                   "local_path": str(path.resolve())}


def formal_model_name(config: dict | None = None) -> str:
    config = config or load_config()
    if config["model"] != "yolo11n-seg.pt" or config["pretrained"] is not True:
        raise ValueError("formal E01 must initialize from official pretrained yolo11n-seg.pt")
    return config["model"]


def verify_weights(**data_paths) -> dict:
    preflight(require_cuda=True, **data_paths)
    _, record = pretrained_weight_record()
    return {"status": "official_pretrained_weight_verified", **record}


def smoke(
    *,
    batch: int = 8,
    oom_note: str | None = None,
    **data_paths,
) -> dict:
    """Run a disposable one-epoch/tiny-fraction CUDA feasibility check, never a formal result."""
    ready = preflight(require_cuda=True, **data_paths)
    if batch not in (8, 4) or (batch == 4) != bool(oom_note):
        raise ValueError("smoke uses batch 8; batch 4 requires the prior batch-8 CUDA OOM note")
    state = git_state()
    if state["experiment_relevant_dirty"]:
        raise RuntimeError(f"E01-relevant Git files are uncommitted: {state['experiment_relevant_status']}")
    model, weights = pretrained_weight_record()
    config = load_config()
    args = {key: value for key, value in config["train"].items() if key != "augmentation"}
    args.update(config["train"]["augmentation"])
    args.update(data=ready["dataset_yaml"], device=FORMAL_DEVICE, batch=batch, epochs=1, fraction=0.02,
                workers=0, val=False, save=False, plots=False)
    with tempfile.TemporaryDirectory(prefix="e01-cuda-smoke-") as temp:
        model.train(**args, project=temp, name="batch_feasibility", exist_ok=False)
    return {"status": "CUDA smoke passed; not a formal experiment", "batch": batch,
            "oom_note": oom_note, "pretrained_weight": weights}


def write_manifest(value: dict) -> None:
    path = RUN_DIR / "run_manifest.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")


def read_manifest() -> dict:
    path = RUN_DIR / "run_manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"formal run manifest absent: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def metric_record(metric) -> dict:
    return {"precision": float(metric.mp), "recall": float(metric.mr),
            "mAP50": float(metric.map50), "mAP50_95": float(metric.map),
            "per_class": {str(int(class_id)): {"name": dataset.CLASSES[int(class_id)],
                                               "precision": float(metric.p[i]),
                                               "recall": float(metric.r[i]),
                                               "mAP50": float(metric.ap50[i]),
                                               "mAP50_95": float(metric.ap[i])}
                          for i, class_id in enumerate(metric.ap_class_index)}}


def train(
    *,
    batch_override: int | None = None,
    oom_note: str | None = None,
    **data_paths,
) -> dict:
    ready = preflight(require_cuda=True, **data_paths)
    config = load_config()
    if (batch_override == 4) != bool(oom_note):
        raise ValueError("batch 4 requires a pretraining CUDA OOM note; no note permitted without fallback")
    state = git_state()
    if state["experiment_relevant_dirty"]:
        raise RuntimeError(f"E01-relevant Git files are uncommitted: {state['experiment_relevant_status']}")
    if RUN_DIR.exists():
        raise FileExistsError(f"formal E01 run already exists; no overwrite: {RUN_DIR}")
    # Weight acquisition/verification is a preflight operation. If it fails,
    # stop without creating a formal-run snapshot or falling back to random initialization.
    model, weight_record = pretrained_weight_record()
    # Persist the requested/default/resolved preflight state BEFORE training.
    pretrain_path = RUN_DIR.parent / "seed_0_resolved_train_config.yaml"
    if pretrain_path.exists():
        raise FileExistsError(f"pretrain snapshot already exists: {pretrain_path}")
    resolved = resolver.resolve(CONFIG, batch_override=batch_override)
    if oom_note:
        resolved["hardware_oom_note"] = oom_note
    resolver.save(pretrain_path, resolved)
    started = datetime.now(timezone.utc).isoformat()
    args = {key: value for key, value in config["train"].items() if key != "augmentation"}
    args.update(config["train"]["augmentation"])
    if batch_override == 4:
        args["batch"] = 4
    args.update(data=ready["dataset_yaml"], device=FORMAL_DEVICE, project=str(RUN_DIR.parent),
                name=RUN_DIR.name, exist_ok=False, pretrained=True)
    model.train(**args)
    actual_dir = Path(model.trainer.save_dir).resolve()
    if actual_dir != RUN_DIR.resolve():
        raise RuntimeError(f"run directory unexpectedly changed: {actual_dir}")
    optimizer = model.trainer.optimizer
    resolved["status"] = "runtime_observed"
    resolved["actual_runtime"] = {
        "optimizer": type(optimizer).__name__,
        "initial_optimizer_lr": float(optimizer.defaults["lr"]),
        "optimizer_defaults": {str(k): str(v) for k, v in optimizer.defaults.items()},
        "warmup_bias_lr": float(model.trainer.args.warmup_bias_lr),
        "train_loader_images": len(model.trainer.train_loader.dataset),
        "trainer_batch_size": model.trainer.batch_size,
        "trainer_epochs_budget": model.trainer.epochs,
        "actual_epochs_from_results_csv": max(0, sum(1 for _ in (RUN_DIR / "results.csv").open(encoding="utf-8")) - 1)
            if (RUN_DIR / "results.csv").is_file() else None,
        "ultralytics_args_yaml": "args.yaml",
    }
    resolver.save(RUN_DIR / "resolved_train_config.yaml", resolved)
    for folder in ("metrics", "predictions", "figures"):
        (RUN_DIR / folder).mkdir(exist_ok=True)
    if (RUN_DIR / "results.csv").is_file():
        shutil.copyfile(RUN_DIR / "results.csv", RUN_DIR / "metrics/results.csv")
    for figure in RUN_DIR.glob("*.png"):
        shutil.copyfile(figure, RUN_DIR / "figures" / figure.name)
    manifest = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    env = ready["environment"]
    freeze_text = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                                 text=True, check=True).stdout
    (RUN_DIR / "pip-freeze.txt").write_text(freeze_text, encoding="utf-8", newline="\n")
    manifest.update(run_id=f"e01_seed0_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
                    status="trained_not_final_tested", git_commit=state["head"], git_dirty=state["dirty"],
                    experiment_relevant_git_dirty=state["experiment_relevant_dirty"],
                    git_status_porcelain=state["status"],
                    experiment_config_sha256=sha(CONFIG),
                    resolved_config_sha256=sha(RUN_DIR / "resolved_train_config.yaml"),
                    timestamp_start_utc=started, timestamp_end_utc=datetime.now(timezone.utc).isoformat(),
                    hostname=socket.gethostname(), os=platform.platform(), python=env["python"],
                    pytorch=env["torch"], torchvision=env["torchvision"],
                    ultralytics=env["ultralytics"], cuda_runtime=env["cuda_runtime"],
                    gpu=env["gpu_name"], gpu_memory_bytes=env["gpu_memory_bytes"],
                    derived_dataset_yaml=ready["dataset_yaml"], pretrained_weights_sha256=weight_record["sha256"],
                    pretrained_weights_filename=weight_record["filename"],
                    pretrained_weights_source=weight_record["source"],
                    pretrained_weights_mtime_utc=weight_record["mtime_utc"],
                    best_checkpoint_sha256=sha(RUN_DIR / "weights/best.pt"),
                    last_checkpoint_sha256=sha(RUN_DIR / "weights/last.pt"),
                    effective_optimizer=type(optimizer).__name__,
                    learning_rate_initial=float(optimizer.defaults["lr"]),
                    effective_warmup_bias_lr=float(model.trainer.args.warmup_bias_lr),
                    batch=args["batch"], notes=oom_note,
                    augmentation=config["train"]["augmentation"],
                    actual_args_yaml=str(RUN_DIR / "args.yaml"),
                    best_checkpoint=str(RUN_DIR / "weights/best.pt"),
                    last_checkpoint=str(RUN_DIR / "weights/last.pt"),
                    results_directory=str(RUN_DIR))
    write_manifest(manifest)
    return {"status": manifest["status"], "run_directory": str(RUN_DIR)}


def evaluate(split: str, *, final_test: bool = False, **data_paths) -> dict:
    if split not in ("val", "test"):
        raise ValueError("only val and test are supported")
    if split == "test" and not final_test:
        raise PermissionError("FINAL TEST EVALUATION requires --final-test")
    if split == "val" and final_test:
        raise ValueError("--final-test is only valid for test")
    if split == "test":
        print("FINAL TEST EVALUATION — one-time locked evaluation; no model selection")
    ready = preflight(require_cuda=True, **data_paths)
    manifest = read_manifest()
    if sha(RUN_DIR / "resolved_train_config.yaml") != manifest.get("resolved_config_sha256"):
        raise RuntimeError("resolved training config changed after formal run")
    if split == "test" and (manifest["validation_metrics_mask"] is None or manifest["test_metrics_mask"] is not None):
        raise RuntimeError("final test requires prior validation and is allowed only once")
    if split == "val" and manifest["validation_metrics_mask"] is not None:
        raise RuntimeError("validation already recorded; do not overwrite")
    evaluation_dir = RUN_DIR / ("validation" if split == "val" else "final_test")
    if evaluation_dir.exists():
        raise FileExistsError(f"evaluation output already exists; do not rerun silently: {evaluation_dir}")
    best = RUN_DIR / "weights/best.pt"
    if not best.is_file():
        raise FileNotFoundError(best)
    from ultralytics import YOLO

    metrics = YOLO(str(best)).val(data=ready["dataset_yaml"], split=split, device=FORMAL_DEVICE,
                                   project=str(RUN_DIR), name=evaluation_dir.name,
                                   exist_ok=False, plots=True)
    manifest[f"{'validation' if split == 'val' else 'test'}_metrics_box"] = metric_record(metrics.box)
    manifest[f"{'validation' if split == 'val' else 'test'}_metrics_mask"] = metric_record(metrics.seg)
    if split == "test":
        manifest["status"] = "final_test_completed"
    write_manifest(manifest)
    return {"status": manifest["status"], "split": split,
            "box": manifest[f"{'validation' if split == 'val' else 'test'}_metrics_box"],
            "mask": manifest[f"{'validation' if split == 'val' else 'test'}_metrics_mask"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "verify-weights", "smoke", "train", "val", "test"))
    parser.add_argument("--require-cuda", action="store_true", help="require the pinned CUDA:0 runtime")
    parser.add_argument("--batch", type=int, choices=(4, 8), default=8,
                        help="smoke/train: 4 requires documented prior batch-8 CUDA OOM")
    parser.add_argument("--oom-note", help="pretraining OOM evidence/reason when using batch 4")
    parser.add_argument("--final-test", action="store_true", help="explicitly unlock the one-time final test")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFESTS)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--derived-root", type=Path, default=DATASET)
    args = parser.parse_args()
    if args.final_test and args.command != "test":
        parser.error("--final-test only applies to test")
    data_paths = {"archive": args.archive, "manifest_dir": args.manifest_dir,
                  "protocol": args.protocol, "derived_root": args.derived_root}
    result = (preflight(require_cuda=args.require_cuda, **data_paths) if args.command == "preflight" else
              verify_weights(**data_paths) if args.command == "verify-weights" else
              smoke(batch=args.batch, oom_note=args.oom_note, **data_paths) if args.command == "smoke" else
              train(batch_override=4 if args.batch == 4 else None, oom_note=args.oom_note, **data_paths)
              if args.command == "train" else evaluate(args.command, final_test=args.final_test, **data_paths))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
