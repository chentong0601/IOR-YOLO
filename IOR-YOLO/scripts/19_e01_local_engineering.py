"""Local-only E01 engineering preflight and CPU/MPS/CUDA train/val checks."""

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
from pathlib import Path, PureWindowsPath

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEVELOPMENT_PROFILE = ROOT / "configs/development/e01_local_engineering.yaml"
MANIFEST_DIR = ROOT / "data/manifests"
PROTOCOL = ROOT / "configs/data/d2_frozen_protocol.yaml"
DEFAULT_ARCHIVE = ROOT / "data/raw/multistage_apple_v4/dataset-20260508.zip"
DEFAULT_DERIVED = ROOT / "data/processed/d2_e01_ultralytics"
ENGINEERING_DATA = ROOT / "data/processed/e01_engineering"
ENGINEERING_RUNS = ROOT / "runs/e01_engineering"
DATASET_SCRIPT = ROOT / "scripts/13_build_e01_ultralytics_dataset.py"
ANALYSIS_SCRIPT = ROOT / "scripts/17_analyze_e01_results.py"
FORMAL_RUNNER_SCRIPT = ROOT / "scripts/15_e01_run.py"
ENGINEERING_MARK = "ENGINEERING VALIDATION ONLY - NOT FOR PAPER"
FORMAL_DEVICE = "cuda:0"


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load project script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dataset = load_script(DATASET_SCRIPT, "e01_local_dataset")
formal_runner = load_script(FORMAL_RUNNER_SCRIPT, "e01_local_formal_runner")
analysis = load_script(ANALYSIS_SCRIPT, "e01_local_analysis")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def is_absolute_for_platform(value: str, platform_name: str) -> bool:
    if platform_name == "Windows":
        return PureWindowsPath(value).is_absolute()
    return Path(value).expanduser().is_absolute()


def load_profile() -> dict:
    profile = yaml.safe_load(DEVELOPMENT_PROFILE.read_text(encoding="utf-8"))
    if (profile["profile_id"], profile["task"], profile["model"]) != (
            "e01_local_engineering_v1", "segment", "yolo11n-seg.pt"):
        raise ValueError("local engineering profile identity changed")
    if profile["rules"]["split_access"] != ["train", "val"] or profile["rules"]["test_access"] != "forbidden":
        raise ValueError("engineering profile must be restricted to Train/Val")
    return profile


def verified_formal_config(*, protocol: Path = PROTOCOL, manifest_dir: Path = MANIFEST_DIR) -> dict:
    config = formal_runner.load_config()
    required = {
        "experiment_id": "e01_yolo11n_seg",
        "task": "segment",
        "model": "yolo11n-seg.pt",
        "pretrained": True,
        "device_policy": "cuda:0_required",
        "training_seed": 0,
        "dataset_protocol_version": "d2-v4-stage2b5-v1",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"formal E01 field changed: {key}")
    train = config["train"]
    for key, expected in (("imgsz", 640), ("epochs", 100), ("batch", 8), ("optimizer", "auto"), ("seed", 0)):
        if train.get(key) != expected:
            raise ValueError(f"formal E01 training field changed: {key}")
    if digest(protocol) != config["protocol_yaml_sha256"]:
        raise ValueError(f"frozen D2 protocol changed: {protocol}")
    for key, path in (
        ("experiment_pool_sha256", manifest_dir / "d2_experiment_pool_frozen.csv"),
        ("dataset_split_sha256", manifest_dir / "d2_split_frozen.csv"),
    ):
        if path is not None and digest(path) != config[key]:
            raise ValueError(f"frozen D2 artifact changed: {path.name}")
    protocol_config = dataset.read_protocol(protocol)
    for name, expected_hash in protocol_config["frozen_csv_sha256"].items():
        path = manifest_dir / name
        if digest(path) != expected_hash:
            raise ValueError(f"frozen D2 artifact changed: {path.name}")
    return config


def split_rows(manifest_dir: Path) -> list[dict[str, str]]:
    with (manifest_dir / "d2_split_frozen.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def select_engineering_rows(rows: list[dict[str, str]], train_limit: int, val_limit: int) -> dict[str, list[dict[str, str]]]:
    selected = {}
    for split, limit in (("train", train_limit), ("val", val_limit)):
        candidates = [row for row in rows if row["final_split"] == split]
        if len(candidates) < limit or limit < 1:
            raise ValueError(f"insufficient frozen {split} rows for engineering subset")
        if len({row["filename"] for row in candidates}) != len(candidates):
            raise ValueError(f"ambiguous filenames in frozen {split} manifest")
        selected[split] = sorted(
            candidates,
            key=lambda row: (hashlib.sha256(f"{split}:{row['filename']}".encode()).hexdigest(),
                             row["filename"]),
        )[:limit]
    return selected


def validate_label(label_path: Path, expected_targets: int) -> tuple[int, int, int]:
    counts = [0, 0, 0]
    lines = label_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != expected_targets:
        raise ValueError(f"target count mismatch: {label_path}")
    for line in lines:
        values = line.split()
        if len(values) < 7 or len(values) % 2 != 1:
            raise ValueError(f"invalid segmentation polygon: {label_path}")
        class_id = int(values[0])
        coords = [float(value) for value in values[1:]]
        if class_id not in (0, 1, 2) or not all(math.isfinite(value) and 0 <= value <= 1 for value in coords):
            raise ValueError(f"invalid class/polygon coordinate: {label_path}")
        counts[class_id] += 1
    return tuple(counts)


def validate_derived_train_val(derived_root: Path, manifest_dir: Path) -> dict:
    rows = split_rows(manifest_dir)
    split_expected = {"train": (769, 646, 297, 512), "val": (165, 143, 65, 115)}
    checked = {}
    for split, expected in split_expected.items():
        selected = [row for row in rows if row["final_split"] == split]
        if len(selected) != expected[0]:
            raise ValueError(f"frozen {split} count changed: {len(selected)} != {expected[0]}")
        images_dir, labels_dir = derived_root / "images" / split, derived_root / "labels" / split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"derived Train/Val folders absent: {split}")
        images = {path.name for path in images_dir.glob("*.jpg")}
        labels = {path.name for path in labels_dir.glob("*.txt")}
        names = {row["filename"] for row in selected}
        expected_labels = {Path(name).stem + ".txt" for name in names}
        if images != names or labels != expected_labels:
            raise ValueError(f"derived {split} files disagree with frozen manifest")
        totals = [0, 0, 0]
        for row in selected:
            name = row["filename"]
            if Path(name).name != name:
                raise ValueError(f"unsafe frozen image filename: {name}")
            image_path = images_dir / name
            with Image.open(image_path) as image:
                image.verify()
                if image.format != "JPEG":
                    raise ValueError(f"derived image is not JPEG: {image_path}")
            expected_targets = int(row["derived_target_count"])
            actual_classes = validate_label(labels_dir / (Path(name).stem + ".txt"), expected_targets)
            row_classes = tuple(int(row[key]) for key in (
                "immature_instances", "semi_mature_instances", "mature_instances"))
            if actual_classes != row_classes:
                raise ValueError(f"derived {split} label classes disagree with frozen manifest: {name}")
            totals = [left + right for left, right in zip(totals, actual_classes)]
        if (len(selected), *totals) != expected:
            raise ValueError(f"derived {split} counts changed: {(len(selected), *totals)} != {expected}")
        checked[split] = {"images": len(selected), "instances": sum(totals)}
    return checked


def profile_output(profile_name: str) -> Path:
    if profile_name not in ("local-smoke", "local-quick"):
        raise ValueError("unknown engineering profile")
    return ENGINEERING_DATA / profile_name


def build_subset(
    derived_root: Path,
    manifest_dir: Path,
    profile_name: str,
    *,
    output: Path | None = None,
) -> dict:
    profile = load_profile()
    settings = profile["modes"][profile_name]
    validate_derived_train_val(derived_root, manifest_dir)
    manifest_path = manifest_dir / "d2_split_frozen.csv"
    selected = select_engineering_rows(
        split_rows(manifest_dir), settings["train_images"], settings["val_images"])
    output = (output or profile_output(profile_name)).resolve()
    if not output.is_relative_to(ENGINEERING_DATA.resolve()):
        raise ValueError("engineering subset output must stay under data/processed/e01_engineering")
    if output.exists():
        current = yaml.safe_load((output / "engineering_subset_manifest.yaml").read_text(encoding="utf-8"))
        expected_ids = {split: [row["filename"] for row in records] for split, records in selected.items()}
        if (current.get("status") != ENGINEERING_MARK or current.get("profile") != profile_name or
                current.get("source_split_sha256") != digest(manifest_path) or
                current.get("source_image_ids") != expected_ids):
            raise FileExistsError(f"existing engineering subset differs; preserve it and choose a new output: {output}")
        verify_subset(output, current)
        return {"status": "existing engineering subset verified", "subset": str(output), **expected_ids}

    data_yaml = yaml.safe_load((derived_root / "dataset.yaml").read_text(encoding="utf-8"))
    if data_yaml.get("names") != {0: "immature apple", 1: "semi-mature apple", 2: "mature apple"}:
        raise ValueError("derived E01 class mapping changed")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="e01-engineering-subset-", dir=output.parent) as temporary:
        stage = Path(temporary) / "subset"
        records = {}
        for split, split_records in selected.items():
            (stage / "images" / split).mkdir(parents=True)
            (stage / "labels" / split).mkdir(parents=True)
            records[split] = []
            for row in split_records:
                name = row["filename"]
                source_image = derived_root / "images" / split / name
                source_label = derived_root / "labels" / split / (Path(name).stem + ".txt")
                target_image = stage / "images" / split / name
                target_label = stage / "labels" / split / source_label.name
                shutil.copyfile(source_image, target_image)
                shutil.copyfile(source_label, target_label)
                records[split].append({
                    "image_id": name,
                    "source_group_id": row["source_group_id"],
                    "image_sha256": digest(target_image),
                    "label_sha256": digest(target_label),
                })
        subset_yaml = {
            "path": str(output),
            "train": "images/train",
            "val": "images/val",
            "names": data_yaml["names"],
        }
        (stage / "dataset.yaml").write_text(
            yaml.safe_dump(subset_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        subset_manifest = {
            "status": ENGINEERING_MARK,
            "profile": profile_name,
            "protocol_version": "d2-v4-stage2b5-v1",
            "source_split_sha256": digest(manifest_path),
            "source_derived_root": str(derived_root.resolve()),
            "source_image_ids": {split: [row["filename"] for row in split_records]
                                 for split, split_records in selected.items()},
            "source_records": records,
            "split_access": ["train", "val"],
            "test_access": False,
            "dataset_yaml": str((output / "dataset.yaml").resolve()),
            "selection": "first N frozen Train/Val image IDs sorted by SHA256(split:filename), then filename",
        }
        (stage / "engineering_subset_manifest.yaml").write_text(
            yaml.safe_dump(subset_manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8", newline="\n")
        stage.replace(output)
    verify_subset(output, subset_manifest)
    return {"status": ENGINEERING_MARK, "subset": str(output),
            "source_image_ids": subset_manifest["source_image_ids"],
            "dataset_yaml": str(output / "dataset.yaml")}


def verify_subset(subset_root: Path, manifest: dict) -> None:
    dataset_yaml = yaml.safe_load((subset_root / "dataset.yaml").read_text(encoding="utf-8"))
    if "test" in dataset_yaml or manifest.get("test_access") is not False:
        raise PermissionError("engineering dataset must not contain a Test split")
    for split in ("train", "val"):
        expected_ids = manifest["source_image_ids"][split]
        records = manifest["source_records"][split]
        if [row["image_id"] for row in records] != expected_ids:
            raise ValueError(f"engineering subset manifest IDs disagree: {split}")
        for record in records:
            image = subset_root / "images" / split / record["image_id"]
            label = subset_root / "labels" / split / (Path(record["image_id"]).stem + ".txt")
            if digest(image) != record["image_sha256"] or digest(label) != record["label_sha256"]:
                raise ValueError(f"engineering subset files changed: {split}/{record['image_id']}")


def check_device(device: str, torch_module=None) -> str:
    profile = load_profile()
    if device not in profile["device_choices"]:
        raise ValueError(f"unsupported engineering device: {device}")
    if torch_module is None:
        import torch as torch_module
    if device == "mps":
        if not (hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available()):
            raise RuntimeError("requested device mps is unavailable")
    elif device == "cuda:0":
        if not torch_module.cuda.is_available():
            raise RuntimeError("requested device cuda:0 is unavailable")
    return device


def environment_report(device: str) -> dict:
    import torch
    import torchvision
    import ultralytics

    if ultralytics.__version__ != "8.3.220":
        raise RuntimeError(f"Ultralytics must be 8.3.220, got {ultralytics.__version__}")
    check_device(device, torch)
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }


def git_provenance() -> dict:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status": status}


def pretrained_record(model) -> dict:
    path = Path(getattr(model, "ckpt_path", "") or "")
    if not path.is_file() or path.name != "yolo11n-seg.pt":
        raise RuntimeError("official pretrained yolo11n-seg.pt is required; random or engineering weights are forbidden")
    return {
        "filename": path.name,
        "source": "Ultralytics official checkpoint loaded by ultralytics==8.3.220",
        "sha256": digest(path),
        "path": str(path.resolve()),
    }


def preflight(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    derived_root: Path = DEFAULT_DERIVED,
    manifest_dir: Path = MANIFEST_DIR,
    protocol: Path = PROTOCOL,
    device: str = "cpu",
    profile_name: str = "local-quick",
) -> dict:
    config = verified_formal_config(protocol=protocol, manifest_dir=manifest_dir)
    if not archive.is_file():
        raise FileNotFoundError(f"raw source ZIP not found: {archive}")
    archive_sha = digest(archive)
    if archive_sha != config["dataset_zip_sha256"]:
        raise ValueError("raw source ZIP SHA256 differs from frozen D2 protocol")
    protocol_hash = digest(protocol)
    derived = validate_derived_train_val(derived_root, manifest_dir)
    subset = build_subset(derived_root, manifest_dir, profile_name)
    environment = environment_report(device)
    if not ENGINEERING_RUNS.is_dir():
        ENGINEERING_RUNS.mkdir(parents=True)
    if not os.access(ENGINEERING_RUNS, os.W_OK):
        raise PermissionError(f"engineering run output is not writable: {ENGINEERING_RUNS}")
    from ultralytics import YOLO

    model = YOLO(config["model"])
    weights = pretrained_record(model)
    return {
        "status": "PASS",
        "git": git_provenance(),
        "scientific_protocol": "UNCHANGED",
        "formal_protocol": {
            "task": config["task"], "model": config["model"], "imgsz": config["train"]["imgsz"],
            "epochs": config["train"]["epochs"], "batch": config["train"]["batch"],
            "optimizer": config["train"]["optimizer"], "seed": config["training_seed"],
            "device": FORMAL_DEVICE, "split_counts": {"train": 769, "val": 165, "test": 165},
            "protocol_sha256": protocol_hash,
        },
        "raw_source": {"path": str(archive.resolve()), "sha256": archive_sha},
        "derived_train_val": derived,
        "engineering_subset": subset,
        "environment": environment,
        "official_pretrained_weights": weights,
        "engineering_output": str(ENGINEERING_RUNS.resolve()),
        "final_test_access": "NOT ACCESSED; engineering data YAML contains Train/Val only",
        "formal_training": "NOT STARTED",
    }


def metrics_record(metric) -> dict:
    return {
        "precision": float(metric.mp),
        "recall": float(metric.mr),
        "mAP50": float(metric.map50),
        "mAP50_95": float(metric.map),
    }


def export_engineering_predictions(model, subset_root: Path, run_dir: Path, device: str, imgsz: int) -> dict:
    output = run_dir / "predictions"
    output.mkdir(exist_ok=False)
    source = subset_root / "images" / "val"
    expected = {path.name for path in source.glob("*.jpg")}
    predictions = []
    stream = model.predict(source=str(source), stream=True, device=device, imgsz=imgsz,
                           save=False, verbose=False)
    seen = set()
    for result in stream:
        image_id = Path(result.path).name
        if image_id not in expected or image_id in seen:
            raise ValueError(f"unknown or repeated engineering Val prediction: {image_id}")
        seen.add(image_id)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        if result.masks is None or len(result.masks.xy) != len(boxes):
            raise ValueError(f"segmentation masks missing from engineering Val prediction: {image_id}")
        height, width = result.orig_shape
        for index, (box, polygon) in enumerate(zip(boxes, result.masks.xy)):
            predictions.append({
                "image_id": image_id,
                "prediction_id": f"p{index:04d}",
                "class_id": int(box.cls[0].item()),
                "confidence": float(box.conf[0].item()),
                "box_xyxy": [float(value) for value in box.xyxy[0].tolist()],
                "polygon_normalized": [[float(x) / width, float(y) / height] for x, y in polygon],
            })
    if seen != expected:
        raise ValueError(f"engineering Val prediction missed images: {sorted(expected - seen)}")
    path = output / "predictions_val.json"
    path.write_text(json.dumps({
        "status": ENGINEERING_MARK,
        "split": "val",
        "images": sorted(seen),
        "predictions": predictions,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {"path": str(path), "images": len(seen), "predictions": len(predictions)}


def engineering_analysis(run_dir: Path, metrics: dict, prediction_info: dict) -> dict:
    report = analysis.parse_training_csv(run_dir / "results.csv")
    result = {
        "status": ENGINEERING_MARK,
        "interpretation": "software-pipeline diagnostics only; metrics are not scientific results",
        "split": "val",
        "training_csv": str(run_dir / "results.csv"),
        "training_csv_sha256": digest(run_dir / "results.csv"),
        "epochs_logged": report["epochs_logged"],
        "last_epoch": report["last_epoch"],
        "last_epoch_metrics": report["last_epoch_validation"],
        "validation_metrics": metrics,
        "prediction_export": prediction_info,
        "final_test_access": False,
    }
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(exist_ok=False)
    (analysis_dir / "engineering_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curves = report["loss_curves"] | report["validation_metric_curves"]
    if curves:
        figure, axis = plt.subplots(figsize=(9, 5))
        for label, values in curves.items():
            axis.plot(range(1, len(values) + 1), values, label=label)
        axis.set(xlabel="logged epoch", ylabel="engineering diagnostic")
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(analysis_dir / "engineering_curves.png", dpi=140)
        plt.close(figure)
    return result


def run_engineering(
    mode: str,
    *,
    device: str = "cpu",
    archive: Path = DEFAULT_ARCHIVE,
    derived_root: Path = DEFAULT_DERIVED,
    manifest_dir: Path = MANIFEST_DIR,
    protocol: Path = PROTOCOL,
) -> dict:
    if mode not in ("local-smoke", "local-quick"):
        raise ValueError("only local-smoke and local-quick are engineering run modes")
    config = verified_formal_config(protocol=protocol, manifest_dir=manifest_dir)
    if not archive.is_file():
        raise FileNotFoundError(f"raw source ZIP not found: {archive}")
    archive_sha = digest(archive)
    if archive_sha != config["dataset_zip_sha256"]:
        raise ValueError("raw source ZIP SHA256 differs from frozen D2 protocol")
    environment = environment_report(device)
    settings = load_profile()["modes"][mode]
    validate_derived_train_val(derived_root, manifest_dir)
    subset_result = build_subset(derived_root, manifest_dir, mode)
    subset_root = Path(subset_result["subset"])
    subset_manifest = yaml.safe_load((subset_root / "engineering_subset_manifest.yaml").read_text(encoding="utf-8"))
    dataset_yaml = subset_root / "dataset.yaml"
    data_cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if "test" in data_cfg:
        raise PermissionError("engineering model input may contain only Train and Val")

    from ultralytics import YOLO

    model = YOLO(config["model"])
    weights = pretrained_record(model)
    started = datetime.now(timezone.utc).isoformat()
    run_name = f"{mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    ENGINEERING_RUNS.mkdir(parents=True, exist_ok=True)
    run_dir = ENGINEERING_RUNS / run_name
    if run_dir.exists():
        raise FileExistsError(f"engineering output already exists: {run_dir}")
    profile = load_profile()
    (ENGINEERING_RUNS / "engineering_profile.yaml").write_text(
        yaml.safe_dump(profile, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    args = {key: value for key, value in config["train"].items() if key != "augmentation"}
    args.update(config["train"]["augmentation"])
    args.update(
        data=str(dataset_yaml.resolve()),
        device=device,
        imgsz=settings["imgsz"],
        epochs=settings["epochs"],
        batch=settings["batch"],
        workers=settings["workers"],
        amp=profile["rules"]["runtime_overrides"]["amp"],
        val=settings["validation_during_training"],
        save=True,
        plots=mode == "local-quick",
        project=str(ENGINEERING_RUNS.resolve()),
        name=run_name,
        exist_ok=False,
        pretrained=True,
        fraction=1.0,
    )
    model.train(**args)
    actual_dir = Path(model.trainer.save_dir).resolve()
    if actual_dir != run_dir.resolve():
        raise RuntimeError(f"engineering output path changed unexpectedly: {actual_dir}")
    required = ["weights/last.pt", "args.yaml", "results.csv"]
    if mode == "local-quick":
        required.append("weights/best.pt")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"engineering run did not persist required outputs: {missing}")

    validation = None
    prediction_info = None
    analysis_report = None
    if mode == "local-quick":
        validation_dir = run_dir / "validation"
        if validation_dir.exists():
            raise FileExistsError(f"engineering validation output already exists: {validation_dir}")
        val_model = YOLO(str(run_dir / "weights/best.pt"))
        val_metrics = val_model.val(
            data=str(dataset_yaml.resolve()), split="val", device=device, imgsz=settings["imgsz"],
            batch=settings["batch"], workers=0, project=str(run_dir.resolve()),
            name="validation", exist_ok=False, plots=True,
        )
        validation = {"box": metrics_record(val_metrics.box), "mask": metrics_record(val_metrics.seg)}
        prediction_info = export_engineering_predictions(
            val_model, subset_root, run_dir, device, settings["imgsz"])
        analysis_report = engineering_analysis(run_dir, validation, prediction_info)

    git = git_provenance()
    manifest = {
        "status": ENGINEERING_MARK,
        "run_id": run_name,
        "mode": mode,
        "device": device,
        "task": "segment",
        "initialization": weights,
        "formal_initialization_allowed": False,
        "scientific_protocol": "UNCHANGED",
        "formal_protocol": {
            "model": config["model"], "imgsz": config["train"]["imgsz"],
            "epochs": config["train"]["epochs"], "batch": config["train"]["batch"],
            "optimizer": config["train"]["optimizer"], "seed": config["training_seed"],
            "device": FORMAL_DEVICE,
        },
        "engineering_overrides": settings | {"amp": False, "device": device},
        "optimizer": config["train"]["optimizer"],
        "augmentation": config["train"]["augmentation"],
        "split_access": ["train", "val"],
        "final_test_access": False,
        "source_image_ids": subset_manifest["source_image_ids"],
        "frozen_split_sha256": digest(manifest_dir / "d2_split_frozen.csv"),
        "raw_source_zip_sha256": archive_sha,
        "engineering_subset_manifest_sha256": digest(subset_root / "engineering_subset_manifest.yaml"),
        "git_head": git["head"],
        "git_dirty": git["dirty"],
        "git_status": git["status"],
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "environment": environment,
        "ultralytics": environment["ultralytics"],
        "timestamp_start_utc": started,
        "timestamp_end_utc": datetime.now(timezone.utc).isoformat(),
        "results_csv": str(run_dir / "results.csv"),
        "args_yaml": str(run_dir / "args.yaml"),
        "last_checkpoint": str(run_dir / "weights/last.pt"),
        "last_checkpoint_sha256": digest(run_dir / "weights/last.pt"),
        "best_checkpoint": str(run_dir / "weights/best.pt") if mode == "local-quick" else None,
        "validation": validation,
        "prediction_export": prediction_info,
        "analysis": analysis_report,
        "interpretation": "engineering pipeline validation only; not for paper or scientific comparison",
    }
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
    return {"status": ENGINEERING_MARK, "mode": mode, "run_directory": str(run_dir),
            "validation": validation, "prediction_export": prediction_info,
            "analysis": str(run_dir / "analysis" / "engineering_validation.json") if analysis_report else None}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    subparsers = command.add_subparsers(dest="command", required=True)
    for name in ("check-protocol", "preflight", "prepare-subset", "local-smoke", "local-quick"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--device", choices=("cpu", "mps", "cuda:0"), default="cpu")
        sub.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
        sub.add_argument("--derived-root", type=Path, default=DEFAULT_DERIVED)
        sub.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
        sub.add_argument("--protocol", type=Path, default=PROTOCOL)
        if name == "preflight":
            sub.add_argument("--profile", choices=("local-smoke", "local-quick"), default="local-quick")
        if name == "prepare-subset":
            sub.add_argument("--profile", choices=("local-smoke", "local-quick"), required=True)
    return command


def main() -> None:
    args = parser().parse_args()
    archive = resolve_path(args.archive)
    derived_root = resolve_path(args.derived_root)
    manifest_dir = resolve_path(args.manifest_dir)
    protocol = resolve_path(args.protocol)
    try:
        if args.command == "check-protocol":
            config = verified_formal_config(protocol=protocol, manifest_dir=manifest_dir)
            print(json.dumps({
                "git": git_provenance(),
                "scientific_protocol": "UNCHANGED",
                "formal_model": config["model"],
                "formal_device": FORMAL_DEVICE,
                "formal_imgsz": config["train"]["imgsz"],
                "formal_epochs": config["train"]["epochs"],
                "formal_batch": config["train"]["batch"],
                "formal_optimizer": config["train"]["optimizer"],
                "training_seed": config["training_seed"],
            }, ensure_ascii=False, indent=2))
        elif args.command == "preflight":
            result = preflight(archive=archive, derived_root=derived_root, manifest_dir=manifest_dir,
                               protocol=protocol, device=args.device, profile_name=args.profile)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print("Local Engineering Preflight: PASS")
            print("Scientific protocol: UNCHANGED")
            print("Formal training: NOT STARTED")
        elif args.command == "prepare-subset":
            result = build_subset(derived_root, manifest_dir, args.profile)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            result = run_engineering(args.command, device=args.device, archive=archive,
                                     derived_root=derived_root, manifest_dir=manifest_dir, protocol=protocol)
            print(json.dumps(result, ensure_ascii=False, indent=2))
    except (FileNotFoundError, FileExistsError, ImportError, OSError, PermissionError, RuntimeError,
            ValueError, KeyError, TypeError) as exc:
        if args.command == "preflight":
            print(f"Local Engineering Preflight: FAIL ({type(exc).__name__}: {exc})")
            print("Scientific protocol: UNCHANGED (verification incomplete)")
            print("Formal training: NOT STARTED")
        raise


if __name__ == "__main__":
    main()
