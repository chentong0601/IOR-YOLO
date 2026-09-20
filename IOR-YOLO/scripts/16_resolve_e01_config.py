"""Resolve E01 against the installed Ultralytics 8.3.220 defaults, without training.

Auto optimizer decisions are projected for inspection, then replaced with the
trainer's observed optimizer/args in the run-local file after formal training.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/e01_yolo11n_seg.yaml"
FIELDS = ("epochs", "batch", "imgsz", "optimizer", "lr0", "lrf", "momentum", "weight_decay",
          "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "patience", "cos_lr",
          "close_mosaic", "amp", "deterministic", "workers", "nbs", "hsv_h", "hsv_s",
          "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
          "fliplr", "mosaic", "mixup", "copy_paste", "copy_paste_mode", "cache", "seed")
HARDWARE = {"batch", "workers"}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(config_path: Path = CONFIG, *, batch_override: int | None = None) -> dict:
    import ultralytics
    from ultralytics.cfg import DEFAULT_CFG, DEFAULT_CFG_PATH, get_cfg

    if ultralytics.__version__ != "8.3.220":
        raise ValueError(f"wrong Ultralytics version: {ultralytics.__version__}")
    requested = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train = {key: value for key, value in requested["train"].items() if key != "augmentation"}
    train.update(requested["train"]["augmentation"])
    if batch_override is not None:
        if (train["batch"], batch_override) != (8, 4):
            raise ValueError("only predeclared hardware OOM fallback 8 -> 4 is allowed")
        train["batch"] = 4
    if requested["training_seed"] != train["seed"] or requested["model"] != "yolo11n-seg.pt":
        raise ValueError("E01 seed or model changed unexpectedly")
    parsed = get_cfg(DEFAULT_CFG, train)
    defaults = {key: getattr(DEFAULT_CFG, key) for key in FIELDS}
    effective = {key: getattr(parsed, key) for key in FIELDS}
    sources = {key: ("hardware-specific override" if key in HARDWARE and key in train and train[key] != defaults[key]
                     else "E01 explicit override" if key in train and train[key] != defaults[key]
                     else "8.3.220 default (explicitly repeated)" if key in train
                     else "8.3.220 default") for key in FIELDS}
    if effective["optimizer"] == "auto":
        # The pinned BaseTrainer._setup_train computes iterations from dataset
        # length, nbs and epochs. This is a *projection*, not a trained result.
        projected_iterations = math.ceil(769 / max(effective["batch"], effective["nbs"])) * effective["epochs"]
        projected_optimizer = "SGD" if projected_iterations > 10000 else "AdamW"
        projected_lr0 = 0.01 if projected_optimizer == "SGD" else round(0.002 * 5 / (4 + 3), 6)
        automatic = {"status": "projected_only; verify from trainer at runtime",
                     "formula": "ceil(train_images / max(batch, nbs)) * epochs; threshold > 10000",
                     "assumed_images": 769, "assumed_classes": 3,
                     "projected_iterations": projected_iterations,
                     "projected_optimizer": projected_optimizer,
                     "projected_lr0": projected_lr0, "projected_momentum": 0.9,
                     "effective_warmup_bias_lr": 0.0,
                     "warning": "optimizer=auto ignores requested lr0/momentum; runtime optimizer is authoritative"}
    else:
        automatic = {"status": "not used"}
    return {"schema_version": "e01-resolved-v1", "status": "pretrain_requested_and_projected",
            "experiment_id": requested["experiment_id"], "model": requested["model"],
            "ultralytics_version": ultralytics.__version__,
            "ultralytics_default_yaml_sha256": file_sha(Path(DEFAULT_CFG_PATH)),
            "e01_requested_config_sha256": file_sha(config_path),
            "dataset_protocol_version": requested["dataset_protocol_version"],
            "frozen_pool_sha256": requested["experiment_pool_sha256"],
            "frozen_split_sha256": requested["dataset_split_sha256"],
            "training_seed": requested["training_seed"],
            "defaults": defaults, "requested": train, "resolved_pretrain": effective,
            "parameter_source": sources, "optimizer_auto_projection": automatic,
            "hardware_batch_fallback": "8 -> 4 after pretraining CUDA OOM" if batch_override == 4 else None,
            "actual_runtime": None}


def save(path: Path, resolved: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True),
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, help="optional local, Git-ignored preflight YAML path")
    args = parser.parse_args()
    resolved = resolve(args.config)
    if args.output:
        save(args.output, resolved)
        print(args.output)
    else:
        print(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
