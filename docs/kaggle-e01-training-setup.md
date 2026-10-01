# E01 Kaggle / Cloud CUDA Setup

Updated: 2026-10-01. Kaggle/cloud is the CUDA validation and formal execution
environment. This document provides future commands; none were executed as
part of Stage 3E-2.

## Separation and compatibility gate

Formal configuration remains
[`e01_yolo11n_seg.yaml`](../IOR-YOLO/configs/experiments/e01_yolo11n_seg.yaml):
YOLO11n-seg instance segmentation, official `yolo11n-seg.pt` initialization,
640px, up to 100 epochs, batch 8, `optimizer=auto`, seed 0, and the frozen D2
769/165/165 split, augmentation and metrics. The formal runner always uses
`cuda:0`. Engineering quick validation is a distinct profile and cannot
initialize the formal run.

The frozen software pins remain Python 3.11, PyTorch 2.5.1 CUDA 12.1,
torchvision 0.20.1 CUDA 12.1, and Ultralytics 8.3.220. Kaggle images can change
their preinstalled stack. The preflight is intentionally strict: if the
selected image cannot run these pins on CUDA 0, stop and resolve the runtime
compatibility before any training. Do not silently alter versions or the
scientific configuration.

## One-time Kaggle setup

Clone or open the reviewed, committed repository revision in the Kaggle
environment. Attach the original D2 archive as a Kaggle input. Set these
variables in a notebook cell or adapt the same explicit paths in a terminal:

```bash
ZIP=/kaggle/input/multistage-apple-v4/dataset-20260508.zip
DERIVED=/kaggle/working/d2_e01_ultralytics
```

The ZIP basename must remain `dataset-20260508.zip`; the path itself is
environment-specific. Use the Kaggle-provided Python 3.11 environment only if
it passes the pinned environment checks. If installing the pinned wheel stack
is required and supported by the active CUDA driver:

```bash
python -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r IOR-YOLO/requirements-e01.txt
python -m pip check
```

Do not add Kaggle mount paths to source/config files. Supply paths on the
command line. The default local paths continue to work on macOS and Windows.

## Preflight and engineering-only CUDA validation

First verify the runtime, build the derived data from the immutable archive and
frozen manifests, and validate the conversion:

```bash
python IOR-YOLO/scripts/14_check_training_environment.py --require-cuda
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py build --archive "$ZIP" --output "$DERIVED"
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py validate --archive "$ZIP" --output "$DERIVED"
python IOR-YOLO/scripts/15_e01_run.py preflight --require-cuda --archive "$ZIP" --derived-root "$DERIVED"
```

Then run the single local engineering preflight and one small CUDA
Train→Val→analysis check. These are marked `ENGINEERING VALIDATION ONLY - NOT
FOR PAPER`; they do not produce formal E01 metrics:

```bash
python IOR-YOLO/scripts/19_e01_local_engineering.py preflight --device cuda:0 --profile local-quick --archive "$ZIP" --derived-root "$DERIVED"
python IOR-YOLO/scripts/19_e01_local_engineering.py local-quick --device cuda:0 --archive "$ZIP" --derived-root "$DERIVED"
```

Optionally run the formal-configuration tiny CUDA batch feasibility check.
This does not create a formal run and never transitions automatically to
training:

```bash
python IOR-YOLO/scripts/15_e01_run.py smoke --archive "$ZIP" --derived-root "$DERIVED"
```

If it reports batch-8 CUDA OOM before formal training, preserve the traceback
and follow the existing one-time hardware fallback rule (batch 4 with the
documented OOM note). Do not performance-tune batch size.

## Future formal E01 launch

Only after all preflight checks and engineering compatibility checks pass,
ensure the committed code has a clean E01-relevant Git status and ask the user
to explicitly authorize the visible formal run. Verify the official weight,
then launch exactly once:

```bash
python IOR-YOLO/scripts/15_e01_run.py verify-weights --archive "$ZIP" --derived-root "$DERIVED"
python IOR-YOLO/scripts/15_e01_run.py train --archive "$ZIP" --derived-root "$DERIVED"
```

The train command restarts from the official pretrained `yolo11n-seg.pt`; it
does not load local/CUDA engineering checkpoints. It writes into the ignored
formal run directory and captures run provenance. If the formal run finishes,
fix the best checkpoint and do Val first:

```bash
python IOR-YOLO/scripts/15_e01_run.py val --archive "$ZIP" --derived-root "$DERIVED"
python IOR-YOLO/scripts/18_export_e01_predictions.py --split val --data-root "$DERIVED"
python IOR-YOLO/scripts/17_analyze_e01_results.py --split val
```

Final Test is not part of this setup or engineering validation. Preserve the
existing explicit `--final-test` lock and execute it only after the formal
protocol's model-selection/Val completion gate and separate user
authorization. Download/persist the run directory before the Kaggle session
ends.

## Portability notes

- `--device cuda:0` is the only formal GPU device; multi-GPU selection is not
  enabled.
- `--archive` and `--derived-root` are explicit so Kaggle paths are not
  hard-coded into the Mac workflow.
- Formal Python/PyTorch/torchvision/Ultralytics pins, initialization,
  optimizer, augmentation, split, batch, image size, epoch budget and metrics
  are not overridden by the local engineering profile.
- A Kaggle CUDA result is formal only when it comes from the formal `train`
  command and its fixed protocol/provenance, not from `local-quick` or `smoke`.
