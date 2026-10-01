# E01 Local-First Development

Updated: 2026-10-01. This workflow is for software engineering validation only.
It does not create paper results, choose model settings, evaluate final Test, or
advance the study to Stage 4.

## Separation of execution paths

- Formal scientific settings stay in
  [`e01_yolo11n_seg.yaml`](../IOR-YOLO/configs/experiments/e01_yolo11n_seg.yaml):
  YOLO11n-seg instance segmentation, official pretrained initialization,
  `imgsz=640`, at most 100 epochs, batch 8, `optimizer=auto`, seed 0,
  frozen D2 split and the existing augmentations/metrics.
- Local overrides stay in
  [`e01_local_engineering.yaml`](../IOR-YOLO/configs/development/e01_local_engineering.yaml).
  The separate [`19_e01_local_engineering.py`](../IOR-YOLO/scripts/19_e01_local_engineering.py)
  runner accepts only `local-smoke` and `local-quick`, stores outputs under
  `IOR-YOLO/runs/e01_engineering/`, and never offers a Test split.
- The formal runner
  [`15_e01_run.py`](../IOR-YOLO/scripts/15_e01_run.py) remains a separate entry
  point. Formal train/Val uses `cuda:0`, and formal initialization is fixed to
  the official `yolo11n-seg.pt`; there is no checkpoint-path argument.

The engineering subset is selected deterministically from frozen Train/Val IDs
only: 4/2 images for smoke and 24/12 for quick validation. Its manifest records
the exact image IDs and file hashes. Its dataset YAML has only `train` and `val`;
the final Test files are not loaded for subset construction, model input,
validation, prediction, or analysis. The frozen 769/165/165 split and its
artifacts are never rewritten.

Local runs reuse the formal task, class map, `optimizer=auto`, seed, and online
augmentation. Only the engineering profile changes runtime scale
(`imgsz`, epoch budget, batch, workers, CPU AMP setting and deterministic subset).
These values are explicitly marked non-scientific and are not copied to the
formal configuration.

## macOS VS Code workflow

Open the repository root in VS Code. Select a Python 3.11/3.12 environment in
which PyTorch, torchvision, Ultralytics 8.3.220, Pillow, PyYAML and matplotlib
are installed. Install a macOS-compatible PyTorch/torchvision build for CPU or
MPS; do not install CUDA on the Mac.

Place the exact raw D2 archive at the default local path
`IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip`, or pass an explicit
path to the script. From the VS Code terminal at the repository root:

```bash
python IOR-YOLO/scripts/14_check_training_environment.py
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py build
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py validate
python IOR-YOLO/scripts/19_e01_local_engineering.py check-protocol
python IOR-YOLO/scripts/19_e01_local_engineering.py preflight --device cpu --profile local-quick
```

The builder reconstructs the complete frozen derived dataset as an artifact;
it does not perform model evaluation. The engineering preflight validates raw
ZIP provenance and Train/Val derived files, confirms dependencies/device,
loads the official model, and builds/verifies the deterministic Train/Val-only
subset. It prints the required PASS/FAIL, unchanged-protocol and formal-not-
started lines. It does not run training.

Run the true CPU training smoke first:

```bash
python IOR-YOLO/scripts/19_e01_local_engineering.py local-smoke --device cpu
```

This performs one training epoch on four Train images at 64px, batch 1 and zero
workers. It must complete training/backward-update and write `last.pt`,
`results.csv`, and `args.yaml`. There is no validation or Test access in this
mode.

Then verify the full engineering Train → checkpoint → Val → prediction export →
analysis path:

```bash
python IOR-YOLO/scripts/19_e01_local_engineering.py local-quick --device cpu
```

This uses 24 Train / 12 Val images, one epoch, 128px and batch 2. Expected
outputs are under a unique `runs/e01_engineering/local-quick_*` folder:
`weights/best.pt`, `weights/last.pt`, `results.csv`, `args.yaml`,
`run_manifest.yaml`, `validation/`, `predictions/predictions_val.json`, and
`analysis/engineering_validation.json` plus diagnostic curves. The run manifest
labels every output `ENGINEERING VALIDATION ONLY - NOT FOR PAPER`. Losses and
metrics are not scientific evidence.

To test MPS explicitly, substitute `--device mps`; the command fails rather
than silently falling back if MPS is unavailable. `--device cuda:0` is accepted
for compatible CUDA development machines and cloud quick checks. Multi-GPU
device strings are rejected.

## VS Code tasks

Use **Terminal → Run Task**:

- `E01: Run Low-Cost Tests`
- `E01: Build Derived Dataset`
- `E01: CPU Smoke`
- `E01: CPU Quick Validation`
- `E01: Check Git/Protocol`
- `E01: Prepare Cloud Preflight`

The task definitions live in [tasks.json](../.vscode/tasks.json). Engineering
outputs and derived data are Git-ignored; source/config/tests and this document
remain reviewable.

## Optional Windows CPU compatibility check

On the Lenovo laptop, install a supported Python (3.11 or 3.12), CPU PyTorch
and torchvision, then the project requirements. Keep the same raw archive and
frozen manifests. In PowerShell from the repository root, use explicit Windows
paths as needed:

```powershell
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py build --archive "D:\datasets\dataset-20260508.zip"
python IOR-YOLO/scripts/19_e01_local_engineering.py local-smoke --device cpu --archive "D:\datasets\dataset-20260508.zip"
python IOR-YOLO/scripts/19_e01_local_engineering.py local-quick --device cpu --archive "D:\datasets\dataset-20260508.zip"
```

This is only a path/CLI/dataloader/checkpoint compatibility check. Do not
attempt 100-epoch training on this CPU machine.

## Interpretation and safety

Local CPU/MPS/CUDA output must not be used for model comparison, tuning, paper
tables, or conclusions. Local checkpoints are written outside the formal run
directory; the formal runner constructs a fresh model from the official
pretrained filename and cannot accept an engineering checkpoint. Do not pass
local checkpoint paths into any formal command.

Stage 3E-2 validation does not modify the D2 protocol/split, run formal training,
or unlock final Test. Kaggle/cloud formal steps are in
[`kaggle-e01-training-setup.md`](kaggle-e01-training-setup.md).
