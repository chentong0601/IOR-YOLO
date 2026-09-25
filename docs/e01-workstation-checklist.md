# E01 Windows RTX 3070 fallback checklist

本清单仅用于备用Windows路径；当前正式执行平台是Kaggle。

## Before Training

- [ ] Checkout/pull the reviewed E01 commit; record `git rev-parse HEAD`.
- [ ] Confirm all E01 scripts, configs, requirements and frozen manifests are committed. Record any known unrelated local changes.
- [ ] Create Python 3.11 environment; install PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121 and Ultralytics 8.3.220.
- [ ] Run `nvidia-smi` and `14_check_training_environment.py --require-cuda`.
- [ ] Place `dataset-20260508.zip` in `IOR-YOLO/data/raw/multistage_apple_v4/`; verify SHA256 `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`.
- [ ] Rebuild the derived dataset on Windows; do not trust a copied Mac processed directory.
- [ ] Run derived dataset validation and E01 CUDA preflight.
- [ ] Resolve the E01 config and verify official `yolo11n-seg.pt`; retain filename, source/version, SHA256 and file timestamp.
- [ ] Run disposable CUDA smoke with batch 8. Only on explicit CUDA OOM, record evidence and rerun smoke/formal training with batch 4.
- [ ] User reviews all preflight output and explicitly decides to start formal training.

## Training

- [ ] Start visibly in VS Code Terminal: `python IOR-YOLO/scripts/15_e01_run.py train`.
- [ ] Do not change batch, optimizer, augmentation, seed or epoch budget after formal training starts.
- [ ] Confirm `best.pt`, `last.pt`, `results.csv`, `args.yaml`, `run_manifest.yaml`, `resolved_train_config.yaml` and standard plots exist.

## After Training

- [ ] Fix `best.pt` as the evaluation checkpoint and verify its SHA256.
- [ ] Check run manifest, Git commit, environment, pretrained-weight hash and resolved runtime configuration.

## Validation

- [ ] Run formal Val with `best.pt`.
- [ ] Save box, mask and per-class metrics.
- [ ] Export Val predictions; generate matched-instance, confusion and failure-case analysis.

## Final Test

- [ ] Confirm training complete, best checkpoint fixed, Val complete, config unchanged and analysis protocol fixed.
- [ ] Run exactly one final Test with explicit `--final-test`.
- [ ] Do not use Test to modify model, augmentation, optimizer, epoch budget or checkpoint selection.
- [ ] Export and analyze final Test only after the locked evaluation is recorded.

## Archive

- [ ] Preserve original run outputs; do not overwrite poor or failed results.
- [ ] Verify provenance and analysis artifacts are complete.
- [ ] Mark `E01 = COMPLETED` only after training, fixed best checkpoint, Val, final Test, complete manifest, verified provenance and analysis all pass.
