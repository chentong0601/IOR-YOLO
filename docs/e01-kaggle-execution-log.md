# E01 Kaggle execution log

本日志只记录E01云端preflight和正式执行事实，不替代冻结科研协议。

## 2026-09-28 Stage 3E Raw Content Identity attempt

| Field | Observed value |
|---|---|
| Timestamp UTC | 2026-09-28T04:30:00Z |
| Timestamp Asia/Shanghai | 2026-09-28T12:30:00+08:00 |
| Git HEAD | `f337a4f6f3f0dad8feee5a29b110a8280500d1ad` |
| Execution context | Local macOS Codex session (`Darwin arm64`), not Kaggle Notebook |
| Requested dataset root | `/kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508` |
| Dataset root accessible | No |
| Command result | `FileNotFoundError` before any dataset file was read |
| Structure status | Previously observed Kaggle structure check remains PASS; not rerun here |
| Raw Content Identity | **NOT YET VERIFIED** |
| Derived rebuild | Not run; gated on `VERIFIED` |
| Formal training | Not started |

Executed command:

```bash
/usr/bin/python3 IOR-YOLO/scripts/19_verify_d2_unpacked.py --root /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

Observed terminal result:

```text
FileNotFoundError: unpacked D2 root not found: /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

This result indicates an execution-context mismatch, not a JPEG/JSON size or SHA256 mismatch. Stage 3E remains **NOT PASSED**. The next attempt must run inside the actual Kaggle Notebook with the dataset mounted; only a returned `status: VERIFIED` permits derived rebuild and later preflight steps.

## Real Kaggle verification reported before Stage 3E-1

The user subsequently supplied results from the actual Kaggle Notebook. The exact execution timestamp was not supplied, so none is invented here.

| Gate | Verified result |
|---|---|
| Git commit | `f337a4f6f3f0dad8feee5a29b110a8280500d1ad` |
| Bundle transfer | VERIFIED |
| Raw Content Identity | VERIFIED |
| Identity scope | 2812 JPEG + 6 JSON, exact bytes and SHA256 |
| Frozen derived split | train 769 / val 165 / test 165, total 1099 |
| Environment | Python3.12.13; torch2.10.0+cu128; torchvision0.25.0+cu128; Ultralytics8.3.220; NumPy2.0.2; opencv-python4.13.0.92; CUDA12.8 |
| GPU | Tesla T4 ×2 visible; formal device `cuda:0` only |
| Formal training | NOT STARTED |

The next runner failure attempted to open the repository-local ZIP rather than the verified Kaggle unpacked source. It is recorded as **Kaggle preflight runner source-resolution defect**, not Raw Identity failure. Stage 3E-1 adds the corrected source resolver and the one-command preflight; its weight and CUDA-smoke results remain pending a new committed Kaggle run.

## Local-first E01 engineering validation (integrated)

- Date: 2026-10-01
- Status: engineering workflow and code integrated onto the Stage 3E-1 line; no CUDA/Kaggle execution started from this entry.
- Scientific protocol changes: **NONE**.
- Formal E01 training: **NOT STARTED**.
- Formal Val: **NOT STARTED**.
- Final Test: **NOT ACCESSED**.
- Stage 4: **NOT STARTED**.

This entry records workflow integration, not an experiment. No Kaggle runtime, GPU job, smoke training, formal training or evaluation result is claimed here.

Local engineering validation runs through a separate, isolated entry point: `IOR-YOLO/scripts/21_e01_local_engineering.py` with `IOR-YOLO/configs/development/e01_local_engineering.yaml`. It offers only `local-smoke` and `local-quick`, samples a deterministic Train/Val subset (4/2 and 24/12 images), never exposes a Test split, writes to the engineering output namespace, and is labelled **ENGINEERING VALIDATION ONLY - NOT FOR PAPER**. It cannot initialize or be initialized by the formal run.

The intended future Kaggle sequence is:

1. Check the pinned Python/CUDA/PyTorch/torchvision/Ultralytics environment.
2. Rebuild and validate the frozen derived dataset from the raw D2 source and frozen manifests using explicit `/kaggle/...` paths.
3. Run the one-command preflight `20_kaggle_e01_preflight.py`, which stops at `READY FOR HUMAN CONFIRMATION`.
4. Optionally run one small engineering `local-quick --device cuda:0` Train/Val/prediction/analysis cycle, keeping every output under the engineering-only location.
5. Stop for user review and authorization before the separate formal training command.

Exact commands and path variables are maintained in [`kaggle-e01-training-setup.md`](kaggle-e01-training-setup.md) and [`e01-local-development.md`](e01-local-development.md). This log must only be extended with observed command, environment, data, output and error details after an actual user-visible run. Do not insert planned results as observed results.
