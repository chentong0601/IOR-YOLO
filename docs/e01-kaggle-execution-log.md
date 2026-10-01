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
