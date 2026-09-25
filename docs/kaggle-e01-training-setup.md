# E01 Kaggle pre-training and execution handoff

Kaggle Notebook是当前E01正式执行平台。Google Colab与Windows RTX 3070仅作备用。平台迁移不改变D2冻结协议、冻结划分或E01科研参数。

## 三层控制

1. **Scientific experiment protocol**：D2冻结数据与70/15/15 group-aware split、YOLO11n-seg、instance segmentation、`imgsz=640`、最多100 epochs、batch8、`optimizer=auto`、seed0、既有augmentation、Val/Test lock及指标定义。这些是受控科研变量，本次全部不变。
2. **Execution environment / provenance**：Kaggle、Python、torch、torchvision、Ultralytics、NumPy、OpenCV、CUDA、实际GPU、可见GPU数、选定设备、Git提交、权重hash和数据证据hash。正式run必须记录实际观测值。
3. **Compatibility requirements**：代码要求Python 3.11或3.12、CUDA可用、torch/torchvision能共同导入、Ultralytics固定8.3.220。torch、torchvision和CUDA的具体版本属于运行provenance，不是科研变量。

## 已人工验证的候选正式环境

| 项目 | 实际值 |
|---|---|
| Platform | Kaggle Notebook |
| Python | 3.12.13 (`/usr/bin/python3`) |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| Ultralytics | 8.3.220 |
| NumPy | 2.0.2 |
| OpenCV distribution | opencv-python 4.13.0.92 |
| CUDA available/runtime | True / 12.8 |
| GPU | NVIDIA Tesla T4 |
| Visible GPUs | 2 |
| Formal device | **`cuda:0` only** |

以上环境已通过import、CUDA availability和`yolo11n-seg.pt`加载。不要为了匹配旧Windows计划而降级torch/torchvision/CUDA。两个T4可见不授权多卡：E01固定`device=0`，不得自动启用DDP或`device=0,1`。若正式运行前环境与上表不一致，停止并重新审查，不静默换环境。

## Kaggle D2输入与身份门

实际只读输入根目录：

```text
/kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

Kaggle已自动解包上传的`dataset-20260508.zip`。当前1041/126/239张original、对应1406张resize和六份JSON仅证明：

```text
D2 STRUCTURE CHECK = PASS
RAW CONTENT IDENTITY = NOT YET VERIFIED
```

原ZIP本身不在挂载目录中，因此不能直接重算固定ZIP SHA256，也不得重新打包目录后比较新ZIP hash。新的ZIP会因元数据、时间、压缩和顺序不同而产生不同字节。

项目的内容身份门使用两类已跟踪证据：

- `files_sha256.csv`中的全部2812张JPEG逐文件字节数与SHA256；
- `d2_unpacked_identity.json`中从固定原ZIP只读提取的六份JSON字节数与SHA256。

只有下列命令实际返回`status: VERIFIED`，才可把Kaggle解包内容视为固定D2来源的内容等价表示：

```bash
/usr/bin/python3 IOR-YOLO/scripts/19_verify_d2_unpacked.py --root /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

这项验证检查全部图像和annotation JSON的字节，不依赖文件数量、文件名或目录结构单独作结论，也不修改冻结manifest。

## Preflight顺序

正式训练前按顺序执行；任何一步失败即停止。

### 1. Git与运行路径

正式Notebook必须来自已审核且已提交的Git HEAD。记录：

```bash
git status --short
git rev-parse HEAD
```

在Notebook Python cell设置只读raw来源和可写输出：

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["E01_RAW_SOURCE"] = "/kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508"
os.environ["E01_RUNS_ROOT"] = "/kaggle/working/ior-yolo-output/runs"
```

`/kaggle/input`只读；derived dataset、日志、checkpoint和分析输出必须写入`/kaggle/working`。

### 2. 已验证环境复核

不安装、不卸载、不降级包。先核验当前环境和依赖一致性：

```bash
/usr/bin/python3 -m pip check
/usr/bin/python3 IOR-YOLO/scripts/14_check_training_environment.py --require-cuda --platform kaggle
```

环境脚本报告Python executable、torch、torchvision、Ultralytics、NumPy、OpenCV、CUDA runtime、GPU inventory、visible GPU count和选定`cuda:0`。`requirements-e01.txt`保留Ultralytics行为pin，torch/torchvision使用实际运行provenance，不触发无依据的降级。

### 3. Raw content identity

```bash
/usr/bin/python3 IOR-YOLO/scripts/19_verify_d2_unpacked.py --root /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

必须得到`VERIFIED`。结构数量通过而任一图像或JSON hash不一致时，Raw Identity仍失败，禁止继续。

### 4. Derived dataset重建与验证

不得复制Mac或其他未受控processed目录。直接从已验证的只读解包目录和冻结项目产物重建：

```bash
/usr/bin/python3 IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py build --source /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
/usr/bin/python3 IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py validate --source /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

结果必须复现正式池1099张和冻结train/val/test 769/165/165；provider原始1041/126/239目录不是正式E01 split。

### 5. Config、权重和CUDA smoke

```bash
/usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py preflight --require-cuda --platform kaggle
/usr/bin/python3 IOR-YOLO/scripts/16_resolve_e01_config.py
/usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py verify-weights --platform kaggle
/usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py smoke --platform kaggle
```

官方`yolo11n-seg.pt`只作初始化；记录来源和SHA256，失败时停止，不回退随机初始化。smoke默认batch8；只有真实CUDA OOM时，才允许保留证据并执行既定8→4 fallback：

```bash
/usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py smoke --platform kaggle --batch 4 --oom-note "Pre-training Kaggle CUDA smoke produced OOM at batch 8"
```

batch变化只处理硬件可行性，不能依据性能选择，正式训练启动后不得改变。

### 6. User gate与正式命令

用户须亲自检查Git、环境、Raw Identity、derived统计、resolved config、权重hash和smoke输出。脚本不会从smoke自动进入训练。全部门禁通过后，正式命令才是：

```bash
/usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py train --platform kaggle
```

本Stage 3D任务不执行该命令。

## 输出持久化与评价顺序

Kaggle session是临时环境。正式输出目录为：

```text
/kaggle/working/ior-yolo-output/runs/e01_yolo11n_seg/seed_0/
```

训练与后续锁定评价完成后，按适用阶段保存并导出：`best.pt`、`last.pt`、`results.csv`、`args.yaml`、`resolved_train_config.yaml`、`run_manifest.yaml`、`pip-freeze.txt`、环境/provenance记录、标准训练图、Val指标、预测导出、分析产物，以及解锁后的一次final Test输出。Notebook结束前必须Save Version并下载或归档完整目录。

训练后固定`best.pt`，先Val、预测导出和分析。只有training完成、best固定、Val完成、配置与分析协议不再改变后，才允许显式`test --final-test`。Test不能指导模型、optimizer、augmentation、epoch、checkpoint或方法设计。

## 停止条件

环境偏离已审核候选、CUDA不可用、Raw Identity未验证、冻结hash或1099/769/165/165统计不一致、E01相关Git路径未提交、官方权重不可得、batch8及允许的batch4均OOM，均须停止并记录。不得通过换数据、换split、换模型、换seed、多卡、随机初始化或调参绕过。
