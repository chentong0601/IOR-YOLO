# E01 Kaggle pre-training and execution handoff

Kaggle Notebook是当前E01正式执行平台。Google Colab与Windows RTX 3070仅作备用。平台迁移不改变D2冻结协议、冻结划分或E01科研参数。

执行尝试及门禁结果记录在[E01 Kaggle execution log](e01-kaggle-execution-log.md)。真实Kaggle会话已经确认2812张JPEG、6份JSON及冻结769/165/165派生划分；之后暴露的问题是旧runner错误回退到本地ZIP，已在Stage 3E-1修复为统一source resolution。正式训练仍未开始。

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

Kaggle已自动解包上传的`dataset-20260508.zip`。目录数量最初只证明Structure Check；后续真实Kaggle逐字节门禁已经证明：

```text
D2 STRUCTURE CHECK = PASS
RAW CONTENT IDENTITY = VERIFIED
```

原ZIP本身不在挂载目录中，因此不能直接重算固定ZIP SHA256，也不得重新打包目录后比较新ZIP hash。新的ZIP会因元数据、时间、压缩和顺序不同而产生不同字节。

项目的内容身份门使用两类已跟踪证据：

- `files_sha256.csv`中的全部2812张JPEG逐文件字节数与SHA256；
- `d2_unpacked_identity.json`中从固定原ZIP只读提取的六份JSON字节数与SHA256。

真实Kaggle会话已运行该门禁并返回`status: VERIFIED`。每个新临时session仍由单命令preflight重新核验：

```bash
/usr/bin/python3 IOR-YOLO/scripts/19_verify_d2_unpacked.py --root /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
```

这项验证检查全部图像和annotation JSON的字节，不依赖文件数量、文件名或目录结构单独作结论，也不修改冻结manifest。

## Session bootstrap

Git bundle恢复完成后，先进入仓库根目录。Kaggle镜像已经具备已验证torch/torchvision/CUDA时，不要降级或替换它们。如果Ultralytics缺失或不是8.3.220，只执行以下独立bootstrap；`--no-deps`避免它静默修改torch、torchvision或CUDA相关包：

```bash
/usr/bin/python3 -m pip install --no-deps ultralytics==8.3.220
```

若随后环境检查报告其他依赖缺失或版本不兼容，停止并报告，不在科研验证逻辑里隐藏安装。

## One-command preflight

正式训练前通常只运行下面一个可见Kaggle cell；任何门禁失败时脚本非零退出并停止，不会调用正式train、Val或Test：

```bash
/usr/bin/python3 IOR-YOLO/scripts/20_kaggle_e01_preflight.py --source /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508 --platform kaggle --batch 8
```

它按顺序检查Git provenance和E01相关dirty状态、Raw Identity、派生集重建/验证、冻结split与类别计数、实际环境、resolved config、官方预训练权重及SHA256，并在`device=0`运行一次可丢弃的batch8 CUDA smoke。输出和session provenance保存到：

```text
/kaggle/working/ior-yolo-output/preflight/
```

预期最终摘要必须包含`Raw identity: VERIFIED`、`Split: 769/165/165`、`Formal device: cuda:0`、权重SHA、`Batch-8 CUDA smoke: PASS`、`Formal training: NOT STARTED`和`E01 readiness: READY FOR HUMAN CONFIRMATION`。

source解析顺序固定为：显式`--source` > `E01_D2_SOURCE`（兼容旧`E01_RAW_SOURCE`）> 存在的Kaggle正式挂载路径 > 存在的本地审计ZIP；否则明确失败。显式或环境变量选择的路径不存在时不会静默换数据。

## Diagnostic commands

以下命令只用于定位单命令失败，不是常规cell-by-cell工作流：

```bash
/usr/bin/python3 IOR-YOLO/scripts/19_verify_d2_unpacked.py --root /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508
/usr/bin/python3 IOR-YOLO/scripts/14_check_training_environment.py --require-cuda --platform kaggle
/usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py preflight --require-cuda --platform kaggle --source /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508 --derived /kaggle/working/ior-yolo-derived/d2_e01_ultralytics
```

官方`yolo11n-seg.pt`只作初始化；获取失败即停止，不回退随机初始化。batch8只有在真实CUDA OOM时才允许另行保留证据后执行既定8→4 fallback，不能依据性能选择。

## Human gate与正式命令

单命令preflight返回`READY FOR HUMAN CONFIRMATION`后，用户须检查Git、环境、Raw Identity、derived统计、resolved config、权重hash和smoke结果。正式训练仍是单独动作，本Stage 3E-1不执行：

```bash
E01_RUNS_ROOT=/kaggle/working/ior-yolo-output/runs /usr/bin/python3 IOR-YOLO/scripts/15_e01_run.py train --platform kaggle --source /kaggle/input/datasets/tongchen0501/ior-yolo-d2-raw/dataset-20260508 --derived /kaggle/working/ior-yolo-derived/d2_e01_ultralytics
```

## 输出持久化与评价顺序

Kaggle session是临时环境。正式输出目录为：

```text
/kaggle/working/ior-yolo-output/runs/e01_yolo11n_seg/seed_0/
```

训练与后续锁定评价完成后，按适用阶段保存并导出：`best.pt`、`last.pt`、`results.csv`、`args.yaml`、`resolved_train_config.yaml`、`run_manifest.yaml`、`pip-freeze.txt`、环境/provenance记录、标准训练图、Val指标、预测导出、分析产物，以及解锁后的一次final Test输出。Notebook结束前必须Save Version并下载或归档完整目录。

训练后固定`best.pt`，先Val、预测导出和分析。只有training完成、best固定、Val完成、配置与分析协议不再改变后，才允许显式`test --final-test`。Test不能指导模型、optimizer、augmentation、epoch、checkpoint或方法设计。

## 停止条件

环境偏离已审核候选、CUDA不可用、Raw Identity未验证、冻结hash或1099/769/165/165统计不一致、E01相关Git路径未提交、官方权重不可得、batch8及允许的batch4均OOM，均须停止并记录。不得通过换数据、换split、换模型、换seed、多卡、随机初始化或调参绕过。
