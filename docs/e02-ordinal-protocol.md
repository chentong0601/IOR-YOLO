# E02 有序成熟度损失（ordinal-aware）实验协议

更新：2026-10-03。状态：**工程实现与审计完成；FORMAL TRAINING = NOT STARTED**。本轮不训练 100 epoch、不访问 Final Test，也不改动 E01 的任何正式代码、配置、manifest 或结果。E01 冻结证据见[E01 基线协议](e01-baseline-protocol.md)、[E01 结果分析协议](e01-analysis-protocol.md)与[研究计划与决策记录](../研究计划与决策记录.md)。

## 1. 科学假设与受控边界

E01 已完成 YOLO11n-seg 正式训练、Formal Validation、预测导出、corrected error analysis 与人工失败案例分析。冻结结果（read-only 参考，来自 E01 analysis commit `c22134cd`）：

| 项目 | E01 冻结值 |
|---|---|
| Box mAP50 / mAP50-95 | 0.973512 / 0.945334 |
| Mask mAP50 / mAP50-95 | 0.973512 / 0.941337 |
| operational conf | 0.65（Validation F1 工作点，Test 前冻结） |
| GT / matched / missed / FP | 323 / 314 / 9 / 9（GT coverage 97.21%） |
| mature / semi-mature / immature 命中率 | 97.39% / 83.33% / 95.68% |
| maturity correct / adjacent / severe | 295 / 19 / 0，MASE ≈ 0.0605 |

人工复核表明：detection/segmentation 已较高；FN 主要来自遮挡、小目标、重叠；部分 FP 可能来自复杂背景或漏标；成熟度误判实例通常 Mask IoU 较高（成熟度错误并非主要由分割失败导致）；**semi-mature 是主要瓶颈**；19/19 成熟度错误全部是 adjacent-stage（immature ↔ mature severe error = 0）。

由此形成 E02 假设：

> 传统三分类把 immature / semi-mature / mature 当作无序类别，未显式利用其自然序关系。加入 ordinal-aware 辅助损失（λ=0.5）可能改善过渡阶段（尤其 semi-mature）的判别，同时保持 detection 与 segmentation 性能。

E02 是**最小化、可解释的 hypothesis test**：dataset、冻结 train/val/test 划分、YOLO11n-seg、官方预训练权重、imgsz=640、epochs=100、batch=8、seed=0、optimizer=auto、单 GPU cuda:0、增强策略、detection head、segmentation head、Final Test 锁定原则全部与 E01 完全一致；`train`/`augmentation` 块由脚本逐键比对，不允许任何差异。不替换模型、不加 attention/Transformer/CBAM、不改分割结构、不使用 Test 调参。

## 2. 方法定义（数学）

类别顺序固定为 `0 = immature apple < 1 = semi-mature apple < 2 = mature apple`（与派生数据集 `dataset.yaml` 的 names 逐项校验）。

对分类 logits `z`（形状 `(N, K)`，K=3），保留原始 Ultralytics 分类损失不变，另加

```
p        = softmax(z)
F_p(j)   = Σ_{k≤j} p_k                     j = 0 … K-2      预测累积概率
F_y(j)   = 1[j ≥ y]                        j = 0 … K-2      目标累积表示
L_ord^(i)= mean_j ( F_p^(i)(j) − F_y^(i)(j) )²
L_ord    = Σ_{i ∈ 匹配正样本锚点} L_ord^(i) / max(target_scores.sum(), 1)
L_total  = L_original_YOLO + λ · L_ord                λ = 0.5（第一轮固定）
```

目标累积表示与冻结编码一致：immature → (1,1)、semi-mature → (0,1)、mature → (0,0)；`configs/experiments/e02_yolo11n_seg_ordinal.yaml` 的 `ordinal.cumulative_targets` 与代码常量逐项断言相等。归一化分母 `target_scores.sum()` 与原分类损失完全相同，因此辅助项与被扩展项保持同一实例尺度。单实例惩罚上界为 1：完全正确 one-hot 为 0，完全自信的相邻错误为 0.5，完全自信的严重（跨两级）错误为 1。

λ 在训练前声明（`lambda_selection: predeclared_before_training`），不自动调参、不依据 Val/Test 修改；代码层面 λ 必须等于 0.5，否则 preflight 直接失败。

## 3. 注入位置与 Ultralytics 侧形式

| 问题 | 记录 |
|---|---|
| 原始分类损失形式 | ultralytics 8.3.220 `v8SegmentationLoss.__call__`：`loss[2] = BCEWithLogitsLoss(reduction="none")(pred_scores, target_scores).sum() / target_scores_sum`，`target_scores` 为 task-aligned assigner 目标（匹配锚点 one-hot、其余为 0），随后 `×hyp.cls`，整体以 `loss * batch_size` 返回，item 向量为 `(box, seg, cls, dfl)` |
| 注入位置 | `IOR-YOLO/ior_yolo/trainers/ordinal.py::OrdinalSegmentationLoss.__call__`（子类，先调用 `super().__call__` 得到**未改动**的原损失，再加辅助项） |
| 合并方式 | `L_total = 原损失总和 + λ·L_ord·batch_size`（与原实现 `loss * batch_size` 同尺度）；item 向量追加第 5 项 `λ·L_ord`，因此 `results.csv` 与 TensorBoard 出现 `train/ordinal_loss`、`val/ordinal_loss`，其余 4 项数值不变 |
| 目标来源 | `AssignerCallRecorder` 代理记录 pinned `TaskAlignedAssigner` 的**最近一次**调用入参/输出，直接复用原分类损失所比对的同一批匹配锚点与阶段标签；不重跑 assigner、不复制目标构建逻辑、用后立即释放（不跨 step 保留张量、不进入 checkpoint） |
| 正确性自检 | 从头部原始输出重建的 logits 与原损失交给 assigner 的 `sigmoid` 张量逐元素比对（`atol=1e-4, rtol=1e-3`），不一致即抛错，避免静默优化错误通道；eval/validation 下 Ultralytics 会以 `(detections, (feats, mask_coeffs, proto))` 包装头部输出，重建逻辑按通道数结构化定位 level 列表，找不到即抛错 |
| 侵入程度 | 不复制/不 patch Ultralytics 源码；仅覆盖 `init_criterion`、`get_model`、`get_validator`，并新增 `OrdinalYOLO` 的 `task_map`（只替换 segment 的 model/trainer，validator/predictor 与其它任务保持原样） |

数值细节：辅助项在 float32 下计算（AMP 下更稳），梯度仍回到 fp16 logits；batch 内无匹配锚点时返回与 logits 相连的可微零，保证每步张量形状一致。

## 4. 实验标识与本轮新增文件

实验 ID `e02_yolo11n_seg_ordinal`，run 目录 `runs/e02_yolo11n_seg_ordinal/seed_0`（Git 忽略；preflight 记录在同级 `preflight/`）。

| 文件 | 作用 |
|---|---|
| `configs/experiments/e02_yolo11n_seg_ordinal.yaml` | E02 配置：与 E01 相同的 train/execution 块 + `ordinal`/`tensorboard`/`run`/`e01_reference` 块 |
| `configs/experiments/e02_run_manifest_template.yaml` | E02 manifest 模板（保留 E01 字段名以便对比，新增 ordinal/TensorBoard/E01 只读引用字段） |
| `ior_yolo/losses/ordinal.py` | 纯数学实现与校验（不导入 Ultralytics，可独立测试） |
| `ior_yolo/trainers/ordinal.py` | criterion / model / trainer / `OrdinalYOLO` 与注入记录 |
| `ior_yolo/utils/tensorboard_logger.py` | E02 唯一 TensorBoard writer、tag 契约、round-trip 自检 |
| `scripts/22_e02_ordinal_run.py` | E02 入口：`preflight` / `smoke` / `tensorboard-smoke` / `train` / `val` |
| `requirements-e02.txt` | 继承 `requirements-e01.txt`，追加 `tensorboard>=2.14,<3` |
| `tests/test_e02_ordinal_loss.py`、`test_e02_ordinal_training.py`、`test_e02_tensorboard.py`、`test_e02_preflight.py` | E02 单元与性质测试（数学、注入、TB、preflight/隔离/锁定） |
| `docs/e02-ordinal-protocol.md` | 本文档 |

E02 与 E01 的差别只有：辅助损失、E02 自身的 config/manifest/preflight/runner/测试/文档，以及 `tensorboard` 依赖。`requirements-e01.txt` 未改动；E01 的 config、configs/data、manifests、scripts、tests 一律只读。

## 5. TensorBoard（正式训练硬要求）

E02 使用**单一 canonical writer**：`torch.utils.tensorboard.SummaryWriter`，写入 `<run_dir>/tensorboard/`，绝不写 E01 目录。Ultralytics 内置 TensorBoard 集成在 E02 进程内被显式禁用（`neutralize_builtin_ultralytics_tensorboard()`，在 `BaseTrainer.__init__` 注册集成回调**之前**执行，不修改任何 Ultralytics 设置文件，也不影响非 E02 路径），原因是该集成无法记录 `ordinal_loss`，重复日志会让 dashboard 与论文溯源产生歧义；禁用结果写入 manifest 的 `tensorboard.builtin_ultralytics_integration`。

| 分组 | tag |
|---|---|
| Training | `train/box_loss`、`train/seg_loss`、`train/cls_loss`、`train/dfl_loss`、`train/ordinal_loss`、`train/total_loss` |
| Validation | `val/box_loss`、`val/seg_loss`、`val/cls_loss`、`val/dfl_loss`、`val/ordinal_loss` |
| Metrics | `metrics/precision(B)`、`metrics/recall(B)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`、`metrics/precision(M)`、`metrics/recall(M)`、`metrics/mAP50(M)`、`metrics/mAP50-95(M)` |
| Optimization | `epoch`、`lr/pg0…`、`perf/gpu_memory_gb`、`perf/epoch_time_seconds`、`perf/elapsed_seconds`、`perf/iteration` |

训练结束后入口脚本执行 `verify_training_artifacts()`：校验 `results.csv` 同时含 `train/ordinal_loss`、`val/ordinal_loss` 及 B/M 指标列、ordinal 损失有限且非负，并用 EventAccumulator 读回 `<run_dir>/tensorboard` 校验全部必需 tag。任一缺失即报错——E02 不允许“无 dashboard 的静默训练”。

Kaggle/Jupyter 启动方式（固定、不做仓库测试依赖）：

```python
%load_ext tensorboard
%tensorboard --logdir /kaggle/working/ior-yolo-output/runs/e02_yolo11n_seg_ordinal
```

本地/Windows 等价命令：`tensorboard --logdir IOR-YOLO/runs/e02_yolo11n_seg_ordinal`。

## 6. 命令

```powershell
# 0) 依赖（E01 pin 不变，仅追加 tensorboard）
python -m pip install -r IOR-YOLO/requirements-e02.txt

# 1) preflight（Kaggle 正式链；--require-cuda 强制 cuda:0 与平台标签）
python IOR-YOLO/scripts/22_e02_ordinal_run.py preflight --require-cuda --platform kaggle --source <D2源>

# 2) TensorBoard 独立 smoke（不训练、不用数据、不占 GPU）
python IOR-YOLO/scripts/22_e02_ordinal_run.py tensorboard-smoke --output-dir IOR-YOLO/runs/e02_engineering/tensorboard_smoke

# 3) disposable smoke：正式设备（完整 D2 身份链 + CUDA）
python IOR-YOLO/scripts/22_e02_ordinal_run.py smoke --device cuda:0 --platform kaggle --source <D2源>
#    本地工程冒烟（CPU/MPS；不重验原始 ZIP，仅按冻结 split manifest 校验派生训练/验证集）
python IOR-YOLO/scripts/22_e02_ordinal_run.py smoke --device cpu --output-dir IOR-YOLO/runs/e02_engineering/local-smoke

# 4) 正式训练（由用户在可见终端启动；不自动执行）
python IOR-YOLO/scripts/22_e02_ordinal_run.py train --platform kaggle --source <D2源>

# 5) 独立 Validation（最终 Test 仍锁定在 E01 协议；E02 入口无 test 命令）
python IOR-YOLO/scripts/22_e02_ordinal_run.py val --platform kaggle --source <D2源>
```

`train` 会把预训练 resolved 配置写为 `runs/e02_yolo11n_seg_ordinal/seed_0_resolved_train_config.yaml`，训练后写入 run 目录内的 `resolved_train_config.yaml`、`run_manifest.yaml`、`pip-freeze.txt`、`metrics/results.csv`、`figures/`，并记录实际观测到的 optimizer/LR、ordinal 诊断与 TensorBoard 溯源。已存在的 run 目录或快照一律拒绝覆盖。

## 7. E01 隔离与 Test 锁定证明

| 证明项 | 实现与验证方式 |
|---|---|
| E01 只读 | config `e01_reference.files_sha256` 固定 E01 config/manifest 模板/`15`、`16`、`17`、`18` 脚本/冻结 protocol/pool/split 的 SHA256；每次 preflight/smoke/train/val 先比对（`assert_e01_immutable`），任一缺失或不一致即抛错并给出文件清单 |
| 训练配方不可变 | `assert_train_parity(config["train"], e01["train"])` 逐键比对，`execution_environment` 也要求与 E01 完全一致；不一致直接失败 |
| 运行目录隔离 | `assert_run_isolation()`：E02 run 目录必须为 `e02_yolo11n_seg_ordinal/seed_0`，不得与 E01 run 目录相等/包含/被包含，路径中不得出现 `e01*` 段 |
| 不覆盖 | run 目录、`seed_0_resolved_train_config.yaml`、`configs/experiments/` 之外的 preflight 记录均拒绝覆盖；preflight 记录不静默重写 |
| 类序不变量 | `verify_derived_classes()` 断言派生数据集 names 与冻结顺序 `immature/semi-mature/mature` 完全一致，并写入 preflight 记录 |
| Test 锁定 | E02 入口命令集合不含 `test`；`evaluate()` 无 split/final-test 参数且只评 `val`；`--final-test` 字样不在 E02 脚本中；manifest 的 `test_metrics_*` 保持 null，`final_test.status = locked_inherited_from_e01` |
| Git 纪律 | `smoke` 要求 E01 相关路径 clean（保护冻结基线），并记录 E02 相关 dirty 状态；`train`/`val` 要求 E01 与 E02 相关路径全部 clean，否则拒绝启动 |

## 8. 评价输出（E02 vs E01）

除标准 Ultralytics Box/Mask 指标外，E02 必须输出与 E01 同构的 operational 分析：matched GT coverage、false positives、missed detections、成熟度混淆矩阵、immature/semi-mature/mature 匹配后分类准确率、adjacent errors、severe errors、MASE、off-by-one accuracy。重点关注：semi-mature 是否改善、adjacent errors 是否减少、severe errors 是否仍接近 0、Box/Mask mAP 是否保持、immature/mature 是否明显退化。

**已知约束（本轮发现）**：`scripts/17_analyze_e01_results.py` 的 provenance 检查把 `experiment_config_sha256` 硬编码指向 `configs/experiments/e01_yolo11n_seg.yaml`，并且 warning/字段措辞均为 E01 专用；`scripts/18_export_e01_predictions.py` 通过它复用同一检查。因此 **E01 的分析与导出脚本不能原样用于 E02 的正式分析**。E02 的 prediction export 与 error analysis 需要在 E02 阶段实现一个薄封装（只读复用 `17` 中与 run 目录无关的纯函数：`match_image`/`maturity_summary`/`failures`/`filter_predictions`/`load_ground_truth`/`read_predictions`，自带 E02 provenance 守卫与 E02 config 哈希），继续沿用同一工作点纪律（conf=0.65、仅 Val、Test 锁定）。本轮**不实现**该封装（无 E02 预测可验证），列为 OPEN QUESTION Q1。

## 9. 本轮已验证证据（工程证据，非实验结果）

| 验证 | 结果 |
|---|---|
| 全量测试 | `python -m unittest discover -s IOR-YOLO/tests -t IOR-YOLO/tests` → 150 tests OK（E02 新增 90 项） |
| TensorBoard smoke（真实 writer） | `tensorboard-smoke` → 25/25 必需 tag 写入并读回，`missing_tags: []`，event file 生成 |
| 本地 CPU 端到端 smoke | 1 epoch / fraction=0.02（15 训练图 + 165 验证图），`criterion_class=OrdinalSegmentationLoss`，`assigner_calls=2`，`train/ordinal_loss=0.282`，`val/ordinal_loss` 列存在，TB 27 个 tag 全部读到，输出留在 `runs/e02_engineering/local-smoke/`（Git 忽略，非论文结果） |
| E01 完整性 | preflight 的 `assert_e01_immutable` 在仓库现状下 `status=verified`（9 个文件） |
| 冒烟修复记录 | 首次 CPU smoke 暴露了 eval/validation 下 head 包装不同（`(detections, (feats, mask_coeffs, proto))`）导致 logits 重建失败；已改为按通道数结构化定位并补回归测试（`test_logits_are_recovered_from_the_eval_mode_wrapping`） |

本地工程环境：Python 3.12(.venv)、torch 2.14.1+cpu、torchvision 0.29.1+cpu、ultralytics 8.3.220、tensorboard 2.21.0（本轮为运行 TB smoke 安装到 `.venv`，并已写入 `requirements-e02.txt`；E01 的 `requirements-e01.txt` 未改动）。本地无 CUDA/MPS，因此 `cuda:0` 的正式 smoke 仍需在 Kaggle/工作站执行。

## 10. OPEN QUESTIONS / 风险

- **Q1（分析封装）**：E02 prediction export 与 error analysis 需 E02 专用薄封装；在实现与验证前，E02 不得借用 E01 的分析脚本作为正式证据。
- **Q2（λ 敏感性）**：λ=0.5 固定，不自动调参。若首轮 E02 未见改善，是否做 λ∈{0.25, 1.0} 的独立敏感性实验需用户决定；不得用 Val 反复挑选后再报首轮结果。
- **Q3（标签缓存）**：本地工程 smoke 会让 Ultralytics 在派生数据集 `labels/` 下生成 `train.cache`/`val.cache`（本轮已清理）。正式训练在 Kaggle 的派生目录上运行，不影响 E01 正式 run；如需保持本地派生目录逐字节洁净，应改用副本或在运行后清理。
- **Q4（Kaggle TensorBoard）**：Kaggle 运行时通常预装 tensorboard；若 `ultralytics` 的 SETTINGS 中 `tensorboard=True`，内置集成本会被激活，E02 已在进程内禁用以保证单一日志源，需在正式运行后核对 manifest 的 `tensorboard.builtin_ultralytics_integration`。
- **Q5（checkpoint 类路径）**：E02 checkpoint 会 pickle `ior_yolo.trainers.ordinal.OrdinalSegmentationModel` 等类，因此任何加载 E02 权重的进程都必须能 `import ior_yolo.trainers.ordinal`（`IOR-YOLO/` 在 `sys.path` 上）。重命名/移动该模块会使历史 checkpoint 不可加载，需保持路径稳定。
- **Q6（指标语义）**：E01 人工复核的 “semi-mature accuracy 83.33%” 来自 operational conf=0.65 的匹配后分析，与 Ultralytics 官方 mAP 不是同一统计量；E02 比较必须同口径（同样 conf、同样匹配规则、同样分母）。
- **Q7（单种子）**：首轮 E02 沿用 seed=0 单种子，与 E01 一致；任何“改善”都不能在单种子下称为稳定结论，需要时另开多种子实验。

## 11. 下一步（需用户授权后才推进）

1. 安装 E02 依赖：`python -m pip install -r IOR-YOLO/requirements-e02.txt`。
2. 在正式平台运行 `preflight --require-cuda --platform kaggle --source <D2源>`，确认 `e02_preflight.json` 中 E01 完整性、类序、TensorBoard 与运行目录隔离均通过。
3. 运行 `tensorboard-smoke`（可本地）与正式设备 `smoke --device cuda:0`，确认 dashboard 与 batch 可行性，再由用户在可见终端启动 `train`。
4. E02 `val` 完成后，实现并验证 Q1 的 E02 分析与预测导出封装，再与 E01 同口径比较；Final Test 仍按 E01 协议处理，且不在 E02 工程轮次内解锁。
