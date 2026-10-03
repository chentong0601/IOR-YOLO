# E02 有序成熟度损失（ordinal-aware）实验协议

更新：2026-10-03（documentation-only freeze patch，逐项对齐代码冻结 commit `f7769ef0adb6ebae8ba91d70dbc2ff6e6a84453f`，branch `e02-ordinal`）。状态：**工程实现、审计与代码冻结完成；FORMAL TRAINING = NOT STARTED**。本轮不训练 100 epoch、不访问 Final Test，也不改动 E01 的任何正式代码、配置、manifest 或结果。E01 冻结证据见[E01 基线协议](e01-baseline-protocol.md)、[E01 结果分析协议](e01-analysis-protocol.md)与[研究计划与决策记录](../研究计划与决策记录.md)。

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

对分类 logits `z`（形状 `(B, A, K)`：B 为图像数，A 为该 batch 的全部 anchor 数，K=3），保留原始 Ultralytics 分类损失不变，另加

```
p        = softmax(z)
F_p(j)   = Σ_{k≤j} p_k                     j = 0 … K-2      预测累积概率
F_y(j)   = 1[j ≥ y]                        j = 0 … K-2      目标累积表示
err_i    = mean_j ( F_p^(i)(j) − F_y^(i)(j) )²          单实例 ordinal error
w_i      = target_scores[i].sum() = target_scores[i, assigned_class_i]   pinned task-aligned 质量权重

L_ord    = Σ_{i ∈ 匹配正样本锚点} w_i · err_i / max(Σ_i w_i, 1)
L_total  = Σ_k L_original_YOLO,k + λ · L_ord · batch_size     λ = 0.5（第一轮固定）
```

目标累积表示与冻结编码一致：immature → (1,1)、semi-mature → (0,1)、mature → (0,0)；`configs/experiments/e02_yolo11n_seg_ordinal.yaml` 的 `ordinal.cumulative_targets` 与代码常量逐项断言相等。**numerator 与 denominator 使用同一批 quality weights**：`w_i = target_scores[i].sum() = target_scores[i, assigned_class_i]` 逐匹配锚点计算一次，numerator 为 `Σ_i w_i · err_i`，denominator 为 `max(Σ_i w_i, 1)`（实现：`ior_yolo/losses/ordinal.py::ordinal_component`，诊断字段 `penalty_sum` / `penalty_sum_unweighted` / `weight_sum` / `normalization` / `quality_min|mean|max`）。**不存在“等权 numerator / 加权 denominator”的组合**；出现该组合即为缺陷。**`target_scores` 不是 one-hot 的 1**，而是 pinned Ultralytics 8.3.220 `TaskAlignedAssigner` 输出的 task-aligned quality score（one-hot 结构再乘以 `norm_align_metric`，匹配锚点上 `w_i ∈ (0, 1]`），与原始分类损失 `loss[2] = BCEWithLogitsLoss(...).sum() / target_scores_sum` 使用的是同一张量，因此辅助项与被扩展项保持同一实例尺度。代码另有两处硬校验：每个匹配锚点只允许一个非零目标类别（否则 `w_i` 不能被读作 assigned-class target score）、分母质量必须全部落在匹配锚点上（否则 numerator/denominator 群体不一致），任一失败即抛错。单实例惩罚上界为 1：完全正确 one-hot 为 0，完全自信的相邻错误为 0.5，完全自信的严重（跨两级）错误为 1。

λ 在训练前声明（`lambda_selection: predeclared_before_training`），不自动调参、不依据 Val/Test 修改；代码层面 λ 必须等于 0.5，否则 preflight 直接失败；`ior_yolo/losses/ordinal.py::E02_LAMBDA_ORD`、`configs/experiments/e02_yolo11n_seg_ordinal.yaml::ordinal.lambda_ord` 与 `scripts/24_e02_analyze_results.py::ORDINAL_LAMBDA` 三处独立断言 λ=0.5，run manifest 记录的 λ 与它们不一致时分析阶段直接拒绝。

## 3. 注入位置与 Ultralytics 侧形式

| 问题 | 记录 |
|---|---|
| 原始分类损失形式 | ultralytics 8.3.220 `v8SegmentationLoss.__call__`：`loss[2] = BCEWithLogitsLoss(reduction="none")(pred_scores, target_scores).sum() / target_scores_sum`，`target_scores` 为 task-aligned assigner 目标（**结构为 one-hot 位置 × 质量分数 `w_i ∈ (0,1]`，不是 0/1 的 one-hot，也不是全 1**），随后 `×hyp.cls`，整体以 `loss * batch_size` 返回，`__call__` 的 item 向量为 **4 元** `(box, seg, cls, dfl)` |
| 注入位置 | `IOR-YOLO/ior_yolo/trainers/ordinal.py::OrdinalSegmentationLoss.__call__`（子类，先调用 `super().__call__` 得到**未改动**的原损失，再加辅助项） |
| 合并方式（最终实现） | **只在 classification 槽位注入一次**：`merged = original_total.clone()` 后 `merged[ORDINAL_LOSS_SLOT] += λ_ord·L_ord·batch_size`（`ORDINAL_LOSS_SLOT = 2`，即 cls 分量），因此优化目标严格为 `Σ_k L_stock,k + λ_ord·L_ord·batch_size`；item 向量追加第 5 项 `λ_ord·L_ord`，因此 `results.csv` 与 TensorBoard 出现 `train/ordinal_loss`、`val/ordinal_loss`，其余 4 项数值不变 |
| 目标来源 | `AssignerCallRecorder` 代理记录 pinned `TaskAlignedAssigner` 的**最近一次**调用入参/输出，直接复用原分类损失所比对的同一批匹配锚点与阶段标签；不重跑 assigner、不复制目标构建逻辑、用后立即释放（不跨 step 保留张量、不进入 checkpoint） |
| 正确性自检 | 从头部原始输出重建的 logits 与原损失交给 assigner 的 `sigmoid` 张量逐元素比对（`atol=1e-4, rtol=1e-3`），不一致即抛错，避免静默优化错误通道；eval/validation 下 Ultralytics 会以 `(detections, (feats, mask_coeffs, proto))` 包装头部输出，重建逻辑按通道数结构化定位 level 列表，找不到即抛错 |
| 侵入程度 | 不复制/不 patch Ultralytics 源码；仅覆盖 `init_criterion`、`get_model`、`get_validator`，并新增 `OrdinalYOLO` 的 `task_map`（只替换 segment 的 model/trainer，validator/predictor 与其它任务保持原样） |

### 3.1 objective 注入：广播缺陷与单次注入修复（audit 记录）

- **缺陷（修复前，audit 发现）**：辅助项被直接加到原始返回值上。pinned 8.3.220 的 `v8SegmentationLoss.__call__` 返回 **4 元向量** `(box, seg, cls, dfl) × batch_size`，因此 `merged = original_total + λ_ord·L_ord·batch_size` 这类写法会把标量**广播到 4 个分量**，trainer 求和后实得 `Σ_k L_stock,k + 4·λ·L_ord·batch_size`——**λ=0.5 实际等效为 2.0**，ordinal 项梯度被放大 4 倍；同一缺陷还会让记录端的 `float(...)` 收到向量而报错。
- **修复（最终实现）**：`merged = original_total.clone()` 后 `merged[ORDINAL_LOSS_SLOT] += λ_ord·L_ord·batch_size`（`ORDINAL_LOSS_SLOT = 2`），**只在 classification 槽位注入一次**；真正被优化的目标为 `Σ_k L_stock,k + λ_ord·L_ord·batch_size`，λ_ord 固定 0.5。
- **硬门**：`ior_yolo/trainers/ordinal.py::scale_record` 每步重新推导 `delta_total`，只有当 `delta_total ≈ λ_ord·L_ord·batch_size` 才写记录，否则抛 `RuntimeError`；item 向量布局未知（非 4 元/5 元）或槽位异常时抛 `ValueError`。因此“静默 4 倍放大”和“静默不注入”都无法通过。

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
| `scripts/23_e02_export_predictions.py` | E02 prediction export：只读复用冻结 E01 exporter（`18_export_e01_predictions.py`），conf=0.001 / iou=0.7，split 固定 `val`，全部 E02 provenance 门在任何导出动作之前执行，已有导出拒绝覆盖 |
| `scripts/24_e02_analyze_results.py` | E02 operational 分析：只读复用冻结 E01 分析原语（`17_analyze_e01_results.py`），conf=0.65，只评 `val`，输出 `analysis/val_conf_0p65_ordinal_v1/`，已有分析拒绝覆盖 |
| `requirements-e02.txt` | 继承 `requirements-e01.txt`，追加 `tensorboard>=2.14,<3` |
| `tests/test_e02_ordinal_loss.py`、`test_e02_ordinal_training.py`、`test_e02_tensorboard.py`、`test_e02_preflight.py`、`test_e02_analysis.py` | E02 单元与性质测试（数学、注入与单次注入硬门、TB 契约、preflight/隔离/锁定、export/analysis provenance 门与版本化输出） |
| `docs/e02-ordinal-protocol.md` | 本文档 |

E02 与 E01 的差别只有：辅助损失、E02 自身的 config/manifest/preflight/runner/分析与导出封装/测试/文档，以及 `tensorboard` 依赖。`requirements-e01.txt` 未改动；E01 的 config、configs/data、manifests、scripts、tests 一律只读。

## 5. TensorBoard（正式训练硬要求）

E02 使用**单一 canonical writer**：`torch.utils.tensorboard.SummaryWriter`，写入 `<run_dir>/tensorboard/`，绝不写 E01 目录。Ultralytics 内置 TensorBoard 集成在 E02 进程内被显式禁用（`neutralize_builtin_ultralytics_tensorboard()`，在 `BaseTrainer.__init__` 注册集成回调**之前**执行，不修改任何 Ultralytics 设置文件，也不影响非 E02 路径），原因是该集成无法记录 `ordinal_loss`，重复日志会让 dashboard 与论文溯源产生歧义；禁用结果写入 manifest 的 `tensorboard.builtin_ultralytics_integration`。

| 分组 | tag |
|---|---|
| Training（代码分组 `train`，5 个） | `train/box_loss`、`train/seg_loss`、`train/cls_loss`、`train/dfl_loss`、`train/ordinal_loss` |
| Validation | `val/box_loss`、`val/seg_loss`、`val/cls_loss`、`val/dfl_loss`、`val/ordinal_loss` |
| Metrics | `metrics/precision(B)`、`metrics/recall(B)`、`metrics/mAP50(B)`、`metrics/mAP50-95(B)`、`metrics/precision(M)`、`metrics/recall(M)`、`metrics/mAP50(M)`、`metrics/mAP50-95(M)` |
| Optimization（代码分组 `optimization`，7 个） | `epoch`、`lr/pg0…`、`perf/gpu_memory_gb`、`perf/epoch_time_seconds`、`perf/elapsed_seconds`、`perf/iteration`、`train/total_loss` |

必需 tag 共 **25** 个（`ior_yolo/utils/tensorboard_logger.py::REQUIRED_TAGS`，`required_tag_list()` 长度 25），上表分组与代码常量逐项一致。
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

**E02 分析封装（已实现并冻结于 `f7769ef`）**：`scripts/23_e02_export_predictions.py` 与 `scripts/24_e02_analyze_results.py` 是 E01 冻结脚本的**只读薄封装**：不重新实现匹配与成熟度统计，也不修改 `17`/`18`（两者都在 config 的 `e01_reference.files_sha256` 内，改动即被 `assert_e01_immutable` 拒绝）。

| 项目 | 冻结口径 |
|---|---|
| prediction export | `conf = 0.001`、`iou = 0.7`（复用冻结 E01 exporter `scripts/18_export_e01_predictions.py`）；导出流**不做阈值过滤**，保留完整 AP/PR 证据 |
| operational 分析 | `conf = 0.65`（E01 Validation F1 工作点，Test 前冻结）；`--operating-confidence` 传任何其他值直接 `PermissionError` |
| split | 仅 `val`（`SPLIT = "val"`，`assert_validation_split` 拒绝其他取值；E02 封装不传递也不接受 `--split test` / `--final-test`） |
| Final Test | locked（继承 E01 协议，E02 本轮不解锁） |
| 分析输出目录 | `<run_dir>/analysis/val_conf_0p65_ordinal_v1/`（`analysis_val.json` + `matched_predictions_val.csv` + `failure_cases/` + `figures/`）；已存在的导出或分析一律 `FileExistsError`，不静默覆盖 |
| 匹配规则 | 复用冻结 E01 原语：不区分类别的 box IoU ≥ 0.5 贪心一对一匹配；mask IoU 单独计算；未配对 GT 记 missed、未配对预测记 FP，均不进入 3×3 混淆矩阵 |

**E02 provenance 硬门（在任何 run 文件被读取之前执行，`required_manifest_evidence`）**：manifest 必需字段齐全；`experiment_id` 正确且 `status = validated_not_final_tested`；Box **与** Mask 的 precision/recall/mAP50/mAP50-95 都必须是有限实数（`measured()`，文本或 flag 值不算）；`experiment_config_sha256` / `resolved_config_sha256` / `best_checkpoint_sha256` 与当前文件逐一对上（比 E01 更严：E01 中 config 摘要只是 warning）；`training_seed = 0`；`lambda_ord = 0.5` 且 manifest 与 config 的 ordinal loss 定义一致；`criterion_class = OrdinalSegmentationLoss`；`scale_history` 非空；loss 向量含 `ordinal_loss`；内置 Ultralytics TB 集成已 neutralized（保证单一日志源）；并用 `verify_training_artifacts()` 从 run 自身重新校验 `results.csv` 列与 TensorBoard 必需 tag。任一失败即抛错，因此不可信的 run 不会产生任何分析产物。

**E01 冻结参考只读**：E02 的 E01 基线数字只从 `configs/experiments/e02_yolo11n_seg_ordinal.yaml::e01_reference.frozen_results` 读取（`compare_to_e01`），**从不重算**；`FROZEN_RESULT_KEYS` 逐键比对，无法重建的键写入 `not_reproduced` 而不是被丢弃；E02 报告与图只写自己的版本化目录，E01 的 `analysis/val_conf_0p65_v2/` 不被写入或覆盖。

**低于阈值预测的语义**：`filter_predictions`（冻结 E01 原语）只保留 `confidence ≥ 0.65` 的预测，其余数量单独记入 `operating_point.predictions_below_threshold_excluded`，并固定记录 `prediction_export_modified: false`。因此低于工作点的导出预测**既不参与匹配，也不得被称为 false positives**：在 `conf=0.001` 的导出流中它们不是部署工作点下的检出，把它们全部计为 FP 会使 FP/missed 计数失去意义。

## 9. 本轮已验证证据（工程证据，非实验结果）

| 验证 | 结果 |
|---|---|
| 全量测试（冻结 commit `f7769ef`、干净工作树） | `python -m unittest discover -s IOR-YOLO/tests -t IOR-YOLO/tests` → **213 tests OK**；E02 子集 `test_e02_*.py` 153 项（ordinal loss+training 73、TensorBoard 23、preflight/隔离/锁定 23、analysis 34） |
| 语法检查 | `python -m py_compile` 覆盖 11 个 E02 模块（`losses/ordinal.py`、`trainers/ordinal.py`、`utils/tensorboard_logger.py`、`22`/`23`/`24`、5 个 E02 test 文件）→ **exit 0** |
| 工作树检查 | `git diff --check` → **exit 0**（无空白错误）；E01 相关 tracked diff 为空 |
| TensorBoard 真实 writer round trip | `test_e02_tensorboard.py` 23 项 OK，含 `test_full_required_contract_roundtrips_with_the_real_writer`：**25/25 必需 tag 写入并读回、`missing_tags: []`**，event file 生成 |
| E01 完整性 | `assert_e01_immutable` 在冻结 commit / 干净工作树现状下 `status = verified`、**files_checked = 9** |
| 运行目录隔离 | `assert_run_isolation` → **isolated = True**（E02 run 目录 `runs/e02_yolo11n_seg_ordinal/seed_0`，不与 E01 run 目录相等/包含/命名重叠） |
| 当前实现 CPU 端到端 smoke | 见 §9.1：`criterion_class = OrdinalSegmentationLoss`、`assigner_calls = 2`、`criterion_steps = 2`，全部恒等式通过 |
| 冒烟修复记录（早期） | 首次 CPU smoke 暴露 eval/validation 模式下 head 输出被包装为 `(detections, (feats, mask_coeffs, proto))` 导致 logits 重建失败；已改为按通道数结构化定位并补回归测试（`test_logits_are_recovered_from_the_eval_mode_wrapping`） |

### 9.1 当前实现的 CPU smoke（仅 engineering scale sanity）

数值来自冻结代码在本地 CPU 上对真实 D2 派生数据的 disposable smoke（1 epoch / fraction=0.02，2 个 criterion step；最后一个 step 为 7 张图）。输出写在**仓库外的临时目录**（未写入 `runs/e02_engineering/`，以免覆盖修复前的历史 smoke 目录），因此不是论文结果。**这些数字只用于 scale sanity check，λ 始终保持 0.5，未据此调参。**

| 量 | 值 |
|---|---|
| matched anchors（该 step） | 280 |
| quality 权重 mean / min / max | 0.6225 / 0.0040 / 0.9899 |
| raw `L_ord`（= weighted num / denom） | 0.344527 |
| weighted numerator `Σ_i w_i·err_i` | 60.052483 |
| unweighted error sum `Σ_i err_i` | 96.683868 |
| denominator `max(Σ_i w_i, 1)`（= `target_scores.sum()`） | 174.304016 |
| `weight_sum`（matched mass） | 174.303986 |
| `λ_ord·L_ord` | 0.172264 |
| `train/cls_loss`（epoch mean，stock 项） | 3.26121 |
| `train/ordinal_loss`（epoch mean） | 0.17410 |
| classification 相关贡献（logged item scale） | 3.26121 + 0.17410 = 3.43531 |
| objective（该 step，batch-scaled） | 51.602467 → 52.808315 |
| delta | 1.205845 = `λ_ord·L_ord·batch_size`（0.172264 × 7） |

真实数据上逐项通过的恒等式：`L_ord == Σ_i w_i·err_i / max(Σ_i w_i, 1)`；`denominator == Σ_i w_i == target_scores.sum()`（无 clamp 生效）；`penalty_mean == Σ_i w_i·err_i / Σ_i w_i`；`λ_ord·L_ord == weighted loss`；**objective 恰增加 `λ_ord·L_ord·batch_size`**；且 `Σ_i w_i·err_i ≠ Σ_i err_i`（证明质量加权在真实数据上确实生效）。

历史记录（superseded，仅留痕）：旧 `runs/e02_engineering/local-smoke/`（2026-10-03 09:56，修复前修订，HEAD `c22134cd` + 未提交 E02 工作树）记录 `last_ordinal.penalty_sum = 96.68387`（= `Σ_i err_i`）、`loss = 0.55469`（= `Σ_i err_i / Σ_i w_i`）、`train/ordinal_loss = 0.28212`，即当时是**未加权分子 / 加权分母**；该目录保留为修复前证据，**不得再作为当前实现或论文证据引用**（两轮 `train/cls_loss = 3.26121` 完全一致，说明 stock 损失路径未被改动）。

本地工程环境：Python 3.12(.venv)、torch 2.14.1+cpu、torchvision 0.29.1+cpu、ultralytics 8.3.220、tensorboard 2.21.0（本轮为运行 TB smoke 安装到 `.venv`，并已写入 `requirements-e02.txt`；E01 的 `requirements-e01.txt` 未改动）。本地无 CUDA/MPS，因此 `cuda:0` 的正式 smoke 仍需在 Kaggle/工作站执行。

## 10. OPEN QUESTIONS / 风险

- **Q1（分析封装）— CLOSED（2026-10-03）**：E02 prediction export 与 error analysis 已实现并随 `f7769ef` 冻结：`scripts/23_e02_export_predictions.py`、`scripts/24_e02_analyze_results.py` 只读复用冻结 E01 原语（`17`/`18`），自带 E02 provenance 硬门与 E02 config 哈希，回归测试 `tests/test_e02_analysis.py` 34 项通过。本条不再是 OPEN QUESTION；冻结口径见 §8。
- **Q2（λ 敏感性）**：**future ablation candidate only**，不属于 E02 round-1 当前任务。round-1 的 λ 冻结在 **0.5**（训练前声明，不得依据 Val/Test 修改，也**不做 validation-guided tuning**）；若将来要做 λ∈{0.25, 1.0} 一类敏感性实验，须作为独立实验另立协议、另记配置与结果，不得替换或回溯修改 round-1 的冻结工作点。
- **Q3（标签缓存）**：本地工程 smoke 会让 Ultralytics 在派生数据集 `labels/` 下生成 `train.cache`/`val.cache`（本轮已清理）。正式训练在 Kaggle 的派生目录上运行，不影响 E01 正式 run；如需保持本地派生目录逐字节洁净，应改用副本或在运行后清理。
- **Q4（Kaggle TensorBoard）**：Kaggle 运行时通常预装 tensorboard；若 `ultralytics` 的 SETTINGS 中 `tensorboard=True`，内置集成本会被激活，E02 已在进程内禁用以保证单一日志源，需在正式运行后核对 manifest 的 `tensorboard.builtin_ultralytics_integration`。
- **Q5（checkpoint 类路径）**：E02 checkpoint 会 pickle `ior_yolo.trainers.ordinal.OrdinalSegmentationModel` 等类，因此任何加载 E02 权重的进程都必须能 `import ior_yolo.trainers.ordinal`（`IOR-YOLO/` 在 `sys.path` 上）。重命名/移动该模块会使历史 checkpoint 不可加载，需保持路径稳定。
- **Q6（指标语义）**：E01 人工复核的 “semi-mature accuracy 83.33%” 来自 operational conf=0.65 的匹配后分析，与 Ultralytics 官方 mAP 不是同一统计量；E02 比较必须同口径（同样 conf、同样匹配规则、同样分母）。
- **Q7（单种子）**：首轮 E02 沿用 seed=0 单种子，与 E01 一致；任何“改善”都不能在单种子下称为稳定结论，需要时另开多种子实验。

### 10.1 仍然存在的限制（不得声明为已解决）

- **formal Kaggle CUDA smoke 尚未运行**：本地无 CUDA/MPS，`smoke --device cuda:0 --platform kaggle` 与正式平台 preflight 仍需用户在 Kaggle 可见执行；本地 CPU smoke 只证明代码路径，不证明 Kaggle 端到端可跑。
- **formal E02 training 尚未运行**：`runs/e02_yolo11n_seg_ordinal/seed_0` 不存在，`results.csv`、`run_manifest.yaml`、`best.pt`、TensorBoard event 均未产生。E02 目前**没有任何实验结果**，只有工程证据。
- **single-seed limitation**：round-1 沿用 seed=0 单种子，任何“改善”都不能在单种子下称为稳定结论。
- **checkpoint module-path stability**：E02 checkpoint 会 pickle `ior_yolo.trainers.ordinal.OrdinalSegmentationModel` 等类路径；加载任何 E02 权重都要求该模块路径稳定且 `IOR-YOLO/` 在 `sys.path` 上。
- **Final Test locked**：E02 轮次内不解锁 Final Test；E02 的 val 结果不得用于最终论文结论或模型选择。
- **E01 分析/导出脚本内部仍是 E01 专用**：`17`/`18` 的 provenance 措辞与 `experiment_config_sha256` 硬编码仍指向 E01（本轮不改动冻结证据），E02 侧改由 `23`/`24` 的外部门禁覆盖。

## 11. 下一步（需用户授权后才推进）

1. 安装 E02 依赖：`python -m pip install -r IOR-YOLO/requirements-e02.txt`。
2. 在正式平台运行 `preflight --require-cuda --platform kaggle --source <D2源>`，确认 `e02_preflight.json` 中 E01 完整性、类序、TensorBoard 与运行目录隔离均通过。
3. 运行 `tensorboard-smoke`（可本地）与正式设备 `smoke --device cuda:0`，确认 dashboard 与 batch 可行性，再由用户在可见终端启动 `train`。
4. E02 `val` 完成后，按冻结顺序运行 `python IOR-YOLO/scripts/23_e02_export_predictions.py`（conf=0.001 / iou=0.7，仅 `val`）与 `python IOR-YOLO/scripts/24_e02_analyze_results.py --summary`（conf=0.65，仅 `val`），与 E01 冻结参考同口径比较；已存在的导出或分析目录会被拒绝覆盖，需要新结论时必须新建版本化目录，而不是改写旧产物。Final Test 仍按 E01 协议处理，且不在 E02 工程轮次内解锁。
