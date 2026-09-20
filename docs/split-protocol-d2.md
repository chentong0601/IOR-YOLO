# D2 v4 — Group-Aware Split Protocol Design

更新：2026-09-20。**Simulation Only / NOT FROZEN**。本文件只评价候选同源图是否容许合理的划分比例；没有建立 `data/splits/train.txt`、`val.txt`、`test.txt` 或任何正式样本归属。

**Stage 2B-4H说明**：下文100-seed及后附500×2模拟数字均来自**裁决前1114组图**，只作历史协议设计证据。人工裁决后当前图为**1112组**；尚未以新图重跑模拟，未选比例或seed，也未进入Stage 2B-5。用户已决定正式实验排除全部dataset-provided `_resize` 表示，输入尺寸由训练管线动态处理；其余评价图像选择与正式split尚未冻结。

**Stage 2B-5P更新**：现已用1112组、1096个guard clusters及1099张候选图做两种比例的**单次确定性模拟**；结果和最新候选规则请转读[split constraint design](split-constraint-design-d2.md)。本页原有100/500-seed数字仍仅代表历史1114组图，未据此选正式比例或seed。

## 拟议原则

1. 单位为 `source_group_id`，同组所有非 resize 图像必须进入同一集合。每张 `_resize` 与同名 original 绑定，不能独立划分。已知 3 对异常 resize 保留“同名伴随、关系 Unresolved”标记，正式使用前需处理。
2. 先处理剩余Candidate边和标注例外，才可冻结最终图。现行1112个候选组（裁决前1114）不等于fruit/tree/session分组；缺少采集ID时，论文不能声称已排除这些更深层泄漏。
3. 对训练集，优先使用真实非 resize 表示，在训练管线中做在线 resize/augmentation；对验证和最终测试，倾向只用可确定的采集代表图，避免将随包离线增强副本当独立评价样本。如何判定所有“原始采集图”尚未解决，故此项暂为提案。
4. 最终测试集在协议冻结后封存，不用于模型选择、调参、方法设计或根据 mAP 反选划分 seed。训练/验证的阶段标签分布、图像和组数可作为**划分前可见的审计信息**；最终评价指标不能参与选 split。
5. 需要记录每组原始图成员、增强关系、类别实例数、多类别图数、采集属性（如有）。当前 ZIP 未提供可靠 fruit/tree/session/light ID；不能做这些层级的真实分层。

## 模拟方法

`07_simulate_group_split.py` 读取 `source_groups.csv`、关系边、原始三份 JSON，只对 1406 张 original 表示做 **100 个固定 seed（0–99）** 的分组 dry-run，比较 70/15/15 与 70/10/20。按组大小从大到小贪心放入 train/val/test；同大小组先经 seed 打乱。透明评分是各集合目标比例的 L1 误差之和：

- 图像比例误差 × 1；
- 三类实例比例误差的均值 × 1；
- 多类别图像比例误差 × 0.5；
- source group 数量比例误差 × 0.25。

上述权重是**当前协议候选**，不是最优性证明；无学习型优化器。Confirmed、Strongly Supported、Supported 边被切断是**硬失败**，不允许以较低评分换取泄漏。Candidate/Unresolved 边不自动并入图，但会报告被切断数。两种比例都可达到接近目标的图像/类别分布；这**不证明**采集样本或最终评价独立。

重跑命令仅向终端输出 JSON，不写 split 文件：

```zsh
python3 IOR-YOLO/scripts/07_simulate_group_split.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/data/manifests --seeds 100
```

## 代表性结果

每行是 100 seed 中按上述评分最低的**展示用候选**，不是正式选择。三类计数顺序为 immature / semi-mature / mature；“多类图”是含至少两种有效阶段标签的 original 图像。

| 比例 / seed | 集合 | 图像 | 组 | 三类实例 | 多类图 | 图像比例偏差 |
|---|---|---:|---:|---|---:|---:|
70/15/15；15 | train | 970 | 801 | 644 / 536 / 574 | 76 | −1.010 pp |
 | val | 210 | 159 | 148 / 140 / 132 | 16 | −0.064 pp |
 | test | 226 | 154 | 129 / 138 / 133 | 16 | +1.074 pp |
70/10/20；36 | train | 968 | 793 | 645 / 534 / 569 | 76 | −1.152 pp |
 | val | 150 | 102 | 85 / 92 / 86 | 11 | +0.669 pp |
 | test | 288 | 219 | 191 / 188 / 184 | 21 | +0.484 pp |

1406 图像与三类 **921 / 814 / 839 = 2574 有效区域**均守恒；1 个 Unknown 区域不进入三类计数。70/15/15 的代表评分 0.080079，100 seed 中位 0.109303、范围 0.080079–0.136677；70/10/20 的代表评分 0.083525，中位 0.112420、范围 0.083525–0.133313。这两个评分差距不足以决定最终比例。所有 100×2 次模拟均保持组完整且 **Confirmed / Strongly Supported / Supported 边跨模拟 split = 0**。代表性候选分别切断 **13 / 12 条 Candidate 关系边**；这些不是已确认泄漏，但必须复核后才可决定最终分组。

## 未满足的前提

- 2 个跨官方 split 命名 Candidate 家族和 23 条仍跨不同组的 dHash Candidate 边没有来源真值；现有模拟可能把其两端放进不同集合。
- 相同 SHA 图像有不同 polygon；Unknown region 的 ignore/evaluation 规则尚未冻结。三对 resize 异常尚未解释。
- 1406 original 包含随包离线增强图；当前模拟平衡的是**表示数**，不是独立采集数。正式协议须决定验证/测试中如何处理这些增强图，并重新计算比例、类别和统计功效。
- 缺少 fruit、tree、session、设备/年份/光照逐图 ID，无法验证更高层的采集隔离。若未来得到元数据，必须优先按更严格的真实来源组修订图，再重做模拟。

因此 **Group-Aware Split 设计可继续，但正式划分与 Baseline 尚不准入**。比例、seed、候选边处置、评价图像选择及 Unknown region 策略需由用户审阅并在新一轮明确冻结；本轮未创建正式 split。

**Stage 2B-4V 复核补注（2026-09-20）**：模拟器已将 group ID 在随机打乱前稳定排序，使相同 seed 不受输入 CSV 行顺序影响；上表 100-seed 展示数字经复跑保持一致。独立审核又运行 500×2 次当前图模拟，确认已纳入图的三档关系始终零跨集，但 Candidate 边切断中位为 14。分布和偏置审查见[审核包](review-packet-stage2b4.md)；**未冻结比例或 seed**。
