# Stage 2B-4V 决策审核包

更新：2026-09-20。**Independent Verification / Review Pending；D2 = NOT FROZEN。**本轮不进入 Stage 2B-5，也未创建正式 split。底层复核证据在 `IOR-YOLO/reports/dataset_audit/verification/`；方法与复现入口见 [d2-audit-reproduction.md](d2-audit-reproduction.md)。

**当前裁决状态**：本页以下1114组、65跨集组、2个命名Candidate和“需要用户决定”的表述属于**Stage 2B-4V裁决前快照**。用户已完成Stage 2B-4H的21案人工裁决，当前重建为**1112组、67跨官方split组**；现行决定、剩余不确定性和逐案理由见[人工复核记录](human-review-stage2b4.md)及[同源组当前补注](source-group-resolution-d2.md)。先前的500×2模拟以旧图运行，**不能冒充新图模拟或正式split**。D2仍NOT FROZEN，未进入Stage 2B-5。

## 10 个最可靠事实

1. 原包 `dataset-20260508.zip` SHA256 为 `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`；独立读取的 original 与 resize 成员哈希和 Stage 2B-3 文件清单一致。
2. ZIP 内有 **1406 original representations + 1406 resize representations = 2812 image files**。这不是 2812 个独立采集样本。
3. 原始三份 JSON 与 1406 个非 resize 文件匹配；共 **2575 polygons，2574 个有效三类标签（921/814/839），1 个 Unknown**。
4. 全图 2812/2812 已在 Stage 2B-3 完整解码；本轮未发现与文件 manifest 不一致的成员。
5. 从 ZIP、JSON、像素与哈希独立重建的当前图为 **1114 groups**；成员集合与 `source_group_id` 均和 Stage 2B-4 清单**完全一致**。
6. 当前图有 **65 组**跨官方 train/val/test；官方 split 另有 3 组不同名但 SHA256 完全相同的跨集图像，故**不可作为正式评价划分**。
7. 67 个命名跨集家族中 65 个满足当前 Strongly Supported 规则，2 个仍是 Candidate；“家族最强边”不保证家族内每条边同样强。
8. 3 组跨集相同像素图的**类别和 region 数一致，polygon 坐标不同**；不是单纯 region 顺序变化。
9. 1403 对 original↔resize 满足现有像素筛查；`172_brightness`、`172_noise`、`327` 三对仍异常且可解码。
10. 500 seed × 两比例的当前图模拟中，Confirmed、Strongly Supported、Supported 边**均零跨集**；这只证明图内约束，不证明真实采集独立。

## 10 个最危险风险

1. **已确认官方 split 泄漏**；继续沿用会污染 Baseline/最终测试。
2. 真实 fruit/tree/session ID 缺失；1114 是候选来源组，不是独立采集样本数。
3. 23 条 dHash Candidate 两端仍跨组；当前模拟每 seed 有 1–25 条 Candidate 边被切断（不同目标比例分布见下）。
4. 两个命名 Candidate (`img_13960`、`img_14040`) 有相同 polygon 与高 SSIM，但缺采集或增强 provenance；当前仍分组。
5. 两条 Supported 跨名视觉边是连通分量的桥；一条弱边可以把强家族与另一图串起来。
6. 命名相关阈值从 0.995 到 0.997 使组数 **1114→1150**；近阈值家族需复核。
7. 相同像素图 polygon 不同；训练目标不一致，验证/测试 mAP 的 GT 定义可变。
8. `IMG_54350.jpg#3` 无类别；简单删除派生 polygon 可能把真实目标当背景或引入评价假阳性。
9. 三对 resize 的像素、比例及坐标异常；“全部只是统一 resize”不能成立。
10. 1406 个 original 表示中含离线增强；若直接全部进入验证/测试，会把派生表示当额外评价样本，且来源侧 1124/2573/2754 数字仍未解释。

## 核心反证

| 协议 | 合并的原始图像关系 | 组数 | 最大组 | 跨官方 split 组 | 70/15/15 与三类分布 |
|---|---|---:|---:|---:|---|
Conservative | Confirmed + Strongly Supported | **1120** | 6 | 65 | 100 seed dry-run 可达近似比例；但 Supported 视觉边可被切断，仍有残余同源风险 |
Current | Confirmed + Strongly Supported + Supported | **1114** | 6 | 65 | 500 seed dry-run 可行；Candidate 边仍可能跨集 |
Worst-Case | Current + **现有 35 条 Candidate 原始图像边** | **1096** | 6 | 72 | 100 seed dry-run 仍可达到相近图像/类别分布；不能因此证明 Candidate 全真 |

Worst-Case 仅假设当前已列出的 Candidate 都是同源；**没有**把未知同果不同视角或同树/场次关系算进去。Conservative 的 Supported 切断中位数为 8 条/seed；当前的 Candidate 切断中位数为 14 条/seed。Worst-Case 下已列边均可保持同组，D2 作为**候选主数据集**在数量上仍可用，但评价独立性与标签质量仍未达到冻结条件。

### 阈值与传递闭包

| 单变量变化 | 组数范围 | 判断 |
|---|---:|---|
命名家族 64×64 Pearson 0.993 / **0.995** / 0.997 | **1110 / 1114 / 1150** | 高侧变化 +36 组（约3.2%），非灾难性但有明显阈值敏感性 |
跨名 dHash 距离 ≤10 / **12** / 14 | 1114 / 1114 / 1114 | 现有其他像素门槛下组数稳定；Candidate 边数 29/35/44 |
跨名 64×64 Pearson 0.993 / **0.995** / 0.997 | 1114 / 1114 / 1116 | 高侧 +2 组 |
跨名全分辨率 Pearson 0.95 / **0.96** / 0.97 | 1114 / 1114 / 1114 | 当前结构稳定；Candidate 数在0.97时 +1 |
resize MAE 18 / **20** / 22 与相关度 0.993 / **0.995** / 0.997 | 均 1403 对通过 | 不改变以 original 为节点的组数；3 对异常均仍不过关 |

连通图存在 “A~B 强、B~C 弱仍同组” 的实际案例：`sg-0006` 内 `train/2340.jpg` 与其亮度/噪声版为 Strongly Supported，而到 `test/1270.jpg` 只有一条作为桥的 Supported 视觉边（64×64 相关 0.995166；全分辨率 0.989091）。另一条桥为 `train/1640.jpg ↔ test/3100.jpg`（0.996983 / 0.984046）。删去各自桥边会拆组；它们是最优先人工复核的 **False Merge 风险**。相反，2 个命名 Candidate、23 条跨组 dHash Candidate 和未知多视角是 **False Split 风险**。无采集 ID 时，不能把任一风险归零。

### 最弱的 10 个 Strongly Supported 家族

按家族内最弱的、仍达到 Strongly Supported 的跨 split 边相关度排序：`1770` 0.995345、`img_13700` 0.995642、`2040` 0.995814、`194` 0.995840、`img_1406` 0.996001、`img_13650` 0.996408、`1180` 0.996412、`257` 0.996486、`1550` 0.996601、`img_1381` 0.996648。它们**按当前规则**仍成立，因为均有明确后缀家族、跨集相关度≥0.995和相同完整 region 列表；这不证明采集身份。全部 67 家族的成员、尺寸、类别、polygon、dHash 与理由见 `family_review_67.csv`。实际文件名仅出现 `_brightness`、`_noise`；规则还允许 gamma/hsv/gaussian，但本包未借这些后缀做结论。数字及 IMG 编号不是独立来源 ID，同名碰撞仍可能。

两个 Candidate 的追加诊断：`img_13960` 的 64×64/全分辨率相关为 0.994468/0.992283，8×8 块 SSIM 为 0.982210；`img_14040` 为 0.990993/0.989174，SSIM 为 0.978693。二者尺寸均 404×303、polygon 完全相同、无 EXIF 采集字段。高相似不能证明同一采集样本；需要原始采集文件、增强脚本/参数或可信 fruit/session ID，故**不升级 Candidate**。26 条 dHash Candidate 均无完全相同 region 列表；9 条像同一场景或来源但证据不足，11 条只能说外观/场景相似，6 条证据更弱。缺少树、场次和背景 ID，无法把“同树/同背景但独立拍摄”与“同一来源派生图”可靠分开。这些是诊断分类，**不是来源真值**。由于 23 条仍跨组，且 500-seed 模拟的 Candidate 切断中位均为 14，**Candidate residual leakage risk = High（未确认泄漏）**。

三对异常 resize：同名原图对 resize 的相关度为 **0.772601 / 0.768708 / 0.848415**，LANCZOS RGB MAE 为 **39.118 / 36.477 / 31.311**；块 SSIM 为 **0.432 / 0.439 / 0.751**。`172` 同家族其他原图与异常 resize 的相关度更低，坐标比例残差也没有改善；`327` 无同家族替代。三对均无 EXIF，且有同名 JSON；不像单纯 JPEG 重压缩，但**裁剪、非统一处理或配错图不能判定**，仍 Unresolved。

三组精确重复的完整坐标见 `exact_duplicate_annotations.json`：均属 **C. polygon 坐标不一致**，不是 A. 类别不一致、B. 区域数不一致或 D. 仅 region 顺序不同。为什么重标，当前无作者标注历史，**不能推断**。若同组训练图保留两套 GT，模型会受到不一致定位监督；若落在验证集，阈值/错误分析会受 GT 边界影响；若落在最终测试，mAP 可因同图标注版本变化而失去稳定解释。必须同组并预先确定派生标注协议，不可静默选择“看起来更好”的 polygon。

## 500×2 模拟的稳定性与 score 偏置

下列每次统计取各 seed 的“train/val/test 中最大绝对偏差”（百分点）；其后汇总 min / median / mean / P95 / max。**没有选或冻结最终 seed**。

| 目标 | 量 | min | median | mean | P95 | max |
|---|---|---:|---:|---:|---:|---:|
70/15/15 | 图像比例偏差 | 0.085 | 2.077 | 2.001 | 2.788 | 3.357 |
 | source group 比例偏差 | 0.108 | 2.083 | 2.052 | 3.070 | 3.878 |
 | 任一成熟类实例比例偏差 | 3.538 | 4.644 | 4.643 | 5.381 | 5.872 |
 | 多类别图比例偏差 | 0.370 | 0.370 | 0.370 | 0.370 | 0.370 |
 | Candidate 边切断数 | 1 | 14 | 14.012 | 20 | 25 |
70/10/20 | 图像比例偏差 | 0.156 | 2.077 | 2.025 | 2.788 | 3.357 |
 | source group 比例偏差 | 0.126 | 1.993 | 2.000 | 2.980 | 4.327 |
 | 任一成熟类实例比例偏差 | 3.292 | 4.521 | 4.525 | 5.258 | 5.749 |
 | 多类别图比例偏差 | 0.556 | 0.556 | 0.560 | 0.556 | 1.296 |
 | Candidate 边切断数 | 2 | 14 | 13.898 | 20 | 25 |

两比例在 500 次中 Confirmed / Strongly Supported / Supported 跨集数的 min、median、mean、P95、max **全部为 0**。各 split 最大组大小之差：两比例均 min=0、median=0、mean=0.602、P95=2、max=3 张。满足“图像最大偏差≤2pp、任一类≤5pp、多类图≤5pp”的 seed 分别有 **165/500** 与 **157/500**，说明可行但并非所有种子都好。

当前评分给多类别图 0.5 权重，使其比例几乎固定，却仍容许成熟类实例约 4.5pp 的中位最大偏差，图像约 2pp；不能仅凭单一总分评估质量。可供以后比较的两个更透明版本：①预先声明硬容差（如图像≤2pp、各类≤5pp、多类图≤5pp），再在合格方案中最小化总偏差；②按最大图像/类别偏差做字典序排序，随后才比较多类图和组数。**本轮不替换现有 score、不选 seed。**

### 脚本审查结论

`06_build_source_groups.py` 对 D2 的1406/2812计数有硬编码，适用于固定v4包而非通用数据集；它信任 Stage 2B-3 的 SHA CSV，而不在每次构图时重算全部原图 SHA。独立验证已核实这一次原包与清单完全一致；后续复现必须先核对 ZIP SHA256。`source_family_review.csv` 取家族最强跨集边作为总状态，可能遮盖同家族其他 Candidate 边，须看逐对关系和本轮 67 行复核表。固定小数阈值周边的图像关系与 Pillow/Python 版本应一起记录（本轮 Python 3.12.14、Pillow 12.3.0）。

`07_simulate_group_split.py` 原先按输入行顺序生成同 seed 的随机处理序列；本轮做了**最小修复**：先稳定排序 group ID 再打乱，并允许最多500 seeds。100-seed 既有展示结果未变，三次隔离重跑输出一致。模拟只用本地 `random.Random(seed)`，无GPU、无原包写入；但分布评分使用全部原始 JSON 标签，**正式 seed/比例须在模型性能观察前预注册**，不能事后依 mAP 反选。两个脚本都没有生成正式 split；现有 provenance 主要靠文档、原包 SHA 与运行记录，不应把目录存在当作协议已冻结。

## 需要用户决定 / 可以继续 / 会阻止 Freeze

需要用户审阅：①两条视觉桥边及 23 条跨组 dHash Candidate 是否要人工来源复核；②验证/测试是否仅保留可核实的非增强采集表示；③ Unknown polygon 的 ignore 或整图不计分协议；④三对异常 resize 是否完全不用于实验并记录例外；⑤正式比例、seed 选择规则和缺少采集 ID 时允许的论文主张边界。上述均**未替用户定案**。

来源侧 1124/2573/2754 差异已正式登记、异常 resize 可暂不作为训练输入，这些不妨碍继续**只读审计和协议准备**。但**正式 Dataset Freeze** 仍被官方泄漏后无已冻结替代 split、Candidate 来源关系、采集 ID 缺失、同像素不同 polygon、Unknown region、评价集增强筛选与异常文件处置阻挡。D2 仍是 Primary Candidate，不是 Frozen。

Freeze 阻断项按严重性排序：**① 已确认官方泄漏且无正式替代 split；② 缺采集 ID 与跨组 Candidate 关系，无法保证来源独立；③ 同像素不同 polygon 和 Unknown region 的评价规则未验证；④ 离线增强/resize 表示的最终评价筛选未定；⑤ 三对异常 resize 的纳入/排除与数值口径记录未冻结。**这些排序是本轮审稿风险判断，不等于用户已经接受处置方案。

未来论文 Dataset/Methods 可考虑的**表述草案**：“D2 v4 压缩包含 1406 个非 resize 图像表示和 1406 个同名 resize 表示，共 2812 个图像文件。基于哈希、命名、标注几何与像素证据，我们构建了 1114 个候选来源组并按组规划评估；真实独立采集样本数未知。”其中前三个文件数为 **Confirmed**，1114 为 **Derived/Candidate**，不能写成 “1114 independent samples” 或 “1406 independent captures”。这不是正式论文正文。

## 严格审稿人问题（至少 15 项）

| # | 问题 | 当前证据能否回答 |
|---:|---|---|
1|官方划分是否有同字节图跨 train/test？|**Can answer**：有，3 组不同名跨集重复。|
2|离线增强发生在划分前还是后？|**Cannot answer**：缺作者执行日志。|
3|同一果实、树或拍摄场次是否跨最终集合？|**Cannot answer**：缺采集 ID，最终集合未建立。|
4|1406 是否为独立采集数？|**Can answer**：不能这样表述；含增强与重复。真实数未知。|
5|resize 是否被当作独立样本？|**Can answer**：候选协议中没有；1406 个同名伴随表示。|
6|全部 resize 都由统一纯缩放得到吗？|**Partially answer**：1403 对相似，3 对异常；生成管线未知。|
7|跨名 SHA 相同图的 class、region、polygon 是否一致？|**Can answer**：class 与数量一致，坐标不一致。|
8|同像素图为何标出不同 polygon？|**Cannot answer**：无标注历史。|
9|无类别 region 如何进入训练和 mAP？|**Partially answer**：有候选 ignore/排除协议，尚未实现验证。|
10|三阶段是真实生理成熟度吗？|**Can answer**：当前是视觉颜色阶段，不支持生理真值宣称。|
11|dHash 是否遗漏不同视角同果图？|**Cannot answer**：可能，缺 fruit ID 与全面人工复核。|
12|阈值微调是否显著改变组？|**Partially answer**：0.995→0.997 加 36 组；物理真值未知。|
13|连通分量是否会由弱边串接？|**Can answer**：有 2 条 Supported 桥边；是否误并仍未知。|
14|组号、清单与模拟可重复吗？|**Can answer**：独立重算匹配，3 次重建哈希相同；同 seed 模拟相同。|
15|三类及多类图是否在模拟中平衡？|**Partially answer**：已统计 500×2 分布，最终选择未冻结。|
16|评价集是否完全不含离线增强副本？|**Cannot answer**：正式评价集尚未建立。|
17|最终测试是否与模型选择完全隔离？|**Cannot answer**：协议要求如此，但测试集未冻结。|
18|1124/2573/2754 的统计差异原因是什么？|**Cannot answer**：已登记，来源口径未证实。|
19|跨设备、年份或光照的独立性如何证明？|**Cannot answer**：缺逐图采集属性/来源组。|

**建议的下一阶段（只作建议，不自动进入）**：先由用户审核上面 5 个决策和桥边/候选关系，再决定是否授权针对少量争议样本的来源核查与正式 split 准入审查。当前停止在 Stage 2B-4V。

## Stage 2B-4H 裁决登记（覆盖上述待决项，不回写历史模拟）

- C01/C02：用户接受同源，合并各自两组；人工复核后关系为Strongly Supported，**并非Confirmed physical acquisition**。命名跨split家族现为67 Strongly Supported、0 Candidate。
- B01/B02：用户接受两条视觉桥并保留各自完整现有component；对应边升为human-reviewed Strongly Supported。S01–S10亦全部接受，组成员不变、仍为Strongly Supported；缺 acquisition ID，均不升级Confirmed。
- D01–D03：同字节图像重复为Confirmed并保持同组；polygon轻微不同，标注冲突仍Unresolved。不得平均、修补或默选某一标注版本。
- R01–R03：认定同名original/resize是same-source companion，但确定性缩放等价Unresolved；**正式实验排除全部数据集提供的resize表示**，训练管线动态处理输入尺寸。Raw Data不变。
- U01：`test/IMG_54350.jpg#3`不赋成熟类；派生三类监督标注不把此region当作有效训练/评价目标，保留原图及其他3个有效region。评价器如何忽略此区域仍须实现验证。

当前关系边计数为Confirmed **10**、Strongly Supported **393**、Supported **1426**、Candidate **33**、Unresolved **3**；其中跨名dHash Candidate 仍有23条跨当前组。旧版阈值敏感性及500×2模拟数值仅描述**裁决前1114组图**，本次没有重新模拟、更没有选正式比例或seed。余下Freeze阻断包括可信正式split、fruit/tree/session与跨组候选来源不确定性、同像素不同polygon的派生GT规则、U01的评价实现验证、来源侧数字差异及视觉阶段标签的主张边界。**Stage 2B-4H已记录，Stage 2B-5未开始。**
