# D2 v4 — Source-Group Resolution（Stage 2B-4）

更新：2026-09-20。**D2 = NOT FROZEN**。本轮只读 `dataset-20260508.zip` 与既有 Stage 2B-3 CSV，建立候选同源图和可复算清单；没有修改 raw、正式划分或模型。原包 SHA256：`049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`。

**Stage 2B-4H 当前状态（覆盖下文裁决前1114组快照）**：用户接受C01/C02同源并将其关系定为Strongly Supported，也接受B01/B02桥边及S01–S10；重跑生成器得到**1112组、67组跨官方split（204张非resize表示）**。裁决输入保存在`IOR-YOLO/configs/data/d2_source_group_adjudications.json`，最新组号见`source_groups.csv`。下文1114组、65跨集组及Candidate 2的数字保留为**裁决前历史结果**，不得当作当前状态。

## 口径与产物

图的独立节点是 ZIP 中 **1406 张非 `_resize` 表示**，但它们包含离线增强和重复，**不是 1406 个独立采集样本**。同名 `_resize` 版本附着于相应节点，不再计作独立样本。`source_group_id` 是保守的**证据支持候选分组**，不是 fruit/tree/session ID；无法排除不同视角、同树或连续帧的剩余相关性。

| 资产 | 用途 |
|---|---|
`IOR-YOLO/data/manifests/source_groups.csv` | 2812 行，含 original/resize 的组号、canonical representative、关系、证据、置信度、类别、区域数和尺寸。 |
`IOR-YOLO/data/manifests/source_group_relations.csv` | 1865 条关系边，逐条给出 relation_type、evidence、confidence 和是否用于合并。 |
`IOR-YOLO/data/manifests/source_family_review.csv` | **67 个**跨官方 split 命名家族逐组状态、成员、最强证据和剩余不确定性；不以家族名直接证明采集身份。 |
`IOR-YOLO/scripts/06_build_source_groups.py` | 稳定排序、确定性连通分量编号；重跑相同输入不依赖 Python hash 或随机 UUID。 |

重算命令（在项目根目录）：

```zsh
uv run --with pillow python IOR-YOLO/scripts/06_build_source_groups.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/data/manifests
```

## 分组规则与结果

原始图像间仅合并 Confirmed、Strongly Supported、Supported 的边。Confirmed = 解压后字节 SHA256 相同；Strongly Supported = 明确增强后缀家族、64×64亮度相关≥0.995且 polygon 列表相同；Supported 的跨名视觉边要求 dHash 筛查、64×64相关≥0.995，另有全分辨率灰度相关≥0.96。后两种仍不是已证明的图像生成链。Candidate 和 Unresolved 边**不自动合并**。其中同 split 后缀家族也经像素和 polygon 检查，而非仅凭名称合并。`_resize` 与 original 同名且一一对应；3 对异常虽暂随对应 original 记入同一清单组，**关系置信度仍为 Unresolved，不据此声称像素等价**。

| 指标 | 本轮结果 |
|---|---:|
非 resize 图像 | 1406 |
证据支持的候选 source groups | **1114** |
组大小 1 / 2 / 3 / 4 / 6 | **949 / 60 / 95 / 4 / 6** |
最大组 | 6 张 original |
跨官方 train/val/test 的组 | **65 组、200 张 original** |

图边的**数量不是独立泄漏事件数**：Confirmed 10（3 个跨 split 精确重复组、7 个同 split），Strongly Supported 389，Supported 1428（1403 original↔resize、25 跨名视觉），Candidate 35，Unresolved 3（异常 resize）。重复关系和传递路径可能在同一组内反复出现。

## 67 个跨 split 命名家族

逐组列表见 `source_family_review.csv`。最终为 **Confirmed 0、Strongly Supported 65、Supported 0、Candidate 2、Unresolved 0**。65 的最强跨集关系兼有高亮度相关和完全相同 polygon；不意味着同一家族的所有配对都同样可靠。`img_13960`（相关度 0.994468）与 `img_14040`（0.990993）虽共享 polygon 且文件名含 `_brightness`，仍低于既定 0.995 阈值，保留 Candidate，分属不同组。`172` 家族也有部分较弱边：train 的 `172_brightness.jpg` 与 val 的 `172_noise.jpg` 在同一组，test 的 `172.jpg` 暂未并入。正式划分前应人工复核这些边。

## 相同字节、不同标注

以下 3 对跨官方 split 文件 SHA256 完全一致，必须同组。其类别、区域数相同，但 polygon 顶点数和坐标不同；**图像重复 = Confirmed，annotation consistency = Unresolved / inconsistent**，不得改 raw 标签：

| 图像对 | 类别与 region | polygon 差异 |
|---|---|---|
`test/138.jpg` ↔ `train/251.jpg` | 各 1 个 semi-mature | 45 对 39 顶点；逐序号比较最大坐标差 104 像素（顶点数量不同，不把它解释为对应点误差） |
`train/138_brightness.jpg` ↔ `val/251_brightness.jpg` | 各 1 个 semi-mature | 同为 45 对 39 顶点，polygon 列表不同 |
`train/398.jpg` ↔ `val/166.jpg` | 各 2 个 immature | 第 1 区域 41 对 46 顶点，第 2 区域 11 对 14 顶点；polygon 列表不同 |

## dHash 与来源图边界

Stage 2B-3 的 25 对 Supported 跨名关系经全分辨率灰度相关复核（本包范围约 0.970–0.998）后仍为 Supported；**仅 6 对**使原本不同的连通分量合并，另外 19 对在已有关系路径内。26 对 Candidate 中，3 对两端已因其他证据同组，**23 对仍在不同组**，不以 dHash 距离单独合并；模拟 split 可能切断其中部分边。这里的阈值是审计筛查规则，不是来源真值证明。缺乏采集 ID，不能宣称已经消除 fruit/tree/session 泄漏。

## canonical 表示与三对异常

`canonical_sample` 优先选组内无明确增强后缀的 original 表示，再稳定按 split/filename 排序；只是**代表文件**，不是核实过的原始拍摄文件。建议后续训练/评价以非 resize 表示为图像候选池，训练管线自行 resize；不把 1406+1406 当 2812 独立样本。正式评价集是否去除随包离线增强图，还需先决定并记录。

| original → resize | SHA256（original / resize） | 尺寸 | 亮度相关 / LANCZOS RGB MAE | polygon |
|---|---|---|---|---|
`train/172_brightness.jpg` | `6d1bd026…` / `7729b804…` | 277×326 → 480×640 | 0.772601 / 39.1184 | 各 1 区域；缩放坐标最大残差约 27.66 px |
`val/172_noise.jpg` | `8f6a897f…` / `95e0c384…` | 277×326 → 480×640 | 0.768708 / 36.4767 | 各 1 区域；最大残差约 27.66 px |
`train/327.jpg` | `8810b177…` / `5b7f52c6…` | 327×254 → 640×480 | 0.848415 / 31.3109 | 各 2 区域；最大残差约 6.43 px |

三对均能完整解码，有同名 resize 和 JSON 记录，均无 EXIF 字段；`172` 另有不同 split 的同名族图，`327` 在本包无其他明确后缀同名族图。是否裁剪、非等比变换、另一次处理或文件配对错误仍 **Unresolved**。本轮不选择性修复或删除。

## 无类别 region

`test/IMG_54350.jpg#3`（及 resize 对应区域）具有有效 4 顶点 polygon，原始面积约 39 px²，但 `region_attributes={}`，图像级 `file_attributes={}`，无可恢复类别。原始三份 JSON 的 **2575 regions = 2574 个有效三类 + 1 个 Unknown**；2573、2754 及 1124 的来源口径继续 Unresolved。

| 方案 | 训练与评价影响 | 结论 |
|---|---|---|
保留图像、直接丢弃该 polygon | 未标目标可能被当背景或在 mAP 匹配中计为预测错误 | 不宜无条件采用 |
保留图像，派生标注将该 polygon 标为 ignore region | 保持其他实例可用；需要框/掩膜评估器明确支持 ignore，并验证与邻近果重叠规则 | **首选候选协议**，实施前写测试 |
只在派生标注删除该 polygon | raw 不变，但评价仍有未标目标偏差 | 不能等同于解决 |
若框架无法正确 ignore，从计分用 val/test 排除此**整张图**并报告数量 | 可避免评价中的未知目标，但减少一张图；训练端仍需处理 | 备选，需预注册并做敏感性报告 |

本轮未生成派生 annotation 或作此正式决定。

## 限制与下一步

官方 split 因精确跨集重复不能用于正式 Baseline/最终评价。1114 组是执行 group-aware 方案的**工作图**，尚未解决 2 个命名 Candidate、23 个跨组 dHash Candidate、3 对 resize 异常、相同像素不同 polygon、缺失类别和采集 ID 问题。划分方案及模拟见 `docs/split-protocol-d2.md`；**未创建正式 split**。

**Stage 2B-4V 复核补注（2026-09-20）**：从 ZIP/JSON 独立重建的图与本清单 1114 个组的成员和组号完全一致；这只验证现有规则的可复算性，不使组成为真实采集身份。两条 Supported 跨名视觉桥和命名相关阈值敏感性见[审核包](review-packet-stage2b4.md)。本页 Stage 2B-4 结论仍为 Candidate / Review Pending。

## Stage 2B-4H 人工裁决与重建（当前）

人工决定见[逐案复核表](human-review-stage2b4.md)。C01 `img_13960`、C02 `img_14040` 的跨split边由Candidate改为**Strongly Supported**并并组；C01两端同属`sg-0583`，C02两端同属`sg-0588`。B01 `test/3100.jpg ↔ train/1640.jpg`、B02 `test/1270.jpg ↔ train/2340.jpg` 原本已并组，本次保留整组并把**对应边**由Supported改为**Strongly Supported**。S01–S10保持原组及Strongly Supported；这些人工裁决均**不宣称Confirmed physical acquisition**。

| 重新生成后的指标 | 当前值 | 裁决前 |
|---|---:|---:|
| 非resize图像 | 1406 | 1406 |
| source groups | **1112** | 1114 |
| 大小1 / 2 / 3 / 4 / 6的组 | **945 / 62 / 95 / 4 / 6** | 949 / 60 / 95 / 4 / 6 |
| 跨官方split组 / 涉及original表示 | **67 / 204** | 65 / 200 |
| 67个命名跨split家族：Strongly Supported / Candidate | **67 / 0** | 65 / 2 |
| 关系边：Confirmed / Strongly Supported / Supported / Candidate / Unresolved | **10 / 393 / 1426 / 33 / 3** | 10 / 389 / 1428 / 35 / 3 |

关系边计数包含1403对常规original↔resize及3对异常伴随关系；它们不是独立图像或独立泄漏事件。余下33条Candidate**关系边**中仍有26条跨名dHash候选，23条两端跨当前组；不能把“命名家族Candidate 0”解释为无任何来源不确定性。组号随确定性排序重建，旧组号只用于历史证据追踪。

D01–D03的相同SHA256图像必须同组，图像重复为Confirmed；polygon有轻微差异，**annotation conflict仍Unresolved**，不平均、不修复、不指定某一版本为真值。R01–R03为人工接受的**same-source companion**，但“纯确定性resize”等价仍Unresolved；正式实验排除**全部dataset-provided `_resize` 表示**，模型输入尺寸由训练管线动态resize，Raw Data不变。U01的`IMG_54350.jpg#3`为invalid/unknown region，派生三类监督标注不将其作为有效训练或评价目标；保留同图其他3个有效region和整张图，原始JSON不改。未来派生标注/评价实现仍需验证未标区域不会被误当背景或计分目标。

本次只重建同源清单，**未创建正式split或选择seed**。fruit/tree/session ID仍缺失，D2 **NOT FROZEN**。
