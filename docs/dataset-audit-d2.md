# D2 v4 — Raw ZIP and Integrity Audit

更新：2026-09-19。**D2 = NOT FROZEN**。这是对用户取得的 `dataset-20260508.zip` 的低成本只读审计：读取ZIP目录、六份VIA JSON和JPG图像头，不解压、不修改原包、不解码全图、不重划分。取得来源、许可及SHA256见[获取记录](data-acquisition-d2.md)。`Confirmed`表示可由本次原包直接复核；`Supported`表示多条线索一致但尚缺像素/采集来源证明；`Unresolved`表示尚无足够证据解释或排除。

**当前补注（2026-09-20，Stage 2B-4H）**：下文分阶段保留原始审计与裁决前1114组快照；用户人工裁决后最新source-group清单为**1112组**。21案决定、关系分级与实验处置见[本页末尾补注](#stage-2b-4h-人工裁决后的数据审计状态)及[人工复核表](human-review-stage2b4.md)。原包和原始标注未变，D2仍NOT FROZEN。

**Stage 2B-5P候选协议补注**：不更改原始审计与1112个source groups，新增[split guard与候选池](split-constraint-design-d2.md)和[Freeze Candidate Gate](dataset-freeze-candidate-d2.md)。本轮在全包同SHA标注检查中新增发现7条**同一官方split内**的polygon冲突关系；连同已知3条跨split关系，共10条，涉及7个source groups。下文早期“3组跨集标注冲突”仅指先前审核范围，不能视为全包冲突总数。Raw Data未修改，D2继续NOT FROZEN。

## 审计范围与复现

- 原包：`IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip`，290131787 bytes；SHA256 `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce` 为用户首次验证记录，本次结构/内容审计未重新计算哈希。ZIP有2826个成员，不等于图像/实例数。
- 本次执行：`python3 IOR-YOLO/scripts/04_audit_d2_zip.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip`。脚本只读图像头和六份JSON，按实际`filename`和`regions`统计；合成PNG/JPEG及跨split命名测试通过。图像头有效不代表像素完全可解码。
- 内部根目录`dataset-20260508/`；六个图像目录`train/`、`val/`、`test/`及各自`_resize/`；六份同名JSON及`ReadME.txt`。2826 = 2812个JPG + 6个JSON + 1个README + 7个显式目录成员。**Confirmed**。
- 下文`original`仅指不带`_resize`的文件夹，**不保证其中每张照片都是未经离线增强的采集原图**。

## 图像和标注矩阵

| 目录 | JPG文件成员/唯一文件名 | JSON顶层图像记录 | 区域/实例条目 | 类别实例（immature / semi-mature / mature / 无有效类别名） | 图像头尺寸（宽×高：张） |
|---|---:|---:|---:|---|---|
| train | 1041 / 1041 | 1041 | 1981 | 713 / 618 / 650 / 0 | 369×277: 540；404×303: 406；277×369: 93；277×326: 1；327×254: 1 |
| val | 126 / 126 | 126 | 185 | 67 / 60 / 58 / 0 | 369×277: 66；404×303: 42；277×369: 17；277×326: 1 |
| test | 239 / 239 | 239 | 409 | 141 / 136 / 131 / 1 | 369×277: 119；404×303: 104；277×369: 14；303×404: 2 |
| **original合计** | **1406 / 1406** | **1406** | **2575** | **921 / 814 / 839 / 1** | 常见横图、竖图及3张非常规尺寸图 |
| train_resize | 1041 / 1041 | 1041 | 1981 | 713 / 618 / 650 / 0 | 640×480: 947；480×640: 94 |
| val_resize | 126 / 126 | 126 | 185 | 67 / 60 / 58 / 0 | 640×480: 108；480×640: 18 |
| test_resize | 239 / 239 | 239 | 409 | 141 / 136 / 131 / 1 | 640×480: 223；480×640: 16 |
| **resize合计** | **1406 / 1406** | **1406** | **2575（对应条目，不能与original相加当独立实例）** | **同original** | 640×480: 1278；480×640: 128 |

以上文件名数、JSON记录数、区域数、类别字段值和图像头尺寸为 **Confirmed**。每个目录内没有重复JPG文件名；JSON逐目录均与JPG文件名集合相等，无空`regions`、无孤儿图或孤儿记录；六目录图像头读取均未报错。各区域`shape_attributes.name`均为`polyline`，本次未发现顶点列表结构缺失、零面积或超出图像宽/高超过一个单位的多边形；但这不等于完成全图解码或人工标注质量审查。

**类别缺口（Confirmed）**：`test.json`的`IMG_54350.jpg`区域索引`#3`（第4个区域）有`region_attributes={}`；`test_resize.json`同一对应区域也如此。1406张original的2575个区域中仅2574个有三种有效阶段名称。这解释“2574”可作为**已分类区域数**的一种本地计数口径，但不能据此猜测README、论文或Mendeley各数字的编写原因，也不能静默删除/补标该区域。

**坐标边界（Confirmed）**：原始train/val/test分别有93/20/20个区域的顶点恰好等于宽或高；resize分别为55/16/15。未发现顶点超出宽/高超过一个单位；宽、高处的顶点在连续多边形坐标约定下可能处于边界，是否需用于未来转换时调整仍 **Unresolved**，不能据此把133个原始区域判作损坏并删除。

**尺寸例外（Confirmed）**：`train/172_brightness.jpg`与`val/172_noise.jpg`的头尺寸均277×326，`train/327.jpg`为327×254；另有常见分辨率的竖向版本。resize不是全部640×480：竖图为480×640。尺寸不同的原因及是否包含裁剪/方向变化 **Unresolved**。

## Original ↔ resize与官方split

| 检查 | train | val | test | 证据等级 |
|---|---:|---:|---:|---|
| 与对应resize同名JPG | 1041/1041 | 126/126 | 239/239 | **Confirmed**，双分辨率不能当独立样本 |
| 与对应resize同名JSON记录 | 1041/1041 | 126/126 | 239/239 | **Confirmed** |
| 逐图区域数及逐区域类别序列不同 | 0 | 0 | 0 | **Confirmed** |
| 按图像头宽/高比例对应，逐区域最大顶点残差>2像素 | 3个区域 | 1个区域 | 0 | **Confirmed**；仅3个文件：`172_brightness.jpg`、`327.jpg`、`172_noise.jpg` |

超过2像素的最大残差约27.6564像素（两张`172`派生文件）和6.4251像素（`327.jpg`）。其余对应区域最大顶点残差≤2像素，**Supported**为按分辨率缩放的同一标注；对上述三张的变换方式 **Unresolved**，不能笼统声称“全部坐标仅按resize比例缩放”。尚未逐像素证明所有同名图确属同一原图。

**现有划分的数量（Confirmed）**：original的train/val/test为1041/126/239，比例按1406张计为74.04%/8.96%/17.00%；`ReadME.txt`声称70%/10%/20%，与本包不一致。原始三个split之间**完全相同文件名和未归一化stem交集均为0**，但这不能排除增强派生图跨split。

去掉末尾`_brightness`/`_noise`等明确增强后缀，再按原始三个split比较，有**train–val 47组、train–test 30组、val–test 14组**共同来源名候选；按三集合去重共**67个跨split候选来源名族**。其中分别**47/29/13组**能在两split中找到**完全相同的`regions`标注列表**。例如train中的`111_brightness.jpg`与val中的`111.jpg`；train中的`1290_brightness.jpg`与test中的`1290.jpg`。这些相同命名及多边形标注为潜在离线增强跨集提供**强支持（Supported）**，但未执行像素级/元数据核对，暂不把每组判为已证实的同一采集图像。未归一化文件名互不相同不能证明官方split可信；fruit/tree/session独立性仍 **Unresolved**。此次没有重建split或修改数据。

## 数字不一致与主张边界

| 外部/随包说法 | 本包证据 | 状态 |
|---|---|---|
| Mendeley v4“1124 apple images”；同页“1406 annotated images”；论文“1406 RGB images”；README“1406 RGB images” | 原始目录1406 JPG+1406记录，resize另有1406个对应JPG | **Confirmed**本包1406组同名原始/resize；**Unresolved**“1124”的统计口径/差额，不以287个`_brightness`/`_noise`命名文件推测原因 |
| README Key Features“2754 annotated instances” | 本包original JSON共2575区域，其中2574有有效三类名称 | **Unresolved discrepancy**；不推测2754如何形成 |
| README Statistics“2574 instances”，分类921/814/839；论文2574；Mendeley v4“2573” | 三类已命名数量严格为921/814/839（合计2574），另有1个未命名区域 | **Confirmed**本包两种计数口径；**Unresolved** Mendeley的2573以及各来源差异原因 |
| README“70%/10%/20%” | 本包74.04%/8.96%/17.00% | **Confirmed discrepancy**；不猜测目标比例与实际比例差异原因 |
| README“640×480”及论文常见369×277/404×303 | 本包还包含竖向480×640、277×369/303×404及三张非常规尺寸 | **Confirmed**尺寸范围比概述更广 |

来源：`dataset-20260508/ReadME.txt`及六份JSON和图像头；[Mendeley Data v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[Data in Brief DOI](https://doi.org/10.1016/j.dib.2026.112856)。网页/论文数字是其作者表述，本包审计数字是另一个层次，不能静默择一或重写原始标签以凑数。

## 风险、后续检查与停止点

| 检查项 | 证据与风险 | 后续必要行动（本轮不执行） |
|---|---|---|
| 离线增强跨集 | 67个命名族跨split候选；大量完全相同区域标注；现有官方划分的独立性不可信 | 核对原图/派生图像素及source group，审查是否存在更多未命名派生关系；正式协议之前确定隔离策略 |
| 视觉标签真值 | 真实JSON为三种英文阶段名，一条缺失；颜色阶段不是生理成熟测量 | 保留缺口，定义可追溯的处理规则并报告影响；不得私自补标 |
| 分辨率/坐标 | 原始存在非常规图，竖向resize及三个坐标残差较大文件；图像边界约定待确定 | 图像解码、局部可视核对、坐标约定审查；不改raw |
| 腐坏、近重复、果实级泄漏 | JSON结构与图像头通过不等于完整像素解码；无fruit/tree/session ID确认 | 后续分批只读内容与近重复/采集分组核查，不能声称无泄漏 |

**D2 = NOT FROZEN**。包可读取并有三阶段逐果polygon，但标签缺口、增强跨split候选与坐标例外仍阻止信任现有官方划分或开展正式Baseline。本轮到只读审计为止，不创建新split，不训练，不改原始数据。

## Stage 2B-3 — Leakage & Annotation Integrity Audit（2026-09-19）

本节更新上文Stage 2B-2的“尚未全图解码”和“潜在泄漏”状态；上文作为阶段记录保留。**D2 = NOT FROZEN**。使用临时隔离的Pillow 12.3.0运行环境，未变更项目环境；脚本只读ZIP并写小型CSV，不解压或修改raw。

```zsh
uv run --with pillow python IOR-YOLO/scripts/05_audit_d2_integrity.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/data/manifests
```

逐成员读取时校验ZIP CRC、计算解压后文件SHA256，并用Pillow执行`verify()`及完整像素`load()`。**Confirmed：2812/2812张成功完整解码**（original 1406、resize 1406）；损坏、截断、不可读、格式不符各为0。本轮独立复算原包SHA256仍为`049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`。解码成功不证明视觉标签准确。

### 可追溯清单与候选图

| 文件 | 内容与边界 |
|---|---|
| [files_sha256.csv](../IOR-YOLO/data/manifests/files_sha256.csv) | 2812行；split、variant、filename、stem、extension、解码尺寸、字节数、成员SHA256、source_family、candidate_group_id及解码状态。`source_family`仅去掉stem末尾**一个**明确的`_brightness`/`_noise`/`_gaussian`/`_hsv`/`_gamma`后缀并转小写，不按数字猜同源。 |
| [duplicate_report.csv](../IOR-YOLO/data/manifests/duplicate_report.csv) | 1600条**关系边**，包括1406个original↔resize配对、命名家族跨split关系、所有原始图像精确SHA重复及轻量dHash候选。逐行记录relation_type、evidence、status和confidence；行数不等于独立样本或泄漏家族数。 |
| [class_counts.csv](../IOR-YOLO/data/manifests/class_counts.csv) | split×variant的JSON类别区域数。resize是同图像另一表示，不重复计为独立实例。 |

候选图规则：同`source_family`、相同原始图像SHA256及`Supported`的跨名视觉关系连边；`Candidate`/`Unresolved`的跨名视觉边不自动合并。resize按同split+filename附在原始节点上。**Confirmed脚本统计**：1406张非resize图构成1107个候选连通组，67组跨现有split，涉及205张图，最大组6张。这是**审计候选图，不是来源真值或新划分**；命名边仍需复核，fruit/tree/session关系未知。

### 官方split泄漏证据

Stage 2B-2的**67个命名候选家族**按其最强跨split关系分级：**Confirmed 0、Supported 65、Candidate 2、Unresolved 0**。Supported表示至少一对64×64亮度结构相关度≥0.995；多数组同时有相同polygon列表，但尚非确定的增强变换证明。Candidate为`IMG_13960`与`IMG_14040`，相同polygon、相关度约0.994468与0.990993，未达保守阈值。家族级Supported不表示内部每条边均成立，例如`172`家族train↔val高相似、与test的关系仍为Candidate。

**这67个家族之外，至少3组跨split图像已确认泄漏**：不同文件名但**解压后SHA256完全相同**。

| 跨split文件对 | SHA256前缀 | 标注 |
|---|---|---|
| `test/138.jpg` ↔ `train/251.jpg` | `7d579f7f…` | 各1区域、类别相同，polygon列表不完全相同 |
| `train/138_brightness.jpg` ↔ `val/251_brightness.jpg` | `f81d8e3a…` | 各1区域、类别相同，polygon列表不完全相同 |
| `train/398.jpg` ↔ `val/166.jpg` | `32e6e897…` | 各2区域、类别相同，polygon列表不完全相同 |

相同SHA256足以确认跨集图像字节重复；标注几何差异意味着不能只去重图像而不审查标签。非resize图另有7组只在同一split内的精确重复。**官方split不得直接用于正式Baseline或最终评价。**

由于跨名精确重复证明文件名规则不足，才对1406张非resize图增加**256-bit dHash**轻量筛查：跨split不同source_family且汉明距离≤12的候选，再用64×64亮度相关度复核；没有用视觉大模型。剔除精确SHA关系后，额外得到**25对Supported**（相关度≥0.995）与**26对Candidate**。它们是**配对数**，不可与65个家族或3组精确重复相加；低于阈值的边不并入候选图。仍无法排除所有未命名近重复及同果不同视角/同树/同场次关系。

### Original ↔ resize与标注完整性

1406对文件名、JSON记录、逐图区域数和类别序列一一对应。完整解码后，以LANCZOS重采样的RGB平均绝对差（MAE）与64×64亮度相关度筛查。JPEG重编码会造成非零像素差，**相似不等于已确定具体resize算法**。1403对满足相关度≥0.995且MAE≤20，属于**Supported的同一样本不同表示**；其余3对均可解码，但“只做resize”仍**Unresolved**：

| original ↔ resize | 尺寸 | 相关度 / MAE | 区域数 |
|---|---|---|---:|
| `train/172_brightness.jpg` | 277×326 → 480×640 | 0.772601 / 39.1184 | 1 ↔ 1 |
| `val/172_noise.jpg` | 277×326 → 480×640 | 0.768708 / 36.4767 | 1 ↔ 1 |
| `train/327.jpg` | 327×254 → 640×480 | 0.848415 / 31.3109 | 2 ↔ 2 |

这正是Stage 2B-2的三张异常尺寸图；对应坐标缩放最大残差约27.66、27.66、6.43像素。原始和resize均为可解码JPEG，均无EXIF Orientation字段。不能据此推断正常采集差异、裁剪、配错图或其他处理原因，也不能把全部1406对无条件当作等价表示。

**无类别区域（Confirmed）**：`test/IMG_54350.jpg#3`及resize对应区域的`region_attributes={}`，图像级`file_attributes={}`，无其他可恢复类别字段。original多边形为4顶点、非零面积约39像素²；几何结构有效不代表语义可靠。保持**Unlabeled / Unknown**，不补标或删除。**2575 = 全部polygon regions；2574 = 三类有效标签regions；1 = 无类别region**。Mendeley的2573与README Key Features的2754继续记为**Unresolved source-side discrepancy**，不推测原因。

### D2 Freeze Gate

| 条件 | 当前判断 |
|---|---|
| 1 source/version；2 license；3 raw SHA256 | **通过**：官方v4/CC BY 4.0记录与本地SHA256复核。 |
| 4 package可解析；5 image/annotation/instance统计 | **包级通过**：六份JSON可解析，2812图完整解码，1406非resize图、2575区域/2574有效类已核实；不等于标注无瑕疵。 |
| 6 来源数字差异解释或正式登记 | **已登记但原因未解**：1124、2573、2754及比例差异。 |
| 7 三阶段标签语义 | **部分通过**：是视觉颜色阶段，不是生理成熟真值；1个region无类别。 |
| 8 离线增强关系；9 双分辨率/重复关系 | **部分通过**：67个命名家族、3组跨集精确重复、25对新增强支持跨名关系；1403对resize强支持，3对异常待解。 |
| 10 无未知严重annotation问题 | **未通过**：缺失类别、同字节图的polygon不完全一致、resize坐标/内容例外。 |
| 11 可信split前提 | **未通过**：官方split已确认泄漏，另有Candidate边与未知fruit/tree/session关系。 |
| 12 支持检测任务 | **任务形式通过**：逐果polygon与阶段字段存在；正式协议仍取决于完整性与分组处理。 |

**停止点**：本轮只读审计，不生成新train/val/test、不改raw、不训练。候选图足以开始**设计**Group-Aware Split规则，但尚不足以执行或冻结可信split；下一阶段需复核候选边、跨名高相似图、标签/坐标例外与可用采集元数据。

## Stage 2B-4 — Source Group 与划分协议设计（2026-09-20）

本节是对 Stage 2B-3 **初步候选图** 的更保守复核，不回写旧审计记录。项目级 `apple-dataset-audit` 规则下，仅只读原始 ZIP 与现有 manifests；新增 [同源组解析](source-group-resolution-d2.md)和[划分协议模拟](split-protocol-d2.md)。**D2 = NOT FROZEN**，无正式 split。

1406 张非 resize 图像形成 **1114 个证据支持候选组**（大小 1/2/3/4/6 组数为 949/60/95/4/6），**65 组、200 张图**跨官方 split。之前的 1107 组/67 跨集组采用更宽的命名并组规则；本轮将仅凭命名/几何但像素阈值不足的边保持 Candidate，故数字改变，并非原始数据变化。67 个跨 split 命名家族最终为 Strongly Supported 65、Candidate 2；3 个跨 split 不同名 SHA256 精确重复组继续确认官方 split 泄漏。25 对高相似 dHash 关系中 6 对实际连接原本分开的组；26 对 Candidate 中 23 对两端仍在不同组。全图 2812/2812 可解码、2575/2574/1 区域计数不变。

### D2 Freeze Gate 再评估

| 条件 | 状态 | 依据与剩余问题 |
|---|---|---|
1 source/version | **PASS** | 官方 Mendeley Data v4 与原包身份已登记。 |
2 license | **PASS** | 官方数据许可 CC BY 4.0 已登记，正式使用需署名。 |
3 raw SHA256 | **PASS** | 本地 `dataset-20260508.zip` 的 SHA256 已两次核对。 |
4 package readability | **PASS** | 2812/2812 图完整解码，六份 JSON 可解析。 |
5 image/annotation counts | **PASS** | 1406 original + 1406 resize；2575 区域、2574 有效标签、1 Unknown。此项只确认统计，不证明标注正确。 |
6 source-side numeric discrepancies | **UNRESOLVED** | 1124、2573、2754 及 README 比例差异已登记，但来源口径未解。 |
7 label semantics | **PARTIAL** | 三阶段按视觉颜色标注，不是生理成熟真值；Unknown region 待处理。 |
8 augmentation relations | **PARTIAL** | 65 命名家族强支持派生关系，2 Candidate；仍缺真正采集来源 ID。 |
9 duplicate relations | **PARTIAL** | 3 跨集 SHA 重复已确认；25 跨名视觉边支持，23 跨组 dHash Candidate 和 3 对 resize 异常待核。 |
10 annotation integrity | **PARTIAL** | 同像素图 polygon 不一致、Unknown region、异常坐标已定位；派生协议与评价影响未验证。 |
11 source group reliability | **PARTIAL** | 1114 个可复算候选组，但 Candidate 边及 fruit/tree/session 关系未知。 |
12 credible split prerequisites | **PARTIAL** | 两比例 dry-run 可保持已纳入图的关系不跨集；正式评估图像选择、ignore 规则、采集分组与 Candidate 审核未完成。 |

**停止点**：官方 split 不可用于正式 Baseline；候选 group-aware 协议只达到 Simulation Only。D2 保持 **NOT FROZEN**；不生成正式 train/val/test、不训练。

**Stage 2B-4V 独立复核（2026-09-20）**：重新从 ZIP/JSON/哈希/像素建图，组数、成员与组号和现有 `source_groups.csv` 完全一致；阈值、桥边、Candidate 残余风险及500×2次模拟分布见[审核包](review-packet-stage2b4.md)与 `IOR-YOLO/reports/dataset_audit/verification/`。这些是候选图的可复算性证据，**不是来源真值或 Dataset Freeze**。未进入 Stage 2B-5。

## Stage 2B-4H 人工裁决后的数据审计状态

用户接受C01/C02同源关系并明确只评为Strongly Supported；C01、C02各合并两组。B01/B02视觉桥得到人工接受，保留原有完整component，**对应边**升为Strongly Supported；S01–S10全部接受，仍为Strongly Supported，均不宣称物理采集身份Confirmed。生成器从原ZIP、Stage 2B-3清单和[裁决输入](../IOR-YOLO/configs/data/d2_source_group_adjudications.json)重建出**1406个非resize表示、1112个候选组、67个跨官方split组（204张表示）**。67个命名家族现全为Strongly Supported；另有23条跨组dHash Candidate，来源风险未消失。图像/区域统计仍为2812张可解码图、2575个原始polygon、2574个有效三类标签和1个无类别region。

三个不同名、跨split、SHA256完全相同的图像组维持**Confirmed duplicate**并必须同组；其polygon存在轻微差异，**annotation conflict = Unresolved**，不修改、平均或选一套为默认真值。R01–R03被人工认定为same-source companion，但valid deterministic resize equivalence仍Unresolved；**全部数据集提供的`_resize`表示不进入正式实验**，模型输入大小以后由训练管线动态处理。U01的`test/IMG_54350.jpg#3`判为invalid/unknown region，派生三类训练/评价标注不把它当有效目标；保留该图其余三个有效区域及整张图，原始JSON不变。派生标注尚未生成，评价器对未知区域的忽略语义仍需验证，不能把这一规则误述为已实现。

**Freeze Gate仍未通过**：官方split确认泄漏且尚无正式替代；fruit/tree/session ID缺失，23条跨组候选关系未裁定；同图不同polygon的派生GT规则尚未定/验证；U01的评价实现未验证；来源侧1124/2573/2754等数字差异、视觉阶段与生理成熟的语义边界仍需在最终数据/论文协议中处理。用户已决定排除dataset-provided resize，这解决输入表示选择，但没有证明那三对异常的生成机制。**D2 = NOT FROZEN；未生成正式split、未选seed、未进入Stage 2B-5。**

## Stage 2B-5P 候选协议补充审计

按全部1406张非resize的SHA与原始JSON确定性检查，确认同字节不同polygon的**10条SHA重复关系/7个source groups**：原有D01–D03为3条跨官方split关系，新增7条在同一官方split。其region数与类别序列在每对中一致，polygon几何不同，原因未查明。候选Policy B将这7组暂从训练与评价池排除，不改Raw Data、不按文件名暗选真值；它是**待确认的协议建议**。候选池不使用1406张dataset-provided resize；非resize中plain/base 1119、brightness 170、noise 117；6个source groups无明显base，已明确登记fallback但暂不启用。每个有base组只留一个稳定代表，再暂存7个冲突组，得到**1099张候选图**。详见[候选池逐行记录](../IOR-YOLO/data/manifests/d2_experiment_pool_candidate.csv)。

23条跨source-group dHash Candidate及另外6条跨组命名Candidate不修改source group，而进入1096个[split guard clusters](../IOR-YOLO/data/manifests/split_guard_clusters.csv)的预防性约束。已列关系在两次**单seed候选模拟**中均零跨模拟split；这不证明未知采集关系不存在，也不等于已建立正式split。U01模型无关评价探针显示：保留3个合法目标并不自动消除#3处预测，标准无ignore规则会把该处苹果预测计为FP；真实框架和语义仍需进一步验证。来源侧1124/2573/2754冲突的原因仍不明，但固定ZIP的1406/2575/2574/1计数已可复算，故将其作为**Documented source inconsistency**，不再单独作为阻断实验准备的硬条件。**D2 NOT FROZEN；未进入正式Stage 2B-5 Freeze。**

## Stage 2B-5 正式冻结审计（覆盖上文各阶段当时的“NOT FROZEN”状态）

用户正式确认candidate guard、严格Policy B、6无base组排除、全部离线增强及dataset-provided resize排除、U01三有效目标加普通背景FP规则，以及70/15/15与固定ZIP+上下文seed。两次独立构建从不变原包重建出的[协议和manifest](dataset-freeze-d2.md)逐字节一致。全图关系形成1112个source groups与1096个guard clusters；正式池1099图、1099有图组、1088有图guard clusters；split图数769/165/165，三类有效实例总数915/425/732。1713条表示排除逐行留存，其中冲突组7个、无base组6个全部排除；来源文字数字仍属Documented Source Inconsistency。

正式split内精确重复、evidence-based group、guard cluster和已登记Candidate边均零跨集；关系中两端均入池8条同split，451条至少一端排除。U01派生函数从4个raw region精确保留3个有效目标，raw JSON未变；YOLO evaluator实际集成仍需Baseline环境复验普通FP行为。**D2 Dataset Protocol与D2 Split = FROZEN**，不能从中推出未知fruit/tree/session独立性，也不能沿用官方泄漏split。参见正式文档的校验、SHA和局限。
