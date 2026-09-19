# D2 v4 — Stage 2B-2 Raw ZIP Audit

更新：2026-09-19。**D2 = NOT FROZEN**。这是对用户取得的 `dataset-20260508.zip` 的低成本只读审计：读取ZIP目录、六份VIA JSON和JPG图像头，不解压、不修改原包、不解码全图、不重划分。取得来源、许可及SHA256见[获取记录](data-acquisition-d2.md)。`Confirmed`表示可由本次原包直接复核；`Supported`表示多条线索一致但尚缺像素/采集来源证明；`Unresolved`表示尚无足够证据解释或排除。

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
