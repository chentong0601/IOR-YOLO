# Dataset Feasibility Gate

核查日期：2026-09-19。状态：**PASSED（数据选择仍为 Not Frozen）**。本轮仅核验官方元数据、数据论文与作者入口，没有下载数据包、建立环境或运行模型。

## 结论与角色

- **Primary Dataset Candidate：Multi-Stage Pixel-Level Apple v4（D2）**。它是三个候选中唯一同时具备公开数据许可、三阶段逐实例标签、明确视觉标注规则和检测/分割标注的数据。正式使用前必须从原始 1124 张中识别所有派生图与双分辨率副本，完成原图族去重并重新建立隔离划分。
- **External Dataset Candidate：Fuji Ripeness & Size（D3，条件性）**。适合做跨地区、设备和采集方式的检测泛化；视觉成熟度只适合预先定义的二类端点 zero-shot 评价。D2 的 semi-mature 不应被强行映射。任何用 D3 微调或域适配后的结果必须单列为 adaptation，不得称为 zero-shot external validation。
- **Secondary / Rejected for primary：Orchard apple maturity（D1）**。保留为小规模补充审计候选，但暂不作为主数据：原始说明未公开可核验的品种、实例数、标注协议、生理依据或四类的严格顺序，且只有 train/val，没有独立 test。

## 统一比较矩阵

| Dataset | Source | License | Variety | Environment | Images | Instances | Classes | Maturity Definition | Annotation | Ordinal Support | Illumination Diversity | Duplicate Risk | Leakage Risk | Official Split | External Validation Suitability | Availability | Evidence Level | Major Risks |
|---|---|---|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 Orchard apple maturity | [Figshare DOI](https://doi.org/10.6084/m9.figshare.29533841.v1)；[数据使用论文](https://doi.org/10.3389/fpls.2026.1820164) | CC BY 4.0（DataCite权利元数据） | 未说明 | 真实果园；不同光照和背景 | 2039 | 未报告 | Young、Pre-growth、Late-growth、Ripe | 仅称按 growth stages；具体观察或生理依据未说明，**Unclear** | 论文证实为目标检测；包内格式待核 | **Weak** | **Moderate**：来源声称有变化，无光照属性 | **Unknown** | **Moderate–High / unresolved**：无果实、树、时段ID及去重报告 | 1618 train / 421 val；无独立 test | **Weak**：类别语义和顺序未证实 | DOI与约130.6 MB的7z存档仍有元数据记录；本轮未下载 | **Moderate**（规模/类别/拆分有论文证据，标签协议证据不足） | 四类顺序、品种、实例数、标注依据和分组不可核；现成划分不能支持强外推 |
| D2 Multi-Stage Pixel-Level Apple v4 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)；[Data in Brief](https://doi.org/10.1016/j.dib.2026.112856) | 数据 CC BY 4.0；SCI研究可在署名条件下使用 | Fuji、Gala | 陕西同一实验果园；2017–2019；多天气、时段、冠层位置、遮挡 | 原始1124；离线增强后1406；每张另有两套分辨率 | 原始2108；论文增强后2574，Mendeley v4为2573 | immature、semi-mature、mature | 绿色、绿红转色、以红色为主；单人标注、农业专家复核；无Brix/硬度/淀粉，视觉语义 **Verified**，生理成熟 **Unsupported** | VIA polygon JSON，逐实例阶段；可导出box | **Moderate** | **Moderate**：晴/阴/局部阴影、上午/傍晚；无逐图光照标签或色卡 | **High**：离线增强图及双分辨率派生图 | **High risk, not proven leakage**：文中流程为先增强后划分，未说明原图族、树或采集session隔离 | 1041/126/239 train/val/test；标称70/10/20与实际74/9/17不符，无seed/group IDs | **不宜作为自己的外部集**；适合作主数据与内部审计划分 | v4公开下载，页面显示 Download All | **Strong**（采集、标签、结构、许可均有原始来源；包级事实待下载） | 颜色代理、边界主观；派生图泄漏；计数差1；单地点；EXIF与颜色校准缺失 |
| D3 Fuji Ripeness & Size | [作者Kaggle](https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset)；[论文](https://arxiv.org/abs/2502.01850)；[作者仓库](https://github.com/zhukeyi-stan/Fuji_Ripeness_And_Size_Estimation) | Kaggle API列为 GPL 3；作者仓库无LICENSE；两个原始数据源的再分发许可链仍需包级确认 | Fuji | Bologna（RealSense RGB-D，10天）+ Agramunt（DSLR/SfM，2日期），真实果园多视角 | 4027 = 102 + 3925 | 16257 = 922 + 15335 | ripe、unripe | 日期作初步参考，再按果面红色比例二分；视觉语义 **Verified**，生理成熟 **Unsupported** | 统一为bounding box；一来源原有box，另一来源由modal/amodal mask转box | **Unsupported**（只有二级，不能研究严重跨级错误） | **Moderate**：论文明确部分多光照，但无统一光照分层 | **Very High**：24个被跟踪果实的多日/多视角；另一来源相邻图>75%重叠 | **High**：论文仅报告约75/25图像划分，未说明按果实、树、日期或重建序列分组 | 约75% train / 25% validation；Kaggle含train/test目录，精确映射待包审计 | **Moderate, conditional**：检测外测可行；二类视觉端点成熟外测可行但标签映射受限 | Kaggle公开v4，约62.36 GB；作者README链接可用 | **Strong**（规模/采集/标签论文明确）；**Moderate**（许可链与包内split尚未核） | 多视角泄漏、日期与类别混杂、二类语义、来源异质、包很大、许可链需复核 |

> Image count 与 object/instance count 在上表分列。D2 的两种分辨率是同一图像的派生版本，不能重复计为独立采集；D3 的 4027 张也包含大量同果或重叠视角，不能按独立果实解释。

## 成熟度 Ground Truth

| Dataset | 已核实依据 | 判断 | 可允许的论文表述 | 禁止外推 |
|---|---|---|---|---|
| D1 | 数据说明只列四个 development-stage 类别；使用论文以外观示例展示，但无标注准则、专家流程、日期或理化指标 | **Unclear** | “dataset-provided visual/development-stage labels” | 不得称真实生理成熟、采收成熟或严格四级序数真值 |
| D2 | 明确按果皮颜色分：纯绿、绿红过渡、红色占主；单一标注者，农业专家复核 | **Verified（视觉颜色阶段）**；**Unsupported（生理成熟）** | “visual color-transition stage” | 不得替代Brix、firmness、starch index或采收标准 |
| D3 | 采集日期作先验，再按果面红色占比分 ripe/unripe；原始Amodal子集日期对应BBCH77/85，但再标注仍是视觉二分 | **Verified（日期辅助的视觉颜色标签）**；**Unsupported（生理成熟）** | “binary visual ripeness proxy for Fuji” | 不得把日期、红色或二类标签等同单果生理成熟 |

三个数据集均不能支持“模型测量了真实生理成熟度”的结论。当前最稳妥任务名称仍是**逐果视觉生长/着色阶段识别**。

## Ordinal 可行性

- **D2：Moderate**。`immature → semi-mature → mature` 的顺序由明确颜色转变定义，能够预注册距离敏感指标和 severe cross-stage error（immature↔mature）。其边界仍是主观颜色分箱，且没有理化校准；因此只能研究视觉阶段顺序。
- **D1：Weak**。类别名称似乎有生长顺序，但 `Pre-growth` 与 `Late-growth` 的原始定义、标注规则和顺序证据缺失。取得包也不能自动补足语义文件；未补证前不得作为ordinal主验证。
- **D3：Unsupported**。二类只有一个阈值，无法评估相邻级与严重跨级错误；可做二类端点外部检测，不能验证多阶段ordinal主张。

**项目结论：Ordinal Dataset Support = Moderate，且只来自D2。** Ordinal hypothesis继续为 **Modify / Need empirical evidence**：先验证标签一致性和基线错误结构，再决定是否进入方法阶段。

## Illumination robustness 可行性

- D2有原始真实果园RGB、晴天/阴天/局部遮阴、上午/傍晚和多设备；可做保留颜色语义的photometric corruption开发。问题是无逐图光照元数据、无色卡，发布包已含HSV亮度派生图，标签本身又依赖颜色。
- D1声称真实果园多光照/背景，但缺少逐图条件、设备和协议；目前只能作为弱分组证据。
- D3包含多日、多视角和论文所称的多种光照条件，能增加跨采集域变化；但光照与日期、地点、设备、视角共同变化，不能把外测差异单独归因于光照。

**项目结论：Illumination Research Support = Moderate。** 可以建立固定、标签语义保持的合成corruption benchmark；在没有天然光照分组前，不主张“真实光照域泛化”。强色偏或使红/绿阶段跨界的变换必须排除或单独标为标签不变性压力测试。

## 泄漏与拆分要求

1. **D2不沿用现成split作为最终协议。** 下载后先建立原图族：同一原始图的分辨率版本、亮度版、噪声版和可能的重命名副本必须同组；用文件名、尺寸、感知哈希和标注几何核查。若无树/日期/session ID，只能做保守的source-family split，并明确不能排除同树或同果跨集。
2. **D3必须按来源和采集组审计。** 24个被跟踪果实、同树多日多视角及>75%重叠的SfM序列不能随机跨集。优先按fruit/tree/date/sequence划分；若Kaggle包不能恢复这些ID，不把论文的75/25 image split当作可信外部评价。
3. **D1只能先做近重复与目录审计。** 官方train/val存在不等于独立；没有稳定group ID时，其成熟性能只可作为同分布补充结果。
4. 所有离线增强只能从training原图生成；validation、internal test和external test保持原始，所有派生图继承source ID。

## External validation 协议边界

**Zero-shot external validation**：模型及阈值只能在D2 train/validation确定，然后直接在未参与开发的D3上评价。定位可按apple box评价；成熟度只评价预注册的兼容端点：D2 `mature → ripe`、`immature → unripe`，D2 `semi-mature`不映射到D3。还需确认D3测试项与任何预训练来源无重叠。

**Fine-tuning / domain adaptation**：在D3使用任何训练样本、伪标签、阈值调节或映射调优后，结果单列为adaptation。它可回答可迁移性，但不能作为zero-shot跨数据集泛化证据。

D3的cultivar相同（Fuji）有利于缩小品种差异，却仍存在中国/意大利/西班牙、手机/RGB-D/DSLR、三类/二类、polygon/box和日期分布差异。当前 **External Validation Feasibility = Moderate, conditional**。

## Research Gap 复审与排序

| 排名 | Gap | 最新状态 | 数据条件带来的解释 |
|---:|---|---|---|
| 1 | 标签语义与可信跨场景评价 | **Strongly Supported** | 三个数据集均为视觉代理，且分组/派生关系会直接影响可信度；这是现阶段最扎实的问题 |
| 2 | 保留标签语义的光照泛化 | **Supported** | D2/D3具备真实变化且颜色是核心标签线索；可研究，但无天然光照标签，不能把跨域差异全归因于光照 |
| 3 | 有序错误及边界不确定性 | **Weakly Supported** | 只有D2提供可用三阶段；D1顺序不明，D3二类无法外验严重跨级错误；需先看D2标签一致性与基线错误量 |

## Gate 判定与下一动作

Dataset Feasibility Gate的八个问题均已得到可执行回答，故 **Gate = PASSED**；这不等于Dataset Frozen，也不等于完整Stage 2完成。

下一次获得下载授权后，建议先固定并下载 **D2 v4**，记录DOI、版本、文件清单、大小与校验和，再只做包级结构、标签、重复和split审计。D1可在D2审计后作为低成本补充包核查。D3约62.36 GB，先补齐原始来源许可链和分组方案，再决定是否下载；不要先下载全部数据再寻找评价用途。

## 证据来源

- D1：[Figshare DOI/元数据](https://doi.org/10.6084/m9.figshare.29533841.v1)；[HRLN-YOLO数据描述](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1820164/full)。
- D2：[Mendeley Data v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)；[Data in Brief全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/)。
- D3：[论文全文](https://arxiv.org/html/2502.01850v1)；[作者数据入口](https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset)；[作者代码入口](https://github.com/zhukeyi-stan/Fuji_Ripeness_And_Size_Estimation)。

