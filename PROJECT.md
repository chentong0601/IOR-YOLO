# Project State

更新：2026-09-19。跨会话先读本页及 [研究计划与决策记录](研究计划与决策记录.md)。当前阶段的权威状态以决策记录为准；本页是便于衔接的摘要。

## Project Goal

基于公开数据开展苹果成熟/生长阶段视觉感知研究，目标SCI投稿。当前不具备自行采集条件。论文问题、贡献与题目由证据决定。

## Current Stage

**Stage 2 — Dataset Acquisition & Audit**。

Stage 1与Stage 2A Dataset Feasibility Gate已通过。**Stage 2B-3 — Leakage & Annotation Integrity Audit** 本轮已完成全图解码、SHA256、候选同源图和标注核查。官方split已确认泄漏；可信评价划分尚未建立，完整Stage 2未完成。

项目起点为科研工程空白：目录与Research Skill System已经建立；D2原包已完成本轮只读数据审计，但尚无正式划分、Baseline实现、训练或实验结果。目录及预留E00–E10不代表完成任何实验。

## Research Question

**Not Frozen**。

工作任务定义：真实果园RGB图像中的逐果定位与视觉成熟/生长阶段识别。候选问题为：在控制样本相关性与场景泄漏后，阶段监督和保留标签语义的光照处理能否改善逐果判断可靠性，减少严重跨级误判并保持检测能力？这是调研建议，不是已冻结课题。

## Research Scope

自然光、多果、遮挡和复杂背景；优先公开苹果RGB实例标签。暂不将视觉颜色标签解释为Brix、firmness或生理可采收性。高光谱、整图BBCH和其他水果只作为边界/方法证据。任务类别数、品种范围和部署硬件未冻结。

## Candidate Research Hypotheses

| 候选 | 必须保留的状态 | 本轮评估 | 待验证/停止条件 |
|---|---|---|---|
| Ordinal | **Hypothesis only** | **Modify；数据支持Moderate** | 仅D2具备明确三阶段颜色顺序；先审计一致性与基线错误，D3不能外验严重跨级错误 |
| CPIP | **Hypothesis only** | **Modify；数据支持Moderate** | D2/D3有真实光照变化但无逐图光照属性；只允许语义保持扰动，不把跨域差异全部归因于光照 |
| IOR-YOLO | **Working concept only** | **Keep as working title** | 不绑定最终YOLO版本、模块或论文题目 |

Modify是助手基于文献的评估，不是已经实现或验证的新模型。

## Literature Status

已完成本轮bounded systematic literature review：15篇核心记录（12篇苹果相关含数据/物候/生理背景，3篇邻域水果）。8篇全文可访问并抽取相关章节，7篇为摘要/正文片段有限证据；不是全部精读或穷尽综述。详见 [研究地图](docs/literature/literature-landscape.md)、[矩阵](docs/literature/literature-matrix.csv)、[协议](docs/literature/search-protocol.md)。

最接近工作：P01 HRLN-YOLO、P02 BGWL-YOLO；数据/任务P03 Fuji、P05三阶段数据；概念P07 FruitProM-V2、P08 CMF-Net、P12苹果连续评分；光照P13颜色量化、P15香蕉光照研究、P09苹果corruption。完整题名/来源见 [索引](docs/literature/source-index.md)。

## Dataset Status

**Not Frozen**。D2仍是Primary Dataset Candidate；D3为条件性External候选，D1为Secondary。D2的2812张图均完整解码；非resize的1406张含2575个polygon，其中2574有三类有效标签，test一条类别缺失。67个命名跨split候选家族中65个获视觉支持、2个仍为候选；**另有3组跨split字节完全相同的图像，确认官方split泄漏**。候选图有1107个组，其中67组跨现有split；1403对original/resize相似度高，3对异常。详见[原包审计](docs/dataset-audit-d2.md)和[文件清单](IOR-YOLO/data/manifests/files_sha256.csv)。

## Baseline Status

**Not Selected**。优先候选YOLO11n、YOLOv8n、Faster R-CNN R50-FPN；补充RT-DETRv2-R18与crop ResNet18，见 [基线候选](docs/literature/baseline-candidates.md)。未安装运行，无复现结果或冻结recipe。

## Proposed Method Status

未设计或实现正式方法。Ordinal/CPIP不构成已确认创新；原工程说明书继续作为Candidate Design v0。**Formal experiments：Not started**。

## Current Evidence

- 文献证据：颜色代理与生理测量任务不同；水果序数/概率成熟、苹果连续成熟与光照处理均已有先例。
- 数据元信息：D1、D2均有CC BY 4.0公开入口；D2采集、视觉标注协议和结构证据最完整，但有派生图泄漏高风险；D3 Kaggle列GPL 3且约62.36 GB，作者论文确认多视角/重叠采集，原来源许可链与分组仍需包级确认。
- 代码证据：静态核验了官方通用框架及部分作者数据/推理仓库；存在链接不等于完整训练可复现。
- **本项目实验证据：无。** 所有外部论文数字均为其作者报告，未在本项目复现。
- D2原包哈希已复核、2812图完整解码；manifest记录图像哈希、类别和关系边。真实字节重复跨split已确认，但全部增强派生关系及fruit/tree/session独立性仍待验证。

## Open Questions

1. D2包内1406非resize图像与2575区域（2574有效类别）为何与外部1124/2573/2754等数字冲突？实际来源链如何解释？
2. 在已确认跨集重复、67个命名候选家族及跨名视觉候选之上，如何复核来源图并建立足够可信的group-aware评价方案？缺少fruit/tree/session ID带来何种主张边界？
3. 基线主要失败来自定位、标签歧义、跨级判断还是光照？
4. 天然光照分组与外部标签兼容性是否足以支撑泛化主张？
5. D3原始两来源的再利用许可与Kaggle打包许可是否一致，能否恢复fruit/tree/date/sequence分组？

## Risks

颜色被当作生理真值；**D2官方split已有确认的跨集重复**，且相同图像的polygon并不总一致；test一条无类别区域、三对original/resize内容/坐标异常；fruit/tree/session分组未知。D2的1124/1406、2573/2574/2754及比例差异未解释。D3同果多视角与SfM重叠、二类外部标签强映射、合成扰动替代真实光照泛化亦为后续风险。详见[原包审计](docs/dataset-audit-d2.md)。

## Decisions

- 用户已确认：进入Stage 1、暂不采集、使用公开数据；停止Skill系统优化；本轮不下载、训练或实现候选模块。
- 已执行：AGENTS Git/工作树策略最小更新；Stage 1研究资产与记录建立。
- 助手建议：工作任务定义、候选清单、Ordinal/CPIP Modify、IOR保留工作名称；不标为用户已冻结研究路线。
- Stage 1验收时的判断：10项产物条件已满足；当时Stage 2只推荐、未开始。
- 用户已确认进入Stage 2；本轮Dataset Feasibility Gate已通过。D2/D3/D1的角色是证据驱动建议，Dataset仍Not Frozen，需下载后的包级审计才能确认。
- Stage 2B-1结束时，用户已可见下载并首次验证D2原包；当时只准备了ZIP结构命令。D2仍Not Frozen。
- 用户进一步确认Stage 2B-2低成本只读审计；已执行真实ZIP图像头和JSON检查，证据与尚未解决风险记于审计文档。D2保持Not Frozen；未建立或冻结Baseline。
- 用户确认Stage 2B-3只读泄漏与完整性审计；已确认官方split有3组跨集字节重复，因此**不得直接沿用官方split做正式Baseline/最终评价**。本轮不创建替代split，D2仍Not Frozen。

## Rejected Ideas

尚无被实验否决的技术方案。当前不采用的**论证方式**：红色等于真实成熟；添加Ordinal即创新；添加亮度/HSV/gamma即创新；联合检测成熟是未探索任务；跨论文mAP直接排名；多来源混合训练等于独立外部验证。拒绝这些表述不等于拒绝相应技术的潜在实用价值。

## Next Actions

本轮在[原包审计](docs/dataset-audit-d2.md)停止。后续可基于[关系边清单](IOR-YOLO/data/manifests/duplicate_report.csv)设计Group-Aware Split规则，但先复核候选边、同字节图不同polygon、test无类别区域、三对resize异常及可得采集元数据；**本轮不生成或冻结split，不运行Baseline**。D3仍需补许可链。D2 Not Frozen。
