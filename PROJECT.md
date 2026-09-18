# Project State

更新：2026-09-18。跨会话先读本页及 [研究计划与决策记录](研究计划与决策记录.md)。当前阶段的权威状态以决策记录为准；本页是便于衔接的摘要。

## Project Goal

基于公开数据开展苹果成熟/生长阶段视觉感知研究，目标SCI投稿。当前不具备自行采集条件。论文问题、贡献与题目由证据决定。

## Current Stage

**Stage 1 — Research Question & Literature Landscape**。

本次bounded调研产物验收：**Stage 1 = PASSED**。推荐下一阶段为Stage 2 — Dataset Acquisition & Audit，**尚未进入**，等待用户下一步指示。

项目起点为科研工程空白：目录与Research Skill System已经建立；本轮新增文献/状态资产，但尚无数据本地审计、Baseline实现、正式训练或实验结果。目录及预留E00–E10不代表完成任何实验。

## Research Question

**Not Frozen**。

工作任务定义：真实果园RGB图像中的逐果定位与视觉成熟/生长阶段识别。候选问题为：在控制样本相关性与场景泄漏后，阶段监督和保留标签语义的光照处理能否改善逐果判断可靠性，减少严重跨级误判并保持检测能力？这是调研建议，不是已冻结课题。

## Research Scope

自然光、多果、遮挡和复杂背景；优先公开苹果RGB实例标签。暂不将视觉颜色标签解释为Brix、firmness或生理可采收性。高光谱、整图BBCH和其他水果只作为边界/方法证据。任务类别数、品种范围和部署硬件未冻结。

## Candidate Research Hypotheses

| 候选 | 必须保留的状态 | 本轮评估 | 待验证/停止条件 |
|---|---|---|---|
| Ordinal | **Hypothesis only** | **Modify** | 标签确有顺序且多于二级、错误足够后再检验；顺序不可靠则暂停 |
| CPIP | **Hypothesis only** | **Modify** | 优于常规增强且不改变标签语义；仅同族合成扰动收益不足以支持真实泛化 |
| IOR-YOLO | **Working concept only** | **Keep as working title** | 不绑定最终YOLO版本、模块或论文题目 |

Modify是助手基于文献的评估，不是已经实现或验证的新模型。

## Literature Status

已完成本轮bounded systematic literature review：15篇核心记录（12篇苹果相关含数据/物候/生理背景，3篇邻域水果）。8篇全文可访问并抽取相关章节，7篇为摘要/正文片段有限证据；不是全部精读或穷尽综述。详见 [研究地图](docs/literature/literature-landscape.md)、[矩阵](docs/literature/literature-matrix.csv)、[协议](docs/literature/search-protocol.md)。

最接近工作：P01 HRLN-YOLO、P02 BGWL-YOLO；数据/任务P03 Fuji、P05三阶段数据；概念P07 FruitProM-V2、P08 CMF-Net、P12苹果连续评分；光照P13颜色量化、P15香蕉光照研究、P09苹果corruption。完整题名/来源见 [索引](docs/literature/source-index.md)。

## Dataset Status

**Not Frozen**。未下载数据。优先候选：D1 Orchard apple maturity、D2 Multi-Stage Pixel-Level Apple v4、D3 Fuji Ripeness & Size。各有标签顺序、增强副本/计数、二分类语义/许可等审计项，见 [数据候选](docs/literature/dataset-candidates.md)。没有已确认的跨数据集类别映射。

## Baseline Status

**Not Selected**。优先候选YOLO11n、YOLOv8n、Faster R-CNN R50-FPN；补充RT-DETRv2-R18与crop ResNet18，见 [基线候选](docs/literature/baseline-candidates.md)。未安装运行，无复现结果或冻结recipe。

## Proposed Method Status

未设计或实现正式方法。Ordinal/CPIP不构成已确认创新；原工程说明书继续作为Candidate Design v0。**Formal experiments：Not started**。

## Current Evidence

- 文献证据：颜色代理与生理测量任务不同；水果序数/概率成熟、苹果连续成熟与光照处理均已有先例。
- 数据元信息：D1有CC BY 4.0公开入口；D2同样有明确数据许可，但论文/仓库存在计数差异；D3需补许可及采集分组。
- 代码证据：静态核验了官方通用框架及部分作者数据/推理仓库；存在链接不等于完整训练可复现。
- **本项目实验证据：无。** 所有外部论文数字均为其作者报告，未在本项目复现。

## Open Questions

1. 最终采用视觉着色还是生长阶段定义？D1顺序能否被原始标注协议证明？
2. 哪个数据集能获得可靠原图族、采集单位及独立划分？三类/四类任务是否可持续？
3. 基线主要失败来自定位、标签歧义、跨级判断还是光照？
4. 天然光照分组与外部标签兼容性是否足以支撑泛化主张？
5. 剩余closest work全文、作者代码和许可能否补齐？

## Risks

颜色被当作生理真值；原图与增强/多视角泄漏；日期/品种与标签混杂；视觉阶段边界不确定；论文版本与数据计数不一致；现有prior art缩小方法空间；以同族扰动替代真实泛化；受限访问造成文献覆盖偏差。详见 [Gap Map](docs/research-gap-map.md)。

## Decisions

- 用户已确认：进入Stage 1、暂不采集、使用公开数据；停止Skill系统优化；本轮不下载、训练或实现候选模块。
- 已执行：AGENTS Git/工作树策略最小更新；Stage 1研究资产与记录建立。
- 助手建议：工作任务定义、候选清单、Ordinal/CPIP Modify、IOR保留工作名称；不标为用户已冻结研究路线。
- 验收判断：10项Stage 1产物条件已满足；Stage 2只推荐、未开始。

## Rejected Ideas

尚无被实验否决的技术方案。当前不采用的**论证方式**：红色等于真实成熟；添加Ordinal即创新；添加亮度/HSV/gamma即创新；联合检测成熟是未探索任务；跨论文mAP直接排名；多来源混合训练等于独立外部验证。拒绝这些表述不等于拒绝相应技术的潜在实用价值。

## Next Actions

等待用户对Stage 1建议的反馈与下一阶段指示。若授权Stage 2，先获取/核验许可和原始标签，再审计原图族、近重复及分组可行性，形成数据选择依据；补读最终选入closest work全文。此前不开始下载、训练、模型设计或永久工作树建设，也不继续优化Skill系统。
