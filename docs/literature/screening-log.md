# Screening / Deduplication Log

2026-09-18。有界筛选；不宣称下表覆盖搜索引擎全部命中。15条核心纳入项逐条理由、阅读位置及访问级别见 [矩阵](literature-matrix.csv) / [来源索引](source-index.md)。其中P08/P07/P15为邻域水果，P14为生理监督背景，P04/P06为物候资源，P09为检测corruption先例；这些不能冒充同一苹果成熟benchmark的直接对照。

## 核心纳入理由

| IDs | 纳入依据 |
|---|---|
| P01/P02 | 苹果逐果成熟检测与轻量baseline最近路线 |
| P03/P05 | 公开逐果阶段数据及标签定义 |
| P04/P06 | 物候和多场景资源，区分整图阶段与逐果成熟 |
| P07/P08 | 对ordinal/连续成熟假设有直接约束的水果方法 |
| P09 | 苹果检测已有系统corruption评价的直接反例 |
| P10/P11 | 苹果分类与Transformer方法族覆盖 |
| P12/P13 | 苹果连续评分与颜色/过曝处理closest prior art |
| P14 | 配对生理真值及不同任务的边界 |
| P15 | 成熟分类专用光照增强/评价直接先例 |

## Pending / 暂缓（不计15篇）

| 条目 | 定位线索 | 原因与下一步 |
|---|---|---|
| DDCA-Net: A Dual-Domain Cross-Attention Network for Real-Time Cashew Tree Localization and Ordinal Fruit Maturity Classification | ICECCME 2025节目单/ResearchGate线索 | 未取得可核正式论文DOI/作者完整原文；不使用其性能/ordinal实现细节；后续重点补证 |
| GhostGS and ECA2 enhanced framework for accurate strawberry maturity detection in smart agriculture | ResearchGate accepted-manuscript线索 | 正式出处尚未核实；不将该线索作为已证实草莓ordinal检测成果 |
| Canopy-attention-YOLOv4-based immature/mature apple fruit detection on dense-foliage tree architectures for early crop load estimation | [DOI](https://doi.org/10.1016/j.compag.2022.106696) | 早期苹果不同阶段检测相关；需读清“检测不同阶段数据”还是“逐果成熟分类”，暂不使用数字 |
| Maturity Detection of Apple in Complex Orchard Environment Based on YOLO v7-ST-ASFF | DOI 10.6041/j.issn.1000-1298.2024.06.023 | 中文期刊入口本轮异常跳转；需核正式全文，未据此作比较结论 |
| Target Detection for Coloring and Ripening Potted Dwarf Apple Fruits Based on Improved YOLOv7-RSES | DOI 10.3390/app14114523 | 盆栽场景和果园适用性待精读；本轮保留检索线索 |
| Apple Ripeness Identification Using Deep Learning | DOI 10.1007/978-3-030-72073-5_5 | 同作者较早工作；P11已覆盖方法族，后续按需要追溯，不当重复版本计数 |
| Convolutional Neural Networks for Estimating the Ripening State of Fuji Apples Using Visible and Near-Infrared Spectroscopy | DOI 10.1007/s11947-022-02880-7 | 光谱任务补充，当前P14已提供生理监督锚点 |

## 排除与合并

- 品种识别、病害分类、一般水果品类识别：不回答成熟阶段问题；不以高accuracy填充成熟矩阵。
- gamma辐照保鲜论文：与图像gamma变换同词不同任务，排除。
- Reddit、博客、商业聚合摘要：可给检索线索，不作方法事实来源。
- P03 arXiv与正式CEA论文合并1篇；P04 arXiv/项目页/CVF合并1篇；P11在线2023与卷期2024合并1篇。数据仓库不额外计“论文”。
- 通用CORAL/CORN软件、YOLO/Torchvision文档：代码/方法背景，不计15篇领域核心论文。

## 文献可见不一致

- P05论文与Mendeley总实例数2574/2573；分辨率副本、增强图与原始图数须分开。
- P04项目页与论文dense subset计数不同；先保留版本差异，不能挑大数。
- P07实验文字中的test/validation指称有不一致；数值仅标作其论文表格报告，不作公平性已证实结论。
- P03/Fuji两类不能无损映射四级成长标签；代码入口存在不等于可复现训练。

这些问题已进入数据候选和研究风险，尚未进行包级审计或实验验证。
