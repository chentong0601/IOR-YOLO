# Method Taxonomy

2026-09-18。分类依据为 [P01–P15文献矩阵](literature-matrix.csv)，按技术路线组织；以下不是本项目实现计划。

| 维度 | 类别 | 代表证据 | 选择约束 |
|---|---|---|---|
| Task formulation | 整图/单果分类 | P08/P10；P04物候分类 | 整图BBCH、单果成熟不可混同 |
| Task formulation | 定位检测 | P04/P09 | 无成熟标签不能评价成熟性能 |
| Task formulation | 联合逐果阶段检测 | P01/P02/P03/P11 | 推荐工作任务，class-aware box评估 |
| Task formulation | 实例分割+阶段 | P05 | mask可导出box；必须统一可见/完整果实边界 |
| Task formulation | 连续/概率估计 | P07/P12/P13/P14 | 分布潜变量、颜色指数、生理回归各有不同GT |
| Model families | CNN分类器 | ResNet、MobileNet（P04）；EfficientNetV2-S（P08） | 轻量成熟，但仅crop准确率会漏掉定位失败 |
| Model families | One-stage CNN detector | YOLO家族（P01/P02/P11）；ATSS（P09） | 检测效率与多尺度改进已有大量路线，本轮不量化占比 |
| Model families | Two-stage | Faster R-CNN（P04）、Mask R-CNN（P05） | 异构参照，计算预算更高 |
| Model families | Transformer / foundation | DETR/Swin（P11）、Grounding DINO/SAM（P03）、RT-DETRv2（P07） | 不能仅以模型新旧选择基线 |
| Maturity supervision | Nominal classes | P01/P03/P05 | 对类别距离无显式约束，但标签未排序时最稳妥 |
| Maturity supervision | Ordinal thresholds | P08 CORAL | 顺序必须经审计，不将软件ID解释为时间顺序 |
| Maturity supervision | Distribution/interval | P07 Beta-CDF | 处理模糊边界；需要校准/标签噪声对照 |
| Maturity supervision | Continuous score | P12/P13 | 连续分数不自动获得生理解释 |
| Maturity supervision | Physiological measurement | P14 firmness/Brix | 需同果、同时间配对测量；现有RGB候选不能替代 |
| Robustness strategies | 常规photometric增强 | P01/P05；Scifresh作者仓库 | 保留原图—派生图关系；不跨split分散 |
| Robustness strategies | 光照图、gamma、颜色/过曝处理 | P15/P13 | 亮度改变可能破坏成熟色彩线索，应单独检验 |
| Robustness strategies | 重建/采集规范 | P12遮挡重建、P06主动闪光 | 属不同机制，不能都叫光照不变 |
| Robustness strategies | 多来源与corruption评测 | P04/P09/P15 | 混合训练、自然外测、合成腐蚀须分开报告 |
| Deployment strategies | 轻量backbone/neck/head | P01/P02/P10 | 参数少不必然更快，需同硬件、输入和后处理 |
| Deployment strategies | 图像级诊断/两阶段pipeline | 分类器+检测器候选组合 | 有GT crop的结果是诊断上界，不是端到端检测成绩 |

建议后续评价分三层：定位能力、匹配果实的阶段辨识、整个感知任务的端到端性能。序数误差只对具有已确认顺序的标签定义，漏检率必须同时报告，不能靠少检难例降低跨级误判率。上述为评价原则，正式指标、匹配阈值与协议待后续冻结。
