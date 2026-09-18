# Stage 1 — Literature Landscape

2026-09-18；bounded systematic literature review / bounded evidence map。范围、检索与限制见 [检索协议](search-protocol.md)、[查询日志](search-log.json)、[筛选记录](screening-log.md)。P01–P15 对应 [文献矩阵](literature-matrix.csv) 与 [来源索引](source-index.md)。这是研究决策地图，不是穷尽综述，也不是模型创新性认证。

## RQ1 — 任务与应用场景

| Task formulation | 已有路线/证据 | 能回答什么 | 本项目适配 |
|---|---|---|---|
| Image/crop classification | P08、P10；P04的BBCH分类 | 已定位果实/整图阶段判断 | 可做诊断；不能代替多果定位 |
| Object detection | P09、P04检测部分 | 哪些位置有苹果 | 必要定位对照，不能直接回答成熟度 |
| Detection + maturity classification | P01、P02、P03、P11 | 每个可见果实的位置与视觉阶段 | **推荐主任务** |
| Ordinal classification | P08的CORAL；P07有序区间分布 | 区分相邻与远距离阶段误差 | 标签确有可靠顺序后才成立 |
| Regression / continuous estimation | P12连续评分、P13着色MI、P14生理回归 | 分别是潜在评分、颜色指数或真实测量量 | 三者不能互相替代；现阶段不预设连续生理成熟回归 |

**推荐明确的工作任务定义**：在真实果园自然光照、遮挡和多果背景的 RGB 图像中，定位每个可见苹果并预测数据集明确定义的视觉成熟/生长阶段。服务于果园视觉监测和选择性采摘的感知研究，不承诺直接预测可采收性、糖度或口感。类别数量、数据源和最终 Research Question 均未冻结。数据若只有生长阶段，题目与报告应相应称“生长阶段识别”。

**候选研究问题**：在控制近重复与场景泄漏的划分下，阶段监督与保留标签语义的光照处理能否改善逐果阶段判断的可靠性，尤其是严重跨级误差，同时保持定位性能？这仍是待数据审计和基线诊断检验的问题，不预设两个模块必须同时存在。

## RQ2 — 成熟真值与颜色代理

| 标签依据 | 已发现证据 | 可解释范围/风险 |
|---|---|---|
| Peel color | P03明确红/绿二分；P05按绿、转色、红；P13以Hue构建指数 | 视觉着色状态；不自动代表糖度、硬度或采收适期 |
| Expert visual annotation | P05单人标注后专家复核；P04专家BBCH | 提升一致性仍不是配对生理测量；需了解分歧与复核标准 |
| Brix / soluble solids | P14配对测量及回归 | 内部质量指标之一；当前RGB候选尚无同果配对证据 |
| Firmness | P14 | 反映质地/软化；与Brix不能互换 |
| Starch index | P14 §2.1记录同果淀粉染色比例测量；当前前三RGB候选未核实配对淀粉标签 | 无法据颜色补造淀粉等级 |
| Days after flowering / dates | P03采集日期参与标签；P04/P06有物候/日期信息 | 拍摄日期≠开花后天数；跨品种/季节不可直接统一阈值 |
| Physiological indicators | P14提供可量测监督对照 | 不能把潜在0–1分数称为已经校准的生理成熟度 |
| Dataset-provided labels | P01/D1四类开发阶段 | 必须追溯原标签含义和顺序；英文类别名不足以证明顺序 |

结论：**确有 red = mature / green = immature 的工作**，最直接是 [P03 §2.1](https://arxiv.org/html/2502.01850v1) 和 [P05 §4.3](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/)。该判据受品种、果面朝向、光照、曝光和着色与内部成熟不同步的影响。后一句是基于测量对象差异提出的风险解释，不是对这两个数据集已完成的实测归因。当前不可泛化到成熟仍为绿色/黄色的品种，亦不能把专家看图复核称为Brix验证。

标签审计应先确认“测的是什么”，再考虑模型。D1的 Young 与 Pre-growth 顺序尤其不能沿用原工程说明书的数字ID直接推定。

## RQ3 — 方法、常见改进与公平基线

| 路线 | 文献实际使用 | 主要变化 | 研究决策 |
|---|---|---|---|
| CNN/ResNet/MobileNet | P04的ResNet152/MobileNetv2；P10残差注意力；P08含ResNet基线 | 迁移学习、注意力、轻量结构 | 分类成熟；适合裁剪诊断，和检测mAP分开 |
| EfficientNet | P08比较EfficientNetV2-S | 高效分类骨干 | 有水果成熟先例；本轮不能据此声称苹果果园首选 |
| YOLO family | P01 YOLO11n/YOLOv8n；P02 YOLO11n；P11 YOLOv5 | 轻量卷积、neck融合、attention、head、框损失/分配 | 适合可控起点；模块叠加本身证据不足 |
| Faster/Mask R-CNN | P04 Faster R-CNN；P05 Mask R-CNN | 两阶段定位或实例mask | 提供异构结构对照；mask mAP与box mAP分开 |
| Transformer/foundation | P11 DETR/Swin；P03 Grounding DINO/SAM；P07 RT-DETRv2 | query检测、基础模型微调、分布输出 | 并非未探索；预算/依赖相对高 |

推荐优先比较 YOLO11n、YOLOv8n、Faster R-CNN R50-FPN；RT-DETRv2-R18和裁剪ResNet18为补充候选，详见 [基线候选](baseline-candidates.md)。这不构成 Baseline Selected。代码与权重核验见 [代码资源](../repositories/code-resources.md)。

公平性要求：同原始样本与分组划分、相同标签和box评价、相同开发/测试边界、透明预训练来源与调参预算。相同epoch不必然等同相同计算预算，跨架构结果与同骨干单因素消融分别解释。不同论文的mAP不组成跨论文排行榜。

## RQ4 — Ordinal hypothesis

| 证据 | Head / loss | Classification or detection | 代码与局限 |
|---|---|---|---|
| P08油棕CMF-Net | CORAL，K−1二分类监督、阈值解码，附颜色回归 | Classification | 未定位可核验作者实现；当前取到出版社§2.5片段，需补全文；不是苹果检测 |
| P07 FruitProM-V2 | Beta参数head，CDF得到类别区间概率，focal监督 | Detection | 预印本；未定位官方实现；邻级标签噪声收益不等于所有干净指标收益 |
| P12苹果遮挡成熟估计 | 自监督表示+连续0–1 predictor；具体loss未全文核验 | 连续评分，非已核验序数检测 | 苹果已有连续表达；GT及评分校准待核 |
| CORAL/CORN通用实现 | rank-consistent/conditional probability ordinal learning | 通用方法 | [作者实现](https://github.com/Raschka-research-group/coral-pytorch)，不能据此视为已有苹果实验 |

**判断：Modify。** 保留“可靠有序标签下，距离敏感评价和监督是否有价值”的问题，撤下“水果成熟序数建模空白”的暗示。本轮没有充分核验苹果实例检测中具体CORAL/CORN应用覆盖度，因此苹果特定新颖性仍为 Need More Evidence，而不是不存在先例。

继续的条件：至少三个语义可排序阶段；顺序经原始协议确认；验证集有足量跨级错误；检测漏检与匹配后分类错误分开评估。若只剩二类数据、有序关系不可靠或误差几乎全是漏检，暂停该主线。与普通分类、距离敏感代价/软标签等公平比较的具体方案留待后续，不在此设计新head。

## RQ5 — Illumination / CPIP hypothesis

| 问题 | 证据与回答 |
|---|---|
| 光照是重要failure mode吗？ | P15直接评估成熟分类在光照变化下退化；P09验证苹果定位的corruption退化；P13针对过曝着色处理。支持其为合理风险，但当前苹果数据上的主导错误尚未实测。 |
| 已有处理是什么？ | 普通HSV/亮度/对比度增强（P01、P05），光照图/gamma（P15），过曝mask与修复（P13），采集端主动闪光（P06）。 |
| brightness / HSV / gamma 常见吗？ | 已有直接应用，不能作为独立创新。此bounded样本不支持估计全领域使用比例。 |
| 专门面向成熟度的photometric工作？ | **有**：P15香蕉使用LIME+gamma，强调保留成熟相关色彩；P13苹果着色估计处理过曝。两者足以否定“专门成熟度光照处理未有人做”的笼统主张。 |
| 有系统corruption评价吗？ | **有**：P09在苹果检测中测试多种、多强度噪声/模糊/天气/数字失真；P15有成熟分类光照条件评价。仍须区分定位与成熟分类、合成与天然变化。 |

**判断：Modify。** 将CPIP从预设Lab扰动模块改为“保留视觉阶段标签语义的光照处理是否优于普通增强，并能泛化到未见变化”的假设。Lab保持a/b不变并不能自动证明RGB转换裁剪后色彩语义不变；既有数据中的离线亮度增强也不能直接当独立测试样本。

需补证据：当前数据天然光照分组是否可得；是否可以验证扰动不改变标注；常规增强是否已解决问题；未见扰动与外部真实场景是否仍有收益。若仅在与训练相同增强族上获益，不能称真实果园光照泛化。方法实现和增强强度均不冻结。

## Closest related work 与工作名称

| 比较层面 | 最接近工作 | 对本项目的约束 |
|---|---|---|
| 数据/检测骨架 | P01 HRLN-YOLO；P02 BGWL-YOLO | 同类数据/基线已有改进，不宜再以轻量neck/head堆叠作为唯一研究动机 |
| 苹果成熟任务及数据 | P03基础模型Fuji；P05三阶段实例数据；P10 AFGL-MC | 标签语义与定位/分类边界须可比较 |
| 有序、连续成熟 | P07 FruitProM-V2；P08 CMF-Net；P12自监督苹果连续评分 | 已有先例与干净数据负面证据必须纳入 |
| 光照与着色 | P13 fine-grained coloration；P15香蕉光照鲁棒；P09苹果corruption | 普通增强、颜色解耦、鲁棒性评测均不能被笼统称作空白 |

**IOR-YOLO：Keep as working title。** 仅为项目工作概念，名字不决定最终YOLO版本、模块、贡献或论文标题；若标签/基线不支持当前路线，可改名或暂停。

## Stage 1 Exit Criteria

| 条件 | 状态 | 证据产物 |
|---|---|---|
| Task definition明确 | 满足（工作定义，非最终RQ冻结） | RQ1 |
| Application scenario明确 | 满足 | 真实果园自然光RGB逐果感知 |
| Literature matrix | 满足 | 15篇去重记录，访问限制逐条标记 |
| Maturity ground truth调查 | 满足 | RQ2；颜色/专家/测量/日期区分 |
| Dataset candidates | 满足 | 3个优先候选+辅助/暂缓项 |
| Baseline candidates | 满足 | 5项候选；前3项优先 |
| Ordinal判断 | 满足：Modify | RQ4；反例及停止条件 |
| Illumination判断 | 满足：Modify | RQ5；直接先例 |
| Closest related work | 满足 | 上表+来源索引 |
| Evidence-based gap map | 满足 | [Research Gap Map](../research-gap-map.md) |

**Stage 1 = PASSED（本次bounded调研产物验收）**。推荐下一步进入 Stage 2 — Dataset Acquisition & Audit；本次未进入。通过不代表RQ、数据或baseline冻结，不代表候选创新成立，也不替代未取得全文的后续精读。新证据可重新打开此阶段判断。
