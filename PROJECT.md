# Project State

更新：2026-10-01。跨会话先读本页及 [研究计划与决策记录](研究计划与决策记录.md)。当前阶段的权威状态以决策记录为准；本页是便于衔接的摘要。

## Project Goal

基于公开数据开展苹果成熟/生长阶段视觉感知研究，目标SCI投稿。当前不具备自行采集条件。论文问题、贡献与题目由证据决定。

## Current Stage

**当前：Stage 3E-2 Local-First E01 Development & CPU Engineering Validation。** E01科学协议与D2 split保持冻结；当前工作只实现/验证平台无关工程路径。正式训练、正式Val及final Test均未开始，Stage 4未开始。Mac VS Code 是主要开发环境；Windows CPU 是可选兼容检查；后续 CUDA quick validation 与正式 E01 在 Kaggle/cloud 使用 `cuda:0`。见[本地开发工作流](docs/e01-local-development.md)、[Kaggle执行准备](docs/kaggle-e01-training-setup.md)和[决策记录](研究计划与决策记录.md)。

Stage 1与Stage 2A Dataset Feasibility Gate已通过。2B-5P候选方案经用户审核确认，已按[正式冻结协议](docs/dataset-freeze-d2.md)完成两次确定性重建与验证。Stage 2B-4H的21案人工决定及1112个evidence-based source groups继续有效。官方split已确认泄漏，不可用于正式评价；现已冻结guard-aware 70/15/15划分。

项目起点为科研工程空白：目录与Research Skill System已经建立；D2原包已审计且有正式manifest与split，但仍无Baseline实现、训练或实验结果。目录及预留E00–E10不代表完成任何实验。

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

**D2 Dataset Protocol = FROZEN；D2 Split = FROZEN**，仅适用于固定ZIP上的当前视觉三阶段任务。正式池1099张，train/val/test为769/165/165；详见[正式冻结协议](docs/dataset-freeze-d2.md)。D3仍为条件性External候选，D1为Secondary。D2的2812张图均完整解码；非resize的1406张含2575个polygon，其中2574有三类有效标签，test一条类别缺失。**另有3组跨官方split字节完全相同图像，确认官方split泄漏**。下述早期计数是历史审计，不是当前正式split。详见[同源组解析](docs/source-group-resolution-d2.md)、[划分协议模拟](docs/split-protocol-d2.md)和[原包审计](docs/dataset-audit-d2.md)。

Stage 2B-4V 的**独立重建**完全复现 1114 组的成员与组号；Conservative / Current / Worst-Case 为 **1120 / 1114 / 1096** 组。1000 次当前协议模拟均保持 Confirmed/Strongly Supported/Supported 零跨集，但 Candidate 边仍会被切断，且缺少果实/树/场次 ID。该验证**不构成 Dataset Freeze**；见[审核包](docs/review-packet-stage2b4.md)。

**Stage 2B-4H裁决后的历史状态**：C01/C02两对接受同源并由Candidate升为Strongly Supported；B01/B02桥边接受、保留完整原组并升为human-reviewed Strongly Supported；S01–S10保持原组/Strongly Supported。程序重建为**1112个候选source groups，67组跨官方split、涉及204张非resize表示**；67个跨集命名家族均为Strongly Supported，但23条跨组dHash Candidate与采集ID缺失仍在。上述1114及1000次模拟为**裁决前图**的历史结果，不能套用到新图。D01–D03仍是Confirmed图像重复且标注冲突Unresolved；R01–R03是same-source companion但确定性resize等价Unresolved。该段当时D2 Not Frozen；当前冻结状态见本节首段。

**Stage 2B-5P历史候选设计**：1112个source groups经23条跨组dHash Candidate和另6条命名Candidate的预防性约束形成**1096个split guard clusters**；guard不证明同源。非resize的1119张plain、170张brightness、117张noise中，6个组没有plain/base表示。每个有base组确定性留一个代表，发现同字节不同polygon共**10条SHA关系、7个source groups**（包括D01–D03）；严格Policy B暂存这7组，得到**1099张图**的canonical experiment pool。来源侧数字冲突为固定包口径下的Documented source inconsistency，原因仍未知。候选时两比例模拟未存正式归属；正式结果改见[冻结协议](docs/dataset-freeze-d2.md)。

## Baseline Status

**E01主Baseline：官方预训练YOLO11n-seg，instance segmentation；协议冻结，正式训练尚未开始。** 原始D2逐实例polygon保留为YOLO segmentation标签，同时可评估box/mask。正式配置仍见[实验配置](IOR-YOLO/configs/experiments/e01_yolo11n_seg.yaml)；另有完全隔离的[工程 profile](IOR-YOLO/configs/development/e01_local_engineering.yaml)与[本地 runner](IOR-YOLO/scripts/19_e01_local_engineering.py)，用于 CPU/MPS/CUDA `local-smoke`、`local-quick`，输出不能用于科研结论或正式初始化。

训练前核对固定Ultralytics 8.3.220源码后，正式配置使用官方默认机制`optimizer=auto`；实际优化器/LR须由正式run记录。正式执行唯一设备为`cuda:0`，`batch=8`、`imgsz=640`、最多100 epochs、seed0与既有augmentation不变。正式 E01 始终从官方`yolo11n-seg.pt`重新初始化。final Test 仍由现有`--final-test`流程锁定。工程输出在`runs/e01_engineering/`，标记 **ENGINEERING VALIDATION ONLY - NOT FOR PAPER**，不会纳入正式run路径。Stage 3E-2不运行正式训练/Val/Test，不产生论文结果。

正式科学协议不变：官方预训练`YOLO11n-seg`、instance segmentation、640、最多100 epochs、batch8、optimizer auto、训练seed0、D2冻结70/15/15、原augmentation和metrics。正式环境继续锁定Python3.11、PyTorch2.5.1+cu121、torchvision0.20.1+cu121、Ultralytics8.3.220；Kaggle CUDA预检不兼容时须停止，不得擅改协议依赖。设备角色现为Mac本地工程开发、可选Windows CPU兼容性核验、Kaggle/cloud CUDA quick validation与未来正式执行。路径通过命令行显式传入，正式执行脚本只接收官方初始化权重。

## Proposed Method Status

未设计或实现正式方法。Ordinal/CPIP不构成已确认创新；原工程说明书继续作为Candidate Design v0。**Formal experiments：Not started**。

## Current Evidence

- 文献证据：颜色代理与生理测量任务不同；水果序数/概率成熟、苹果连续成熟与光照处理均已有先例。
- 数据元信息：D1、D2均有CC BY 4.0公开入口；D2采集、视觉标注协议和结构证据最完整，但有派生图泄漏高风险；D3 Kaggle列GPL 3且约62.36 GB，作者论文确认多视角/重叠采集，原来源许可链与分组仍需包级确认。
- 代码证据：静态核验了官方通用框架及部分作者数据/推理仓库；存在链接不等于完整训练可复现。
- **本项目实验证据：无。** 所有外部论文数字均为其作者报告，未在本项目复现。
- D2原包哈希已复核、2812图完整解码；manifest记录图像哈希、类别、逐边证据与候选source group。真实字节重复跨split已确认；100 seed × 两比例的group-aware dry-run可维持图内确认/支持关系完整，但Candidate边及fruit/tree/session独立性仍待验证。

## Open Questions

1. D2包内1406非resize图像与2575区域（2574有效类别）为何与外部1124/2573/2754等数字冲突？实际来源链如何解释？
2. 29条跨组Candidate已加候选guard约束但尚无物理采集身份；如何确认严格Policy B、6个无base组的处理和7个标注冲突组的派生GT？缺少fruit/tree/session ID带来何种主张边界？三对resize虽已决定排除，来源机制仍未知。
3. 基线主要失败来自定位、标签歧义、跨级判断还是光照？
4. 天然光照分组与外部标签兼容性是否足以支撑泛化主张？
5. D3原始两来源的再利用许可与Kaggle打包许可是否一致，能否恢复fruit/tree/date/sequence分组？

## Risks

颜色被当作生理真值；**D2官方split已有确认的跨集重复**，且相同图像的polygon并不总一致（现识别10条SHA冲突关系/7组）；test一条无类别区域、三对original/resize内容/坐标异常；fruit/tree/session分组未知。D2的1124/1406、2573/2574/2754及比例差异未解释。D3同果多视角与SfM重叠、二类外部标签强映射、合成扰动替代真实光照泛化亦为后续风险。详见[原包审计](docs/dataset-audit-d2.md)。

## Decisions

- 用户已确认：进入Stage 1、暂不采集、使用公开数据；停止Skill系统优化；本轮不下载、训练或实现候选模块。
- 已执行：AGENTS Git/工作树策略最小更新；Stage 1研究资产与记录建立。
- 助手建议：工作任务定义、候选清单、Ordinal/CPIP Modify、IOR保留工作名称；不标为用户已冻结研究路线。
- Stage 1验收时的判断：10项产物条件已满足；当时Stage 2只推荐、未开始。
- 用户已确认进入Stage 2；本轮Dataset Feasibility Gate已通过。D2/D3/D1的角色是证据驱动建议，Dataset仍Not Frozen，需下载后的包级审计才能确认。
- Stage 2B-1结束时，用户已可见下载并首次验证D2原包；当时只准备了ZIP结构命令。D2仍Not Frozen。
- 用户进一步确认Stage 2B-2低成本只读审计；已执行真实ZIP图像头和JSON检查，证据与尚未解决风险记于审计文档。D2保持Not Frozen；未建立或冻结Baseline。
- 用户确认Stage 2B-3只读泄漏与完整性审计；已确认官方split有3组跨集字节重复，因此**不得直接沿用官方split做正式Baseline/最终评价**。本轮不创建替代split，D2仍Not Frozen。
- 用户授权Stage 2B-4同源组解析与协议设计；已生成可复算候选source groups及100 seed × 两比例模拟。方案**Simulation Only / Not Frozen**，未创建正式split，需复核Candidate关系、评价图像选择及Unknown region处理后才可作正式决定。
- 用户授权Stage 2B-4V独立复核；已从原包重建图、压力测试阈值及500×2次模拟，并登记两条视觉桥边和仍未解决的来源风险。未授权、也未进入Stage 2B-5；D2仍Not Frozen。
- 用户已完成Stage 2B-4H的21案人工裁决。C01/C02合组，B01/B02及S01–S10接受，D01–D03保留标注冲突，R01–R03排除dataset-provided resize，U01仅将未知区域排除出派生有效三类目标。已重建候选source groups；Raw Data与正式split未改变，D2仍Not Frozen。
- 用户授权Stage 2B-5P仅准备可冻结候选协议。已构建独立于source group的guard clusters、候选representation manifest与单次确定性模拟；严格Policy B及无base组处置是助手建议，**尚非用户确认的最终Freeze决定**。未产生正式split或选择最终seed。
- 用户随后明确确认Stage 2B-5：严格Policy B、6无base组排除、U01普通背景FP语义、70/15/15、SHA+固定上下文seed以及guard cluster划分规则。已生成冻结manifest并通过两次逐字节重建及关系检查，**D2协议与划分FROZEN**；此前2B-5P条目保留为当时状态。
- 用户授权Stage 3A仅准备E01：以冻结manifest转换出1099张逐实例分割派生集，确定YOLO11n-seg、固定训练配置与运行记录，验证后交给用户在Windows RTX 3070上可见执行正式训练；不得按模型性能改变冻结split。本机MacBook Air M5/16 GB用于开发、转换与极小CPU/MPS smoke；无独显联想机不作训练平台；RTX 3070工作站稍后用于全部正式训练、验证与最终测试。所有跟踪脚本和配置使用跨平台相对路径或`pathlib`，Mac不安装CUDA。

## Rejected Ideas

尚无被实验否决的技术方案。当前不采用的**论证方式**：红色等于真实成熟；添加Ordinal即创新；添加亮度/HSV/gamma即创新；联合检测成熟是未探索任务；跨论文mAP直接排名；多来源混合训练等于独立外部验证。拒绝这些表述不等于拒绝相应技术的潜在实用价值。

## Next Actions

Stage 3C已达到**E01 EXECUTION = FROZEN；E01 READY TO RUN，不是COMPLETED**。RTX 3070可用后，用户先按[工作站清单](docs/e01-workstation-checklist.md)和[Windows步骤](docs/windows-e01-training-setup.md)完成11步交接；第10步人工检查通过后，才在VS Code Terminal可见启动正式训练。现场CUDA、官方权重和batch8显存可行性尚未验证，任一项失败均停止并记录。最终Test只在训练、固定best、Val和协议锁定后一次运行。尚无正式训练或结果，不进入Stage 4。
