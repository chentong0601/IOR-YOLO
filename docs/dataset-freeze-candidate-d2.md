# D2 v4 — Dataset Freeze Candidate Gate（Stage 2B-5P）

历史记录更新：2026-09-20。**本页保留Stage 2B-5P当时的候选状态；用户后来已确认规则并完成[正式Stage 2B-5冻结](dataset-freeze-d2.md)，以新协议为准。** 原始ZIP/JSON、正式模型代码与已有source groups均未修改。本页以下“未冻结/待确认”表述只描述候选审查当时，不代表当前状态。依据为[人工裁决](human-review-stage2b4.md)、[当前同源组](source-group-resolution-d2.md)、[guard/representation设计与单次模拟](split-constraint-design-d2.md)及两个候选manifest。

## 已确认的数据口径和新的完整性发现

固定D2 v4 ZIP SHA256 `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`；实际包内为1406个非resize图像表示、1406个同名resize表示；原始JSON有2575个polygon regions，其中2574个带有效三类视觉阶段标签，1个是U01 invalid/unknown。**不把1406称作独立采集数，也不把视觉颜色阶段称作生理成熟真值。**

本轮重新按ZIP图像SHA和原始JSON比较所有相同字节的非resize图像：发现**10个SHA重复关系组**具有不同polygon列表，落在**7个source groups**。先前D01–D03是其中3个**跨官方split**关系；另外7个关系发生在同一官方split。各对的类别列表和region数均一致，差异在polygon几何，原因未知。对应7个source group ID为`sg-0008`、`sg-0012`、`sg-0013`、`sg-0263`、`sg-0265`、`sg-0269`、`sg-0419`；逐文件及SHA可由[候选生成器](../IOR-YOLO/scripts/10_prepare_d2_freeze_candidate.py)从原ZIP重算。此发现不改变D01–D03的人类决定：**图像重复Confirmed，标注冲突Unresolved**。

## Exact duplicate / conflicting annotation：A vs B

| 项 | Policy A：确定性择一annotation | Policy B：冲突组不进入正式评价；当前严格版也暂不训练 |
|---|---|---|
| 可复现性 | 可按路径/哈希稳定选同一文件，但规则只保证可重复，不保证其polygon更正确 | 从SHA/JSON确定性标出冲突组，暂存规则完全可复算；不用声称哪套GT是真值 |
| 训练影响 | 可保留这些组，但模型被任意一套定位边界监督；若保留另一副本又造成不一致 | 当前严格版少7个canonical代表（相对1106个有base组约0.63%）；未来若核实一致性，可独立讨论是否仅train使用 |
| 验证/测试公平性 | 同一像素图的评价GT依路径而变，mAP和误差解释易受质疑 | 不对有冲突的图打分；不会因该图的任意polygon选择影响validation/test指标 |
| 论文解释 | 必须解释为何一个任意文件名有更高真值地位 | 如实披露SHA重复与annotation差异、暂存数量及对覆盖的影响 |

**推荐Policy B的严格版作为当前候选**：把全部7个冲突source groups暂从train/val/test候选池排除；不平均、不自动修复、不挑一份annotation为真值，不改Raw Data。D01/D02实际上同属一个source group，D03属于另一个；因此不能把“3条跨split冲突关系”误说成“3个冲突source groups”。这一排除规则仍须用户在真正Freeze前确认。完整候选池为**1099张图、1099个有图像source groups、1088个有图像guard clusters**；还保留无base的6组供人工审核，未静默丢弃。候选池三类实例数为**915/425/732**，与原包921/814/839不同，主要因不使用预生成增强及冲突组暂存；未来论文必须使用派生池的实际统计，而非套用原包类别数。

## U01：最小评价器探针与候选规则

Stage 2B-4H已确认`test/IMG_54350.jpg#3`不给成熟类，整图和另外三个有效polygon保留。Raw JSON仍有4个区域；派生三类目标只取另3个，不将#3伪标为背景真值或成熟类。[最小synthetic box-evaluator探针](../IOR-YOLO/scripts/11_probe_u01_evaluator.py)从原始JSON读取几何，用三条合法GT构造3个精确命中预测，再在#3的box处加1个苹果类别预测。标准无ignore匹配得到 **TP=3、FP=1、FN=0**；若把#3当ignore-zone则变为 **TP=3、FP=0、FN=0、ignored=1**。#3 box与三个有效目标的最大IoU仅**0.019236**，该预测并未匹配到其他有效目标。

这说明**直接删掉#3会让该处苹果预测被计为FP**，但不能单靠synthetic test证明这个FP是“错误惩罚”。用户人工复核认为该区域**不是明显苹果实例**；当前较可解释的候选做法是保留图和3个有效目标、#3不作为三类GT、按标准评价器正常计算该处预测（若模型仍报苹果即FP）。不要自动设置ignore-zone遮蔽可能真实的误检；可把ignore-zone结果仅作为预先声明的敏感性分析。正式检测/分割框架尚未选择，本探针**不等于已验证未来实际evaluator的实现**；该集成与测试是Freeze前阻断项。

## 来源文字数字冲突

官方页面/README/论文的1124/1406图像数、2573/2574/2754实例数及划分比例不完全一致。其成因仍**Unresolved**，不得杜撰解释。但固定版本、许可、包SHA、1406张非resize、2575个总region、2574个有效类和1个invalid/unknown均可从原包确定性复算。因此把这项从“必须解释来源文字原因”的硬性Dataset Freeze blocker，降级为**Documented source inconsistency**：方法/数据章节应列清本包实际口径并引用具体版本，不能借来源侧数字当实验分母。若未来换包或来源声明改变，须重新审计。

## Freeze Candidate Gate

| 项 | 状态 | 当前证据与未满足条件 |
|---|---|---|
| A Raw integrity | **PASS** | SHA固定、六份JSON可读、2812/2812图已完整解码；Raw未修改。 |
| B Source-group integrity | **PARTIAL** | 1112组可复算、21案已人审；缺fruit/tree/session/acquisition ID，不能证明所有真实同源关系。 |
| C Candidate leakage containment | **PASS** | 23条跨组dHash Candidate和额外6条跨组命名Candidate都纳入guard；已列关系边的模拟跨集为0。只覆盖**已列**关系。 |
| D Representation policy | **PARTIAL** | 候选池与排除理由已生成；正式排除dataset resize已确认，但base-only训练、6个无base组的fallback/排除及评价集选择尚未冻结。 |
| E Exact duplicate policy | **PARTIAL** | 图像同字节关系已核实，推荐严格Policy B；正式采用与数据损失仍待确认。 |
| F Annotation conflict policy | **UNRESOLVED** | 10个SHA冲突关系/7组已记录；未选polygon真值，严格暂存可绕开评价但冲突成因未解。 |
| G U01 policy | **PARTIAL** | 三有效region保留、#3 invalid；synthetic探针通过，实际检测/分割evaluator尚未集成验证。 |
| H Class semantics | **PARTIAL** | 只支持视觉颜色阶段；若研究主张生理成熟，需要额外客观ground truth。 |
| I Split algorithm | **PARTIAL** | guard完整的透明greedy与两比例单次模拟可行；尚未评审/冻结正式比例和派生池。 |
| J Seed policy | **PARTIAL** | 原包SHA+固定context导出候选seed，避免mAP与多seed shopping；未冻结正式协议版本。 |
| K Evaluation independence | **FAIL** | 官方split已确认泄漏，尚无正式替代split，无法宣称最终test独立。 |
| L Reproducibility | **PASS** | 原包SHA、输入清单、脚本、LF候选CSV和重复构建/合成测试可核；不存在正式实验结果。 |

**真正仍阻断Dataset/Split Freeze的事项**：①用户尚未确认严格Policy B与6个无base组的处置，且新发现7个冲突组需要纳入数据损失/样本覆盖判断；②实际evaluator对U01三有效目标和未知区域的处理未验证；③没有冻结候选池、比例、seed和正式split，最终测试独立性未建立；④缺采集ID，guard只能限制已发现的相关图，论文泛化主张必须收窄并清楚披露剩余风险。来源文字数字的成因仍未知，但在固定包可复算的前提下属于已记录的文档局限，不单独阻止继续准备。**本轮不进入正式Stage 2B-5 Freeze。**
