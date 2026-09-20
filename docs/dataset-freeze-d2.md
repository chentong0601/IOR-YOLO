# D2 v4 — Stage 2B-5 Dataset Protocol & Group-Aware Split Freeze

冻结日期：2026-09-20。**D2 Dataset Protocol = FROZEN；D2 Split = FROZEN。** 范围仅为固定 D2 v4 包上的视觉三阶段逐果任务、派生目标、实验池及确定性组约束划分；不是 Research Question、Baseline 或生理成熟真值的冻结。原包、原始 JSON 与 source groups 未修改。上游[2B-5P候选记录](dataset-freeze-candidate-d2.md)保持历史状态；本页及[机器可读协议](../IOR-YOLO/configs/data/d2_frozen_protocol.yaml)是当前执行依据。

## 数据依据与适用边界

- 官方数据集 DOI `10.17632/gfcmdbvw65.4`，v4；原包 `dataset-20260508.zip`，290131787 bytes，SHA256 `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`；原包为 CC BY 4.0。完整读取、2812张图像解码及原始包数字见[数据审计](dataset-audit-d2.md)。
- 固定包实际为1406张非 resize 表示、1406张提供的 resize 表示；原始非 resize JSON 有2575个 region，2574个带有效三类视觉标签，1个 invalid/unknown。来源文字的1124/2573/2754及官方比例文字属于 **Documented Source Inconsistency**，原因没有得到解释，不能当作本协议统计分母。
- 现有关系图形成1112个 evidence-based `source_group_id`，29条跨组 Candidate 边（23 dHash、6其他）仅以预防约束连接为1096个 `split_guard_cluster_id`。剩余 Candidate **没有升级为同一物理采集**；两个ID均不证明果实、树、场次或采集事件独立。

## 冻结规则与执行产物

最小划分单位为整个 split guard cluster。仅取每个有明确 plain/base 且无冲突的 source group 的一个确定性非 resize 表示；排除全部 dataset-provided resize、brightness/noise 等离线增强、6个无base组及7个同像素但polygon冲突组（严格 Policy B）。模型输入 resize 和在线增强属于后续训练管线，原包不改。候选池经这些规则成为**1099张图、1099个有图source groups、1088个有图guard clusters**；另8个guard clusters无入池图。不同排除原因按优先规则逐表示记录在[排除清单](../IOR-YOLO/data/manifests/d2_exclusions_frozen.csv)，因此287个离线增强表示中有8个归入更具体的“无base”原因。

U01 `test/IMG_54350.jpg`原始4个region中，零基索引#3无有效类；[派生转换函数](../IOR-YOLO/scripts/12_freeze_d2_protocol.py)只保留前3个有效region，不赋予#3类别，不删整张图或其他实例。manifest写入`derived_target_count=3`、`dropped_raw_region_indices=3`；测试以原包JSON逐项核对。正式评价采用普通背景语义，该区域产生的错误苹果预测按FP处理，不设ignore zone。实际YOLO evaluator的端到端集成核验留到Baseline环境建立时进行；若与冻结语义不符，不得静默运行正式实验，须修复实现并复验，或版本化变更协议。

正式比例**70/15/15**，不以模型性能选择。正式64位 seed **13436313853456744620**，由固定ZIP SHA256字节与 ASCII `IOR-YOLO/D2/Stage2B-5P/guard-greedy-v1` 拼接、做 SHA256、取前8字节大端整数重算；如果依赖只接受32位，使用 `seed64 & 0xffffffff = 3589472428`，两者均记录在配置和逐行 manifest 中。使用已有确定性 `assign_clusters` 算法：固定seed的并列次序，guard cluster按图像数降序，按全局图像比例 L1、三类实例 L1 均值、多类图 L1、cluster L1 顺序贪心放入train/val/test；未重搜seed、未依模型结果调整。

| split | 图像 | source groups | guard clusters | immature | semi-mature | mature | 多类图 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 769 | 769 | 758 | 646 | 297 | 512 | 71 |
| val | 165 | 165 | 165 | 143 | 65 | 115 | 16 |
| test | 165 | 165 | 165 | 126 | 63 | 105 | 18 |
| 合计 | 1099 | 1099 | 1088 | 915 | 425 | 732 | 105 |

| 排除原因（每表示一行） | 数量 |
|---|---:|
| dataset-provided resize | 1406 |
| offline augmentation（其他6组内8张按无base优先归因） | 279 |
| 7个annotation-conflict groups的plain代表 | 14 |
| 6个no-base groups的original表示 | 8 |
| 同源组另一份不同字节plain/base | 6 |
| 合计 | **1713** |

正式池[逐图清单](../IOR-YOLO/data/manifests/d2_experiment_pool_frozen.csv)、[按split排列清单](../IOR-YOLO/data/manifests/d2_split_frozen.csv)只引用ZIP内部路径，不复制、覆盖或更名raw。每行包含DOI/版本、原包身份、协议/脚本版本与脚本哈希、Git HEAD、64/32位seed、比例、组ID、representation和annotation策略、排除/划分、派生实例与类别统计；YAML汇总全局计数及三个CSV的SHA256。当前Git HEAD是生成时的提交上下文，生成器本身含未提交变更，**以脚本SHA和协议版本定位实际代码**，不能仅以HEAD指称重现了本次生成器。

## 冻结校验与风险

从未改动的ZIP两次独立完整重建，四个输出逐字节相同；原包SHA已复核，source-group/guard候选清单从原始关系重新构建并与持久清单完全一致。池内无resize或offline augmentation，7个冲突组与6个无base组均无任何入池表示；U01只含3个派生目标，派生三类总数逐split与分配器一致。精确重复跨split **0**、source group跨split **0**、guard cluster跨split **0**；在正式池中两端均存在的已登记原图关系8条均在同split，各等级 crossing 均为 **0**。另外451条关系至少一端排除，不能把它们当作两端参与评价的零跨集验证；全部29条跨source-group Candidate已包含在guard图中。官方提供的train/val/test仍已确认泄漏，**不可复用其split作正式实验**。

| 冻结产物 | SHA256 |
|---|---|
| `d2_experiment_pool_frozen.csv` | `a4c95aca830b514876495a7cdd36e735861758982b3e1529d6b45006b3092aa5` |
| `d2_split_frozen.csv` | `0ad786432a368badce89acf033813bd7b1e47e27770a03f70fd96dc869eebfd1` |
| `d2_exclusions_frozen.csv` | `92d31b4cd351e6392227d29af331efc174b623d56f98c5fb096e93a4130be20e` |
| `d2_frozen_protocol.yaml` | `09db366f059ac933a68a05ec2c8de40a4df1600b859819ca1507d3cf4d719870` |

Freeze Gate 本轮技术项全部通过：来源/版本/许可与包SHA、完整解析与实际统计、视觉三阶段标签界限、增强/双分辨率排除与异常记录、无未知严重标注损坏、组约束可信划分、逐果定位标签、U01派生目标与确定性双重重建。来源文字数字差异正式登记，未解释其原因。**限制仍然显著**：没有真果实/树/场次采集ID；evidence-based source groups并非独立物理采集；未登记关系仍可能跨split；原包官方split曾泄漏；原始polygon存在已记录异常与annotation conflict；三阶段只代表视觉颜色标签、不能推出生理成熟。冻结仅保障已登记关系的协议内隔离及可重复性，不能宣称完全独立的外部泛化。

本阶段未训练、未下载权重、未开始Baseline。可以**准备** E01 Baseline 的环境、实际evaluator集成核验与受控实现；Baseline本身仍未选择/冻结，正式训练仍按项目“用户终端可见执行”规则推进。
