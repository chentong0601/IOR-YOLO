---
name: apple-research-workflow
description: "推进苹果成熟度论文、制定完整研究计划、判断下一步、从文献到实验到论文的综合任务或检查研究完成度时使用。仅限当前苹果项目；不用于独立单篇解释、单次训练报错或一般 Skill 管理。"
metadata:
  version: "1.0.0"
  origin: local-independent
---

# 苹果研究总控

先读项目根目录 `研究计划与决策记录.md` 和 `docs/skill-registry.md`，从本 skill 目录可经 `../../../` 到达项目根目录。该项目是阶段性 SCI 研究，依赖公开数据，暂无自行采集条件；苹果知识不得写入用户级通用 Skill。原工程说明书是候选 v0，不是冻结路线。

识别当前阶段、已有证据、用户本次目标和缺口，只执行当前必要步骤。参考完整路径：

Research Question → Literature Survey → Task Definition → Dataset Investigation → Baseline Selection → Baseline Reproduction → Method Hypothesis → Proposed Improvement → Experiment Design → Main Experiment → Ablation → Robustness / Sensitivity → Error Analysis → Novelty Review → Paper Writing → Pre-submission Audit。

这不是机械单向流水线：数据可用性、最近工作和最小验证可提前交叉进行；创新性检查应在昂贵实验前做一次，投稿前更新。遵循已记录的七阶段计划及后续明确决定。

- 文献不足不得宣称 first、novel、state-of-the-art、no previous work。
- 四阶段究竟是生长阶段还是成熟度，先核验原始标签；光照稳定性、序数学习与域泛化均为待验证方向。
- 先数据审计和最小基线诊断，再决定模块及完整预算；最终 test 不指导研发。
- 需要哪项子能力才读取对应 Skill：系统调研、论文精读、数据审计、模型实验、实验设计、代码复现、创新审查。正文撰写可沿用现有 research-writing-skill。
- 输出当前阶段、证据、关键缺口、下一步及验收依据；重要决定回写研究记录，明确用户确认与助手建议。
- 投稿前检查主张—表图—run/来源的对应、统计可靠性、数据泄漏、失败案例、引用与复现材料；不凭空认证投稿质量。
