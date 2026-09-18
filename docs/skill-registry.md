# Research Skill Registry

日期：2026-09-18。此文件管理本轮八个科研 Skill，不代表全机只有八项技能。旧技能盘点见 [环境审计](skill-environment-audit.md)，第三方取舍见 [来源评估](skill-source-review.md)，结构校验见 [验证记录](skill-validation.md)。

新用户级根目录 `~/.agents/skills/`；项目级根目录 `.agents/skills/`。技能采用最小 SKILL.md 结构，未设置禁止隐式调用的策略。描述匹配是语义选择，不是固定关键词自动执行器；AGENTS.md 提供路由，注册表状态不控制运行时加载。

| Skill | Scope | Source | Version / Commit | Status | Trigger | Do Not Trigger | Current Project Use | Candidate for Global | Notes |
|---|---|---|---|---|---|---|---|---|---|
| research-systematic-literature-review | User | 本地独立编写；已审查 Yanan 候选 | 1.0.0 | Active | 系统文献调研、研究现状、state of the art、related work survey、技术路线分类与 evidence map | 仅解释一篇论文，也不替代具体 idea 的最终创新性审查。 | 调研阶段主入口（待真实任务验证） | 已是全局 | `~/.agents/skills/research-systematic-literature-review/SKILL.md` |
| research-paper-review | User | 本地独立编写；已审查 Yanan 候选 | 1.0.0 | Active | 阅读一篇或少量具体论文，技术分析 architecture、dataset、method、loss、experiment、limitations 和 reproducibility | 大规模领域综述、纯文字润色或尚无具体方案的 idea 生成。 | 最接近论文技术精读（待真实任务验证） | 已是全局 | `~/.agents/skills/research-paper-review/SKILL.md` |
| research-novelty-review | User | 本地独立编写；已审查 Yanan 候选 | 1.0.0 | Active | 审查已形成 research idea、method 或 contribution 的创新性：创新点有人做过吗、closest prior work、novelty check、incremental improvement | 无具体 idea 的头脑风暴、普通单篇解释或单独性能评估。 | idea 成形后及投稿前（待真实任务验证） | 已是全局 | `~/.agents/skills/research-novelty-review/SKILL.md` |
| experiment-design | User | Academic Skills 精简改编 | 1.0.0；上游 71e9c42c60636602e87985f4306d134a3b63809e | Active | 科研实验方案设计：hypothesis、变量、baseline、ablation、metrics、参数敏感性、robustness、统计可靠性与可复现性 | 单独运行既定命令、单篇解读或无实验问题的泛泛讨论。 | 探索/正式实验协议（待真实任务验证） | 已是全局 | `~/.agents/skills/experiment-design/SKILL.md` |
| apple-research-workflow | Project | 本项目原创 | 1.0.0 | Active | 推进苹果成熟度论文、制定完整研究计划、判断下一步、从文献到实验到论文的综合任务或检查研究完成度时使用 | 独立单篇解释、单次训练报错或一般 Skill 管理。 | 当前阶段与全流程衔接（待真实任务验证） | 否，随项目归档 | `.agents/skills/apple-research-workflow/SKILL.md` |
| apple-dataset-audit | Project | 本项目原创 | 1.0.0 | Active | 当前苹果项目涉及图像 dataset、maturity labels、采集/标注、train/val/test、augmentation、类别不平衡、data leakage、domain shift、拍摄条件时使用 | 无数据内容的架构解释或其他领域数据审计。 | 公开数据可用性及泄漏核查（待真实任务验证） | 仅在未来 CV 项目需要时泛化 | `.agents/skills/apple-dataset-audit/SKILL.md` |
| apple-model-experiment | Project | 本项目原创 | 1.0.0 | Active | 当前苹果项目的 CNN、ResNet、EfficientNet、MobileNet、YOLO、Vision Transformer、attention、feature fusion、迁移学习、架构修改、训练、baseline、ablation 和指标分析 | 无关模型任务、纯文献全景或仅检查数据来源。 | 基线、候选 IOR 方法与结果分析（待真实任务验证） | 有条件，generic deep-learning-experiment | `.agents/skills/apple-model-experiment/SKILL.md` |
| cv-reproduction-audit | Project | 本项目原创 | 1.0.0 | Candidate | 复现计算机视觉 GitHub 项目、运行开源 baseline code、核查 pretrained weights 或 reproduce paper results 时使用 | 仅浏览仓库简介、纯论文精读或自行设计新模型。 | 开源基线复现试用（待真实任务验证） | 是，research-code-reproduction | `.agents/skills/cv-reproduction-audit/SKILL.md` |

## 状态与审计

- Candidate：候选试用，尚待真实任务验证。
- Active：可使用，尚不代表已验证。
- Validated：有真实任务产物、触发和排除场景证据，可记录适用范围。
- Deprecated：已有替代或已知问题，保留迁移记录；仅改状态不会停止自动发现。
- Archived：退出活跃目录并随项目留档；仅改注册表不会停用技能。

实际使用日志从首次科研任务开始记录：日期、Skill、输入/任务、产物路径、发现的问题、修订及版本。当前只有安装与结构校验，不登记虚构的科研验证。修改 Skill 时同步版本、来源及校验值；保留真实变更理由。

## 迁移与回退

- 迁移本项目时携带 `.agents/skills/`、AGENTS.md、研究记录和 docs；项目技能不含机器绝对路径。
- 用户级技能各自为独立目录，迁移时从旧机器 `~/.agents/skills/` 复制这四项至新机器同位置，并携带 SOURCE.md 和适用 LICENSE；先检查重名，不覆盖。前三项为本地实现，不能用上游同名技能无差别替换。
- `skill-system-manifest.json` 保存本轮技能文件的 SHA-256 与版本，可验证复制内容；本项目未复制一套活跃全局技能，避免双份发现和漂移。
- 根入口和研究记录改动前原文保存在 `skill-system-before/`。回退时先比较后续改动，只撤回本轮新增区段，不盲目用备份覆盖。
- 本次未删除、禁用、移动或改写旧 Skill，也未修改全局 config.toml。新旧功能重叠通过本项目路由控制，其他项目的全局路由尚未重新治理。

## Skill Promotion Rule

只有同时满足以下条件才建议晋升到 `~/.agents/skills/`：

1. 已在真实任务中实际使用，留下产物和改进证据。
2. 不依赖苹果项目特定知识、数据、指标阈值或路径。
3. 在另一个研究项目仍然适用，并有可说明的使用场景。
4. trigger / non-trigger 已稳定。
5. 与现有全局 Skill 不重复；存在重叠时先合并或明确边界。

晋升须完成领域内容剥离、名称和依赖检查、两类场景验证及注册更新。不能只复制目录并标记成功；避免同名全局与项目级双份活跃。具体迁移按用户授权执行。

## Project Closeout Checklist

逐项选择 KEEP IN PROJECT / PROMOTE TO GLOBAL / REFACTOR / ARCHIVE / DELETE，记录证据与理由。Refine 对应 REFACTOR 后重新验证；DELETE 不是自动清理动作。

- [ ] 盘点真实使用次数、任务产物、触发误报/漏报、维护成本与重复度。
- [ ] apple-research-workflow：预期 ARCHIVE with project。
- [ ] apple-dataset-audit：预期 ARCHIVE；未来 CV 项目确有需要才 REFACTOR 泛化。
- [ ] apple-model-experiment：评估 REFACTOR 为 deep-learning-experiment，满足晋升规则才 PROMOTE。
- [ ] cv-reproduction-audit：重点评估 PROMOTE 为 research-code-reproduction，未达标准则 KEEP/REFACTOR。
- [ ] 四个通用入口：根据实际收益选择 KEEP、REFACTOR 或 Deprecated，避免与旧技能重复维护。
- [ ] 归档项移出 `.agents/skills/` 等自动发现目录，并保留来源、版本和结果索引；需要完全退役的项目不能仍把归档规则作为活跃路由。
- [ ] 删除仅针对明确无价值且用户授权的内容，先记录理由及可恢复副本，不删除研究证据。
- [ ] 同步 AGENTS.md 路由、注册表和迁移清单，检查悬空引用及同名冲突。

## 八个路由示例

以下是预期语义路由案例，不是假称已运行的自动触发测试：

| 用户请求 | 主 Skill | 边界 |
|---|---|---|
| 调研苹果成熟度视觉检测有哪些技术路线，建立文献证据表 | research-systematic-literature-review | 跨论文综合，不自动深审全部论文 |
| 精读这篇论文的检测头、损失和消融，指出复现缺口 | research-paper-review | 需要具体正文，不能据摘要推断实验 |
| 序数监督加光照一致性是否已经有人做过？找最接近工作 | research-novelty-review | 先用已有具体主张，优先反证 |
| 为这个候选方法设计公平对照、消融和统计方案 | experiment-design | 设计协议，不自动开始训练 |
| 看看当前苹果论文进度，下一步应该做什么 | apple-research-workflow | 判断阶段，不机械执行全部流程 |
| 检查 Orchard 数据，同一苹果会不会跨 train/test | apple-dataset-audit | 没有果实 ID 不宣称彻底排除泄漏 |
| 给 YOLO 加序数头是否合理，应该看哪些指标和代价 | apple-model-experiment | 先问题与机制，再控制实验 |
| 复现这个 GitHub 基线并核对论文表格结果 | cv-reproduction-audit | 跑通与复现分开，记录期望及偏差 |
