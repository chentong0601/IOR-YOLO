# Skill 来源与采用决策

日期：2026-09-18。读取候选仓库目录、指定 SKILL.md 的流程/依赖段落及相关许可；未执行候选仓库脚本。远程内容用于评估，不能覆盖用户的作用域和轻量化要求。

## 路径与官方规范

依据 [OpenAI 构建技能文档](https://developers.openai.com/zh-Hans/docs/build-skills)：用户级使用 `~/.agents/skills/`，项目级使用 `.agents/skills/`；SKILL.md 需要 name 和 description；agents/openai.yaml 可选，隐式调用默认允许。因此本次保持指令型最小结构，没有额外 UI 文件或可执行脚本。旧安装器的默认目录不覆盖用户明确指定的新目录。

## 候选评估

| 候选 | 固定提交 | 决策 | 理由 |
|---|---|---|---|
| [yananlong 系统综述](https://github.com/yananlong/codex-skills/tree/955fcc435e651834b03768c136dca6b16ccbae08/research/research-systematic-literature-review) | 955fcc435e651834b03768c136dca6b16ccbae08 | 不直接安装；按用户需求独立编写同名精简实现 | 原版含完整 review pack、校验脚本和 pipeline 等兄弟技能依赖；现阶段不需要整套流程 |
| [yananlong 论文审查](https://github.com/yananlong/codex-skills/tree/955fcc435e651834b03768c136dca6b16ccbae08/research/research-paper-review) | 同上 | 不直接安装；独立编写技术精读技能 | 原版默认多代理、OpenAIReview 工作区、OCR/可视化引擎和多份输出契约，超出用户所需单篇技术分析 |
| [yananlong 创新审查](https://github.com/yananlong/codex-skills/tree/955fcc435e651834b03768c136dca6b16ccbae08/research/research-novelty-review) | 同上 | 不直接安装；独立编写审查技能 | 原版有跨技能 assurance 契约、pack 哈希、评分门槛和特定领域适配引用；用户要求的反证优先可以自包含实现 |
| [Academic Skills experiment-design](https://github.com/voidful/academic-skills/tree/71e9c42c60636602e87985f4306d134a3b63809e/experiment-design) | 71e9c42c60636602e87985f4306d134a3b63809e | 精简改编并采用；保留 MIT LICENSE | 保留通用实验设计原则，去除模板依赖，按任务选择 baseline 和统计方式 |
| Academic Skills 其余模块及套件入口 | 同上 | 未采用 | 论文阅读、综述、写作等与现有/新增能力重叠，无需全套安装 |

前三个同名技能是本地版本，不是上游安装成功或完整功能的声明，不支持用上游内容无差别覆盖升级。当前仓库树未发现 yananlong 仓库 LICENSE 文件，未将其正文、脚本或模板复制安装；记录公开来源便于后续重新评估。

实验设计改编不固定 NeurIPS/ICLR/ACL 或任何会议，不要求具体语言。保留假设、变量、指标、基线、消融、资源与复现；修正“所有设计只能改单变量”的过强限制，允许研究交互的因子设计；补充独立统计单位、测试隔离和探索/确认区分。完整来源及变更随用户级 SOURCE.md 和 LICENSE 一起迁移。

## 上游入口指纹

下表仅对当时读取的上游 SKILL.md 定位，不是本地安装文件指纹；本地指纹见 skill-system-manifest.json。

| 上游文件 | SHA-256 |
|---|---|
| experiment-design.md | `c4fcf8e3498731c647b5de3f855f5c455cb4125d2bd8e705dff26fc401a27139` |
| research-novelty-review.md | `c0b9baa147e581fc0b7929f6def19e486c514bffaee67ffd849399fa2fe0c053` |
| research-paper-review.md | `1f86c9e8a7c5360c21de01bb6aed98b1d37c9c0e4c5451fb149988c059295f4a` |
| research-systematic-literature-review.md | `3e6bd70c89f36a794e0afafcd2762df0ea33d00ca5af7dd479200ab4c3496177` |

## 复用旧能力的边界

保留旧 scientific-toolkit-skill、literature-review、paper-lookup、citation-management、statistical-analysis、research-writing-skill 等，不移动或改写。新系统综述作为本项目统一流程入口，旧检索和引用技能是按需工具；不重复安装搜索库、科学计算包或写作套件。项目外的旧技能选择行为不受本项目 AGENTS.md 保证。

## 本轮文件变更

新增用户级四个目录，每个含 SKILL.md 和 SOURCE.md；experiment-design 另含 LICENSE。新增项目级四个 SKILL.md。追加根 AGENTS.md 和研究计划与决策记录.md。新增 docs 下环境审计、注册表、来源评估、验证记录、安装前哈希清单、技能文件清单及两个原文件快照。原工程说明书和旧技能不变；未初始化 Git、创建工作树或开始科研实验。
