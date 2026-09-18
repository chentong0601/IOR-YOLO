# Skill 验证记录

日期：2026-09-18。对本次新增内容运行系统 skill-creator 的 quick_validate.py 校验函数，并用 PyYAML 解析元数据。PyYAML 通过 uv 的隔离临时依赖使用，未安装进项目科研环境。

| Skill | 官方本地校验器 | Scope | 描述检查 |
|---|---|---|---|
| experiment-design | PASS | User | 触发和排除范围明确 |
| research-novelty-review | PASS | User | 触发和排除范围明确 |
| research-paper-review | PASS | User | 触发和排除范围明确 |
| research-systematic-literature-review | PASS | User | 触发和排除范围明确 |
| apple-dataset-audit | PASS | Project | 触发和排除范围明确 |
| apple-model-experiment | PASS | Project | 触发和排除范围明确 |
| apple-research-workflow | PASS | Project | 触发和排除范围明确 |
| cv-reproduction-audit | PASS | Project | 触发和排除范围明确 |

- 八个新增名称互不冲突，并与扫描到的 85 个旧目录/插件缓存/上级目录/管理员位置的 SKILL.md 比对，无新增同名冲突。既有 PDF 同名问题不在本次修复范围。
- 所有新增 frontmatter 可解析，name 与目录一致，description 均含触发及不触发边界。
- AGENTS.md 的八条路由均可定位至对应文件；项目/全局作用域与注册表一致。
- 四个项目技能没有写入用户级目录；四个通用 SKILL.md 中未发现苹果、Fuji、Orchard、IOR-YOLO 或本项目硬件设定。人工复核 SOURCE.md 也无领域专属规则。
- 30 个既有 SKILL.md 与原工程说明书哈希不变；AGENTS.md 和研究记录原文仍为修改后文件的完整前缀。
- SOURCE.md、实验设计 LICENSE、注册表及清单均存在；文件哈希与清单一致。
- 未执行两个候选第三方仓库的脚本，未克隆或安装全套依赖，未删除任何旧技能，未更改全局配置。
- 人工检查注册表中的八个正向场景与描述一致；负向边界包括单篇不触发系统综述、无具体 idea 不作创新性结论、仅浏览仓库不作完整复现、其他领域不触发苹果技能。

## 验证边界

这是结构校验与静态路由审查，没有在八项真实科研任务上做端到端行为评测，也未证明运行时对每种自然语言输入都会自动选择预期技能。Active/Candidate 保持原状态，不标记 Validated。

官方说明支持通过 description 隐式选择，并自动检测新增技能；下一轮可使用，未出现时重启 Codex。注册表的状态只是管理信息，不等于运行时禁用开关。依据：https://developers.openai.com/zh-Hans/docs/build-skills 。
