# Bounded Review Protocol and Coverage

检索/核验日：2026-09-18。主题：Deep-learning-based apple maturity / ripeness detection under real orchard conditions。窗口：2021-01-01至2026-09-18；必要时向前追溯（P09在线2020、卷期2021）。本文件与检索过程同步整理，不声称事先注册。

## Questions / scope

对应用户RQ1任务、RQ2成熟真值、RQ3模型与baseline、RQ4序数、RQ5光照。核心优先RGB、自然果园、逐果成熟/生长阶段；纳入少量物候、单类苹果检测、生理回归和其他水果以检验假设边界。非苹果证据必须标明可迁移性限制。

纳入：可核验正式论文或作者预印本；直接苹果成熟相关方法/数据，或能实质改变Ordinal/CPIP判断的邻域研究。仅摘要可读仍可纳入有限证据，但不得推断未读取的loss、split、许可或实验结果。

排除：仅品种识别/病害分类、与视觉无关的gamma辐照、无可核原始出处的二手结论、营销页面；重复预印本/正式版合并。无法核验的高相关标题列pending，不混入核心数量。

## Retrieval

- 使用联网搜索，组合 apple / fruit、maturity / ripeness、classification / detection / ordinal / regression、orchard、illumination / color constancy / photometric / corruption等词；完整保存的44条实际查询见 [JSON日志](search-log.json)。查询包括无结果/低相关结果，不是每条均产生纳入文献。
- 使用出版社（Frontiers、MDPI、Springer、Elsevier）、CVF正式会议、arXiv、作者机构仓库；通过论文引用和data availability追踪数据与GitHub。
- 辅以Crossref REST元数据核题名/DOI；Figshare API核许可/文件入口；Europe PMC fullTextXML读取P05；GitHub API只读许可和根目录。不下载研究数据、代码包或模型。
- 额外API题名检索：GhostGS and ECA2、DDCA-Net、multi-stage pixel-level apple、self-supervised occluded apple、fine-grained coloration。前两及部分请求429；multi-stage与coloration定位成功。后续直接DOI核验P02/P08/P10成功。
- P05全文入口：https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13226782/fullTextXML 。PMC网页验证码时使用同论文全文XML，没有将验证码页当论文。

## Screening / extraction

以题名/DOI/版本去重，正式版元数据优先；全文细节若来自arXiv v1单独标记。每篇提取任务、场景、标签、模型、对照、指标、局限、代码、访问范围及证据位置；见矩阵。矩阵“未核验”表示本次访问不足，不等于作者未报告。

本轮是目标导向的bounded evidence map，未执行Web of Science/Scopus订阅库全量导出、完整前后向引文穷举或双人独立筛选；不提供伪造的PRISMA总命中数。核心15篇计数可由CSV复核；另有pending/excluded候选，见筛选日志。

## Coverage and stop rule

完成五个RQ的直接证据覆盖，找到反证、数据入口、可用基线与closest work后停止横向扩张。光照问题通过P15补检，避免只找到普通苹果检测论文；序数通过P08/P07补检，避免将未搜到苹果CORAL误读为空白。

覆盖局限：部分MDPI维护/403/429，作者PDF也有访问失败；P02/P08/P10仅出版社索引片段，P12/P13等仅摘要可核。共8篇全文可访问并抽取相关章节，7篇部分证据；不称15篇全部精读。P07是2026预印本，P06新近在线文章，稳定性须后续复核。部分代码仅数据/推理资源，未验证训练可复现。

本轮结论用于Stage 1路线筛选，足以提出Modify/待审计建议，不能支撑强新颖性或优越性声明。后续选定closest work时必须补其完整方法、split和对照，再做具体贡献审查。
