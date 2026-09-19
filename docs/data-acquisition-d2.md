# D2 v4 — Controlled Acquisition Record

更新：2026-09-19。阶段：Stage 2B-1，**准备下载；未下载；D2 = NOT FROZEN**。本文件记录官方元数据和未来原始包的取证位置，不表示完成数据解析或泄漏检查。

| 字段 | 记录 | 证据 / 状态 |
|---|---|---|
| 正式名称 | A Multi-Stage, Pixel-Level Annotated Apple Dataset for Precision Agriculture Research | [Mendeley Data v4](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| 版本 | v4，发布于 2026-05-09 | [Mendeley Data v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[DataCite DOI 元数据](https://api.datacite.org/dois/10.17632/gfcmdbvw65.4) |
| 官方数据源 | Mendeley Data 的该数据集 v4 页面 | [官方页面](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| 对应论文 | Wang, D. & Wang, B., *A multi-stage, pixel-level annotated apple dataset for precision agriculture research*, Data in Brief 66 (2026), 112856 | [论文 DOI](https://doi.org/10.1016/j.dib.2026.112856)、[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 数据 DOI | `10.17632/gfcmdbvw65.4` | [官方页面](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| License | 数据集 **CC BY 4.0**；SCI研究可按许可署名引用使用。论文的开放获取条款不代替数据许可 | [Mendeley v4许可](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[DataCite rights](https://api.datacite.org/dois/10.17632/gfcmdbvw65.4) |
| 官方下载方式 | 在 v4 页面选择 **Download All**，由用户在浏览器可见下载；不使用镜像 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| 官方单文件名 | **Unverified**：公开网页当前未可靠显示文件列表。下载时保留浏览器提供的原始文件名；不要据论文猜测文件名 | 下载后填入 |
| 官方压缩包大小 | **Unverified**：DataCite未提供字节大小，网页未可靠显示文件大小。论文 Table 2 写数据文件夹约 **279 MB**，但其列出的图片子目录71 MB与213 MB相加已超过279 MB；这些近似值不能用于验证压缩包 | [论文 Data Description/Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 图像数 | Mendeley v4 同一页面分别写 **1124 apple images** 与 **1406 annotated images**；Data in Brief 论文写 **1406 RGB images**。**1124 vs 1406 = Unresolved discrepancy**：尚未根据实际文件核实两个数字的计数单位及对应关系，不以离线增强或双分辨率直接解释差异 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[Data in Brief](https://doi.org/10.1016/j.dib.2026.112856) |
| 类别数与语义 | **3**：immature（绿色）、semi-mature（绿红转色）、mature（红色占主）；属于视觉颜色阶段，不是Brix/硬度/淀粉生理真值 | [论文标注协议](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 标注格式 | VGG Image Annotator（VIA）导出的 **JSON polygon**，逐果实例阶段 | [论文 Data Description / Annotation](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 实例数 | 原始 **2108**；增强后论文 **2574**，Mendeley v4 页面 **2573**。差异 **Unresolved discrepancy**；不得删改任何 annotation 来凑数 | [论文 Table 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/)、[Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| Offline augmentation 风险 | 作者对仅含semi-mature的图像与选定的45张mature图像做HSV Value×1.15、Gaussian noise；文中先描述增强后描述划分，未说明原图族隔离。**高风险，非已证实泄漏** | [论文 §§4.4–4.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| Dual-resolution / duplicate 风险 | 发布两套分辨率：369×277或404×303，以及640×480。需要关联同一原图及其增强派生图，再考虑任何正式split | [论文 Table 2 / §4.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 下载日期 | **Unverified / 未下载**。完成可见下载后填ISO日期与时区 | 待实际获取 |
| 原始下载文件名 | **Unverified / 未下载** | 待实际获取 |
| 实际文件大小（bytes） | **Unverified / 未下载** | 待本地 `stat` 或检查脚本 |
| 本地 SHA256 | **Unverified / 未下载** | 待本地 `shasum -a 256` 或检查脚本；与实际文件名、日期同时登记 |
| 本地 raw 存储路径 | `IOR-YOLO/data/raw/multistage_apple_v4/` | 已按现有 `IOR-YOLO/data/{raw,interim,processed}` 结构选择；raw目录被根 `.gitignore` 排除 |

## 原始包处理边界

原始下载文件保留原名、原字节、只读用途；不覆盖、删除、重命名或在raw内做增强、划分、标签修正。将来若需要解压用于读取，工作副本放在 `IOR-YOLO/data/interim/`，派生图和格式转换放在 `IOR-YOLO/data/processed/`。不得把两个分辨率目录当作独立样本集合。`data/splits/` 和 `data/manifests/` 仅用于经验证后生成的小型、可追溯清单；本轮不创建正式split。

## 下载后第一轮只读审计

在 `IOR-YOLO/` 工作目录运行脚本，传入实际下载文件名；它报告文件名、字节数、SHA256，并在zip/tar格式时只读列出成员及可疑路径。它不解压、不修复数据。对于其他压缩格式，包格式将保持 `unverified`，应先识别格式后制定只读解包步骤。未来解压副本可使用 `--root` 检查图像数、JSON记录、类别字段/取值、实例数、分辨率、空标注、孤儿图与无效多边形；默认图像检查仅验证头部和尺寸，可选 `--verify-decode` 以Pillow完成逐图解码检查。当前未安装Pillow也未运行完整数据扫描，因此真实损坏情况仍为 **Unverified**。`03_check_leakage.py` 仅列出SHA256精确重复和文件名推断的可能原图族；缩放/增强与近重复需要后续核查，不据此宣称无泄漏。

**两项 Unresolved discrepancy**：①1124 vs 1406 图像数；②2573 vs 2574 实例数。后续必须通过实际下载包的文件结构、原始图像、增强图像、双分辨率关系和逐实例标注计数解释；不能解释时继续保留未解决状态。不得删除或改写图像、annotation来强行匹配数字。任何正式train/val/test split之前，先建立原图、缩放版、离线增强版之间的候选source group，并审查是否存在fruit/tree/session元数据。随机按图像划分不能单独作为可信评价协议。

论文 Table 2 的近似数据大小分项与总数也不完全自洽。实际下载包的字节数和文件清单以官方页面或本地原包记录为准，不能用论文的约279 MB预设校验值。

## D2 Freeze Gate（未来，不在本轮通过）

必须同时满足：①source/version；②license；③raw包SHA256；④完整可解析；⑤image/annotation/instance统计；⑥1124/1406与2573/2574两项差异解释或正式登记；⑦三阶段视觉标签语义；⑧离线增强关系；⑨双分辨率/重复关系；⑩无未知严重标注损坏；⑪可靠split前提；⑫支持目标检测任务。当前只满足来源、版本、许可和文献层面的标签语义；**D2 = NOT FROZEN**，不得启动正式Baseline。
