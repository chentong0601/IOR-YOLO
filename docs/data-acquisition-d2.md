# D2 v4 — Controlled Acquisition Record

更新：2026-09-19。Stage 2B-2已获取原包并完成ZIP/JSON结构审计；Stage 2B-3又独立复算原包SHA256、完整解码2812张图及核对跨split重复。**D2 = NOT FROZEN**。以下表格保留获取时的来源/首次验证记录；后续实测结论与风险以[原包审计](dataset-audit-d2.md)的Stage 2B-3节为准。Raw Data未改动。

| 字段 | 记录 | 证据 / 状态 |
|---|---|---|
| 正式名称 | A Multi-Stage, Pixel-Level Annotated Apple Dataset for Precision Agriculture Research | [Mendeley Data v4](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| 版本 | v4，发布于 2026-05-09 | [Mendeley Data v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[DataCite DOI 元数据](https://api.datacite.org/dois/10.17632/gfcmdbvw65.4) |
| 官方数据源 | Mendeley Data 的该数据集 v4 页面 | [官方页面](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| 对应论文 | Wang, D. & Wang, B., *A multi-stage, pixel-level annotated apple dataset for precision agriculture research*, Data in Brief 66 (2026), 112856 | [论文 DOI](https://doi.org/10.1016/j.dib.2026.112856)、[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 数据 DOI | `10.17632/gfcmdbvw65.4` | [官方页面](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| License | 数据集 **CC BY 4.0**；SCI研究可按许可署名引用使用。论文的开放获取条款不代替数据许可 | [Mendeley v4许可](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[DataCite rights](https://api.datacite.org/dois/10.17632/gfcmdbvw65.4) |
| 官方下载方式 | 在 v4 页面选择 **Download All**，由用户在浏览器可见下载；不使用镜像 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| 官方单文件名 | **Unverified**：公开网页当前未可靠显示文件列表；本地实际下载文件名见下，不把浏览器生成的包名冒认为平台预告的单文件名 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)；用户本地实测 |
| 官方压缩包大小 | **Unverified**：DataCite未提供字节大小，网页未可靠显示文件大小。论文 Table 2 写数据文件夹约 **279 MB**，但其列出的图片子目录71 MB与213 MB相加已超过279 MB；这些近似值不能用于验证压缩包 | [论文 Data Description/Table 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 图像数 | Mendeley同页写 **1124 apple images** 与 **1406 annotated images**；论文及随包README写1406。本包非resize目录实际**1406 JPG + 1406 JSON记录**，resize还有同名对应1406 JPG。1124统计口径仍**Unresolved**；不能把两套尺寸合计2812张独立图像，也不猜测差额原因 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)、[原包审计](dataset-audit-d2.md) |
| 类别数与语义 | **3**：immature（绿色）、semi-mature（绿红转色）、mature（红色占主）；属于视觉颜色阶段，不是Brix/硬度/淀粉生理真值 | [论文标注协议](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 标注格式 | VGG Image Annotator（VIA）导出的 **JSON polygon**，逐果实例阶段 | [论文 Data Description / Annotation](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| 实例数 | 论文增强后2574，Mendeley v4为2573；随包README Key Features写2754、Statistics写2574。原始三份JSON共**2575区域**，其中三类名称合计**2574**（921/814/839），另有test一条类别缺失。2754与2573的来源原因仍**Unresolved discrepancy**；不得删改或补标来凑数 | [原包审计](dataset-audit-d2.md)、[Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4) |
| Offline augmentation 风险 | 论文说明离线增强。实际train/val含`_brightness`/`_noise`；归一化后67个候选来源名族跨原始split，跨集存在大量完全一致多边形列表。**强支持潜在派生图泄漏；像素/来源身份仍待核实** | [原包审计](dataset-audit-d2.md)、[论文 §§4.4–4.5](https://pmc.ncbi.nlm.nih.gov/articles/PMC13226782/) |
| Dual-resolution / duplicate 风险 | 六个目录JPG与JSON按文件名逐一对应；resize既有640×480，也有480×640。原图头另有三张非常规尺寸；三个文件对应polygon缩放残差超过2像素。需要核查图像身份、方向及异常坐标，不能把两版本当独立样本 | [原包审计](dataset-audit-d2.md) |
| 下载日期 | **Unverified**：2026-09-19为本次报告/登记日期，不等于实际下载日期 | 用户尚未提供下载时间 |
| 原始下载文件名 | `dataset-20260508.zip` | 用户在VS Code/本地终端完成首次验证后的报告 |
| 实际文件大小（bytes） | **290131787** | 用户首次验证值；本轮本地只读 `stat` 独立核对一致 |
| 本地 SHA256 | `049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce` | 用户首次验证值；Stage 2B-3已独立复算一致，见[原包审计](dataset-audit-d2.md) |
| 原包格式 / 成员 / 可疑路径 | ZIP；**2826 ZIP members**；`unsafe_paths=[]`（首次路径检查） | 用户首次验证值；成员总数包括可能的目录、图像、标注等，不等于图像数或实例数；`unsafe_paths=[]`不证明成员内容安全/完整 |
| 随包划分比例 | README声称70%/10%/20%；实际非resize图像1041/126/239 = **74.04%/8.96%/17.00%** | [原包审计](dataset-audit-d2.md)；差异原因Unresolved |
| 本地 raw 存储路径 | `IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip` | 用户实际保存位置；raw目录被根 `.gitignore` 排除 |

## 原始包处理边界

原始下载文件保留原名、原字节、只读用途；不覆盖、删除、重命名或在raw内做增强、划分、标签修正。将来若需要解压用于读取，工作副本放在 `IOR-YOLO/data/interim/`，派生图和格式转换放在 `IOR-YOLO/data/processed/`。不得把两个分辨率目录当作独立样本集合。`data/splits/` 和 `data/manifests/` 仅用于经验证后生成的小型、可追溯清单；本轮不创建正式split。

## Stage 2B-2 原包结构只读审计（已完成）

在项目根目录的 VS Code Terminal 运行以下命令。`--structure-only`仅读取ZIP中央目录及本地文件大小；不读取成员正文、不解压、不写文件，也不重新计算已登记的SHA256。输出JSON至终端，含顶层文件成员分布、扩展名、按扩展名统计的图像成员、按扩展名/命名筛出的**候选**标注成员数，以及增强、分辨率、重复basename、README/metadata/class的文件名线索和有界预览。

```zsh
python3 IOR-YOLO/scripts/01_inspect_dataset.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --structure-only
```

用户已运行上述目录模式。本轮进一步运行`04_audit_d2_zip.py`，只读图像头和六份JSON，结果登记于[原包审计](dataset-audit-d2.md)。这些计数不验证全图可解码性或独立采集来源；同名、分辨率及增强提示不等于每一组已证实的像素派生关系。2826不能用于解释1124/1406差异。

## 后续内容审计边界

脚本默认`--archive`模式仍可报告文件名、字节数、SHA256及成员和可疑路径；这次先使用上面的`--structure-only`，不重复读完整包内容。未来解压副本可使用 `--root` 检查图像数、JSON记录、类别字段/取值、实例数、分辨率、空标注、孤儿图与无效多边形；默认图像检查仅验证头部和尺寸，可选 `--verify-decode` 以Pillow完成逐图解码检查。当前未安装Pillow也未运行完整数据扫描，因此真实损坏情况仍为 **Unverified**。`03_check_leakage.py` 仅列出SHA256精确重复和文件名推断的可能原图族；缩放/增强与近重复需要后续核查，不据此宣称无泄漏。

**仍未解释的来源差异**：①Mendeley 1124与包内1406的口径；②Mendeley 2573、论文/README Statistics 2574、README Key Features 2754；③README声称70/10/20与本包实际比例。包级统计证实1406非resize图、2575区域（其中2574个具有效类别名），但不能猜测来源不一致为何发生。不得删除或改写图像/annotation来凑数。任何正式split或信任官方split之前，先核对原图、双分辨率、离线增强的source group并审查fruit/tree/session信息。

论文 Table 2 的近似数据大小分项与总数也不完全自洽。实际下载包的字节数和文件清单以官方页面或本地原包记录为准，不能用论文的约279 MB预设校验值。

## D2 Freeze Gate（未来，不在本轮通过）

必须同时满足：①source/version；②license；③raw包SHA256；④完整可解析；⑤image/annotation/instance统计；⑥所有来源数字差异解释或正式登记；⑦三阶段视觉标签语义；⑧离线增强关系；⑨双分辨率/重复关系；⑩无未知严重标注损坏；⑪可靠split前提；⑫支持目标检测任务。当前已有来源/许可、哈希记录和初步图像头/JSON统计，但全图解码、类别缺失处理、坐标例外、增强跨split、可信split前提仍未解决。**D2 = NOT FROZEN**，不得启动正式Baseline。
