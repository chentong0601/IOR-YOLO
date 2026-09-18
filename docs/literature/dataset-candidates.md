# Dataset Candidates — Not Frozen

核验日期：2026-09-18。只读取论文、网页和元数据；未下载图像、标签包或权重。下列“可下载”表示有公开入口，不代表已验证压缩包、文件完整性、类别映射或训练许可链条。

## 优先候选：按研究用途选择，不立即锁定

| 字段 | D1 Orchard apple maturity | D2 Multi-Stage Pixel-Level Apple v4 | D3 Fuji Ripeness & Size |
|---|---|---|---|
| Name | Orchard apple maturity dataset | A Multi-Stage, Pixel-Level Annotated Apple Dataset for Precision Agriculture Research | Fuji Ripeness & Size Dataset |
| Source | [Figshare](https://doi.org/10.6084/m9.figshare.29533841.v1)、[API](https://api.figshare.com/v2/articles/29533841)、P01 | [Mendeley v4](https://data.mendeley.com/datasets/gfcmdbvw65/4)、P05 | [作者Kaggle](https://www.kaggle.com/datasets/zhukeyi1/fuji-ripeness-and-size-dataset)、[作者仓库](https://github.com/zhukeyi-stan/Fuji_Ripeness_And_Size_Estimation)、P03 |
| Apple Variety | 未明确核验 | Fuji/Gala | Fuji |
| Environment | 真实果园，多光照/背景 | 陕西果园；2017–2019，多天气/遮挡 | Bologna RGB-D与AmodalAppleSize多视图来源 |
| Images | P01报告2039；1618 train/421 val；原包待核 | 原始1124；增强后1406；双分辨率不算独立样本 | 4027；102+3925（P03 v1） |
| Instances | 未核验 | 原始2108；论文增强后2574，仓库2573，待实包核对 | 16257；922+15335（P03 v1） |
| Classes | 4：Young、Pre-growth、Late-growth、Ripe；顺序未确认 | 3：immature、semi-mature、mature | 2：ripe、unripe |
| Maturity Definition | 开发阶段类别；标注规则/生理校准不明 | 绿/转色/红视觉规则；单人标注、专家复核 | 颜色与采集日期；红/绿二分；无已核验Brix配对 |
| Annotation Type | 检测框（论文）；原始存储格式待核 | VIA polygon JSON，逐实例阶段 | bounding boxes与成熟类别；尺寸估计另用几何信息 |
| License | CC BY 4.0，Figshare API确认 | 数据CC BY 4.0；论文为CC BY-NC，不混淆二者 | Kaggle页面未返回可读许可；原来源许可链待核 |
| Download Availability | Figshare列出7z文件入口；未下载 | Mendeley公开下载入口；未下载 | 作者README指向Kaggle数据/权重；未下载或登录 |
| Detection Labels | 论文证实用于检测，待包内验证 | polygon可导出box；要统一遮挡边界定义 | 论文/作者推理任务提供框；实际格式待核 |
| Classification Labels | 实例类可派生crop标签；非独立分类集 | 实例类可派生；非整图单标签 | 二类实例标签；不能自动映射D1/D2 |
| Known Issues | 类别含义/顺序、品种、原始来源、近重复、独立test均待审计 | 离线增强、双分辨率副本；计数差异；论文split百分比与个数不完全一致；无EXIF/色卡 | 重复观察同果、多视角；日期/年份与成熟类别可能混杂；只有两级 |
| Suitability | 与P01最接近的四阶段主数据候选；标签顺序不清时暂停ordinal | 标注依据更清晰的三阶段主数据候选；须能识别原图与派生关系 | 二分类备选或语义兼容后的外部评价；不能承担多阶段序数主验证 |

D2已包含亮度增强及噪声派生图；不能把全部1406张当独立自然采集样本。论文Table4实际train/val/test为1041/126/239，和标称70/10/20并不严格相等。以上是文档可见问题，**不是已经证明发生数据泄漏**。必须在后续授权审计中检查原图族隔离、分辨率副本和划分来源，再决定是否重新划分。

D1的4级与D2的3级、D3的2级不能仅按名称合并。若无法得到可信标签映射，可以只比较定位泛化，或分别训练/报告，不能拼出虚假的统一成熟基准。

## 辅助与暂缓候选

| 数据 | 来源/环境与规模 | 标签/许可/下载 | 用途与限制 |
|---|---|---|---|
| D4 AppleGrowthVision | [作者页面](https://fraunhoferhhi.github.io/AppleGrowthVision/)，德国两果园；9317立体图，论文dense子集1125/31084 | 图像级6主BBCH+苹果框；数据入口公开；dataset license未核验，网页CC BY-SA不是数据许可 | 物候/定位辅助；不是逐果6级成熟。官网另有777/21934旧计数，须固定版本 |
| D5 Scifresh RGB multi-class | [作者仓库](https://github.com/Smart-Ag/Scifresh-apple-RGB-images-with-multi-class-label)，WA商业园，Kinect V2，800原图+12000增强图 | README称4类框；XML未上传，需联系作者；实例数/类别真义/许可未核验 | P13数据声明指向此处；不是其模型实现。暂缓主数据，不能把12800视为独立样本 |
| D6 DeepPhenoTree-Apple | [P06](https://link.springer.com/article/10.1186/s13007-026-01591-w)，4欧洲果园，摘要48320图/精选808 | BBCH物候；实例单位、数据许可与下载待核；文章许可不替代数据许可 | 新近候选，当前不列入优先三组；先核数据内容 |
| D7 Apple hyperspectral | [Essex资源](https://researchdata.essex.ac.uk/228/)，P14多国多季节5756果实 | 配对firmness/Brix，非RGB逐果检测；类别/box不适用；数据许可另核 | 真值设计背景或未来改题，当前不作为RGB检测数据 |

## 后续审计准入问题（未执行）

1. 原始许可允许哪些再利用？数据版本、文件清单、来源链能否固定？
2. “成熟”是视觉颜色、生长阶段还是同果配对生理真值？类别顺序有无原始说明？
3. 原图、增强、裁剪、分辨率与多视角是否能关联到同一采集单位？能否按果实/树/日期/果园分组？
4. 每类独立采集单位是否充足？有没有天然光照属性，或需要仅在开发数据进行人工审计？
5. 若元数据缺失，采用什么保守去重与划分，并怎样限定外推结论？

主数据选择尚未作出；不存在“已冻结数据集”或“已取得可用检测标签”的本地事实。
