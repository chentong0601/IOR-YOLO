# Stage 1 Code Resource Survey

2026-09-18，静态网页/README/GitHub API核验。未clone、下载权重、安装或执行。API元数据见 [source-metadata.json](../literature/source-metadata.json)。难度为助手基于入口完整度的估计，不是复现结果。

| Repository / 归属 | Paper | Task | Framework | Dataset | License | Checkpoint | Reproduction difficulty |
|---|---|---|---|---|---|---|---|
| [Fuji author repo](https://github.com/zhukeyi-stan/Fuji_Ripeness_And_Size_Estimation)，作者 | P03 | 检测成熟+尺寸 | MMDetection/PyTorch、SAM | Fuji Kaggle | API未识别、根目录无LICENSE；待确认 | README指向Kaggle，未下载验证 | 中至高：根目录有inference.py，未见完整训练入口 |
| [AppleGrowthVision](https://github.com/fraunhoferhhi/AppleGrowthVision)，作者 | P04 | 数据准备/BBCH/检测 | 数据脚本；论文使用多模型 | AppleGrowthVision | API未识别；网页许可不等于代码/数据许可 | 未核验公开训练权重 | 中：prepare_bbch.py、calib_conversion.py；不是已证实完整训练复现仓库 |
| [Scifresh data](https://github.com/Smart-Ag/Scifresh-apple-RGB-images-with-multi-class-label)，P13指向的数据资源 | P13 data statement | 原图与增强图 | 数据资源；无已核验模型实现 | Scifresh | 未识别明确许可 | 不适用/未见 | 高：框XML需向作者索取；不能算FreqViT官方实现 |
| [Apple HSI](https://github.com/EIS-Ressearch-Lab/Apple_maturity_hyperspectral_imaging)，作者 | P14 | Brix/firmness回归 | Python；具体框架版本未逐文件核验 | Essex HSI | GPL-3.0（代码） | 未核验 | 中至高：清理/训练分析目录存在，但与RGB检测不兼容 |
| [BananaRipeness](https://github.com/luischuquim/BananaRipeness)，作者 | P15及2023前作 | 成熟分类/光照数据 | 数据资源；训练框架未核验 | 真实/合成banana | 页面未见明确许可，待核 | 未核验 | 高：数据链接与例图不等于完整训练代码 |
| [Ultralytics](https://github.com/ultralytics/ultralytics)，模型维护方 | P01/P02所用基线；不是其改进模型作者实现 | YOLO检测/分割/分类 | PyTorch | COCO预训练；项目数据需适配 | AGPL-3.0；另有商业授权 | 官方COCO模型；非苹果论文权重 | 低至中：成熟训练验证入口，仍需固定版本 |
| [Torchvision](https://github.com/pytorch/vision)，官方 | 通用Faster R-CNN/ResNet；非苹果专用复现 | 检测/分类 | PyTorch | COCO/ImageNet预训练 | BSD-3-Clause代码；权重使用条件另看模型卡 | 官方模型卡提供 | 中：检测数据/评测适配；crop分类更简单 |
| [RT-DETR](https://github.com/lyuwenyu/RT-DETR)，作者 | RT-DETR/RT-DETRv2；不是FruitProM head实现 | 实时Transformer检测 | PyTorch/Paddle | COCO预训练 | Apache-2.0 | 仓库模型表提供；未下载 | 中：训练recipe与YOLO不同 |
| [coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch)，作者 | CORAL/CORN通用方法 | Ordinal classification | PyTorch | 通用示例；非苹果 | MIT | 未核验苹果权重，不能假定有 | 通用层低；检测集成与公平验证中至高；本阶段不实现 |

## 高相关但尚未找到可核验实现

| Paper | 已检查路径 | 当前状态 |
|---|---|---|
| P01 HRLN-YOLO | 正文、data availability、题名+GitHub | 未定位官方训练仓库；不能把Ultralytics当HRLN代码 |
| P02 BGWL-YOLO | 出版社检索正文、题名 | 未核验；完整页面维护造成覆盖限制 |
| P07 FruitProM-V2 | arXiv v1全文与题名+GitHub | 未定位作者实现；RT-DETRv2仅是base |
| P08 CMF-Net | 出版社§2.5索引正文、题名+GitHub | 未定位作者实现；CORAL库不等于CMF-Net代码 |
| P10 AFGL-MC | 出版社索引正文 | 未定位作者实现；待全文availability核查 |
| P12 self-supervised apple | ORCA作者条目、题名+GitHub | 已有作者论文入口，未找到可核验代码 |
| P13 FreqViT/coloration | 正式页Data availability及跳转仓库 | 已定位数据仓库，未找到方法完整实现 |

本轮没有确认可复现的第三方完整实现，不能据此推断其不存在。下一步若选择某篇作为正式比较，再逐项核对commit、依赖、训练/验证入口、权重来源、split和授权。现在不申请或发送作者邮件。
