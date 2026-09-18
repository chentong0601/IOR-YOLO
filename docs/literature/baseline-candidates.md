# Baseline Candidates — Not Selected

2026-09-18。只做候选筛选，不安装、不运行。以下容量是公开模型大致等级，改变类别数/配置后会变化；速度不引用不同硬件的FPS作横向排名。

| 候选 | 为什么适合/真实使用证据 | 公开代码与复现难度 | 论文Baseline角色 | 容量/速度等级 | Controlled comparison |
|---|---|---|---|---|---|
| **B1 YOLO11n** | P01/P02已用于苹果成熟检测；与候选主任务直接相符 | [Ultralytics](https://github.com/ultralytics/ultralytics)，COCO权重；低至中，须固定版本和recipe | 优先主基线候选；不是因预案指定而自动选定 | 约2.6M，轻量实时取向；端到端延迟待实测 | 同骨干下检验监督/增强最直接；锁定原始普通增强 |
| **B2 YOLOv8n** | P01同数据比较、P04果园检测已有使用 | 同一官方框架与COCO权重；低至中 | 优先独立版本对照，便于检查结论是否仅对某版本有效 | 约3.2M（COCO）；同属nano等级 | 与B1复用数据/评测较方便；模型训练细节仍需记录 |
| **B3 Faster R-CNN ResNet50-FPN** | P04实际使用Faster R-CNN；代表两阶段检测 | [Torchvision](https://github.com/pytorch/vision) 官方训练参考与权重；中等，需适配数据 | 优先异构检测对照；本文选择R50-FPN并不声称P04所有配置完全相同 | 约41.8M（COCO官方配置）；显著重于nano | 相同数据与指标；单独披露预算与预训练差异，不能当单组件消融 |
| B4 RT-DETRv2-R18 | P07使用RT-DETRv2；适合概率成熟路线的后续对照 | [作者仓库](https://github.com/lyuwenyu/RT-DETR)，PyTorch/Paddle；中等 | Transformer检测扩展候选；R18是预算候选，不等于P07已核实的具体骨干 | 数千万参数级，实时取向；具体配置待核 | 足够预算再加入，不与小YOLO假定等算力 |
| B5 ResNet18 crop classifier | P04/P08有ResNet族成熟/物候对照；本候选用于定位与分类错误分离 | [Torchvision](https://docs.pytorch.org/vision/main/models/resnet.html)，ImageNet权重；低至中 | 诊断基线或固定检测器后的分类阶段；**不与检测mAP直接并表排名** | 千万参数级；只算单crop速度会低估多果pipeline成本 | 使用同一原图分组；GT crop诊断与预测crop端到端评估分开 |

优先集合为B1/B2/B3，最终选2–3个取决于数据和预算。若采用mask任务则应重新考虑P05的YOLO11n-seg/Mask R-CNN，不把分割直接塞入box任务比较。

规模依据：[YOLO11官方表](https://docs.ultralytics.com/models/yolo11/)、[YOLOv8官方表](https://docs.ultralytics.com/models/yolov8/)、[Faster R-CNN官方模型卡](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)。以上是参考模型，不是本项目测量结果。YOLO官方较新版本存在不构成追新理由；当前选择依据是任务相关先例和可复现性。

## 后续比较边界

- 优先建立统一数据划分与评估器；同一测试集不参与调参。
- Baseline与候选方法保持同训练数据、预训练政策和开发预算，实际执行配置留档。
- 基线已有HSV、亮度增强必须记录，否则CPIP对照失真；普通增强与语义受约束处理都需在开发集评估。
- 同模型方法效应与不同架构性能比较分开；对跨框架结果同时报告预算差异。
- 后续统一报告定位与阶段表现；是否采用macro-F1、距离误差、严重跨级错误及校准指标，由标签和错误诊断决定，当前不冻结。
- 参数量、FLOPs与实际速度分别测量；端到端延迟包括预处理与后处理。

Git策略：Baseline未冻结，当前不建立永久工作树。不同候选架构、种子或超参也不自动构成独立永久工作树需求。
