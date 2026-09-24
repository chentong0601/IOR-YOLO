# Stage 3A — E01 baseline reproduction protocol

更新：2026-09-20。**E01 任务：instance segmentation；主模型：官方预训练 YOLO11n-seg。** 本阶段只准备与小规模验证，尚无正式训练或指标。固定数据协议见[Stage 2冻结记录](dataset-freeze-d2.md)，实验参数见[机器可读配置](../IOR-YOLO/configs/experiments/e01_yolo11n_seg.yaml)。

## 任务和对照边界

| 形式 | 与原始 D2 标注关系 | E01 取舍 |
|---|---|---|
| Object detection | 从polygon推box，丢掉逐果轮廓；后续仍可按框分析阶段错误，但失去边界质量与mask误差证据 | 可作为后续独立兼容性对照，**不作为本次主E01** |
| Instance segmentation | 原始逐实例polygon可直接转换，保留逐果mask、阶段类别与派生box；Ultralytics官方支持YOLO11n-seg的训练、验证和box/mask分别计分 | **E01主任务**；更贴近数据形成方式，利于SCI论文说明标注到监督的转换及后续严重跨级错误分析 |

E01固定 `yolo11n-seg.pt` 官方预训练权重作迁移学习，不改结构、不加Ordinal/CPIP或自定义loss。YOLOv8n-seg只保留为后续有明确理由时的兼容性baseline，不在E01同时启动。选择依据为[Ultralytics YOLO11任务/权重表](https://docs.ultralytics.com/models/yolo11/)与[分割标注格式](https://docs.ultralytics.com/datasets/segment/)。Ultralytics包为AGPL-3.0体系，若未来发布包含其代码或提供服务，应另行核对许可；本阶段不发布软件。

## 冻结数据与派生实现

[转换器](../IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py)要求固定ZIP、冻结YAML及pool/split/exclusion三个manifest的SHA完全吻合；任何差异直接停止。只按`final_split`复制正式池1099张plain原图JPEG字节，逐图从官方原始VIA JSON读取polygon并写一行一实例的YOLO segmentation label。类别0/1/2固定对应immature/semi-mature/mature；原始坐标按实际JPEG宽高归一化到`[0,1]`，检查有限数、至少3个非退化点、文件一一对应和各split统计。若文件名、标注或冻结计数冲突，报错而不修补raw或冻结manifest。派生输出位于Git忽略的`IOR-YOLO/data/processed/d2_e01_ultralytics/`，可删除重建；`dataset.yaml`与图像/标签同目录，省略`path`键，让Ultralytics以该YAML所在目录作root，不含跟踪的绝对路径。

| Split | 图像 | immature | semi-mature | mature |
|---|---:|---:|---:|---:|
| Train | 769 | 646 | 297 | 512 |
| Val | 165 | 143 | 65 | 115 |
| Test | 165 | 126 | 63 | 105 |

U01 `test/IMG_54350.jpg`按其**原官方定位**读取4个region，派生标签仅含另外3个有效实例；无类#3不给新类别、不设ignore。其正式split位置只由冻结manifest决定，与原官方`test`路径名无关。评估器若在原#3区域报错苹果预测，按普通FP处理。YOLO实际evaluator端到端的该行为必须在正式训练前或首轮验证环境中核验；模型无关探针和转换测试不冒充正式框架验证。

## 固定训练设置与设备分工

固定Python 3.11、Windows `torch==2.5.1+cu121`、`torchvision==0.20.1+cu121`、Ultralytics `8.3.220`。软件组合在正式RTX 3070上仍须通过预检；不以Mac MPS吞吐确定正式超参数。RTX 3070承担正式train、val、最终test；Mac M5只承担代码/数据/配置校验和必要CPU/MPS极小smoke；无独显联想Windows机仅按需做路径/PowerShell兼容检查。PyTorch官方提供该[版本与CUDA 12.1 wheel组合](https://docs.pytorch.org/get-started/previous-versions/)。不在Mac安装CUDA或NVIDIA包。

单一训练seed为 **0**，即固定Ultralytics默认seed；与冻结dataset split seed `13436313853456744620` 完全不同，后者不会因模型结果再选择。100 epochs、640输入、batch8、**optimizer=auto**、请求值`lr0=.01`、weight decay .0005、线性LR至`lrf=.01`、3 epoch warmup、patience100、workers4、deterministic=true、AMP=true、cache=false。核对安装包证实auto才是8.3.220默认；trainer会忽略请求的lr0/momentum并选择实际优化器，详见[8.3.220默认与更正](e01-ultralytics-83220-defaults.md)。batch8与workers4是预先声明的3070/Windows资源选择，不据本项目实验结果调优。若正式训练前短smoke出现CUDA OOM，仅按预定义的**8→4**显式记录和执行，不运行多个batch择优。100 epoch是最大预算，Val fitness决定best；Test不参与。正式执行会保存预训练resolved配置、观测到的实际optimizer/LR、Ultralytics `args.yaml`、best/last权重及run manifest。

在线augmentation全部显式固定：`hsv_h=.015`、`hsv_s=.7`、`hsv_v=.4`、`degrees=0`、`translate=.1`、`scale=.5`、`shear=0`、`perspective=0`、`flipud=0`、`fliplr=.5`、`mosaic=1`、`mixup=0`、`copy_paste=0`、`copy_paste_mode=flip`、`close_mosaic=10`。这些是Ultralytics 8.3.220默认/标准策略的显式记录；无CPIP，也不使用D2提供的离线brightness/noise或resize表示。参数依据可对照[官方训练与增强文档](https://docs.ultralytics.com/modes/train/)及锁定wheel的`default.yaml`。未来CPIP须与E01相同基线条件独立比较。

Train用于拟合；Val用于训练监控、early stopping及确定最终best checkpoint；**Test在best确定后只做一次正式评价**，不得用于调epoch、增强、loss、架构或阈值。模型的正式box与mask指标分别保存：Precision、Recall、mAP50、mAP50–95及每类mAP；Ultralytics plots可保存confusion matrix、PR/F1曲线。二者不可混称。预测记录预留`image_id,pred_instance_id,pred_class,confidence,box,mask_reference,source_group_id,split_guard_cluster_id,split`，配对文件另存GT类别与box/mask IoU，按[结果分析协议](e01-analysis-protocol.md)分析邻级/严重跨级错误；E01不实现ordinal loss，也不把尚未运行的指标当结果。框架[验证模式](https://docs.ultralytics.com/modes/val/)支持显式`split=val/test`；[性能指标文档](https://docs.ultralytics.com/guides/yolo-performance-metrics/)区分box/mask。

## 运行记录和停止条件

[run manifest模板](../IOR-YOLO/configs/experiments/run_manifest_template.yaml)包括Git提交/dirty状态、主机/OS、Python/Torch/Ultralytics/CUDA/GPU、固定数据SHA、模型与权重SHA、seed、训练/增强实际参数、best/last、box/mask验证与测试指标、输出目录及备注。正式输出在Git忽略的`IOR-YOLO/runs/e01_yolo11n_seg/seed_0/`；训练结束以框架`args.yaml`记录真实配置，后续按需将小型summary和论文表格版本化。生成器/runner仅使用repo-relative定位及`pathlib`，可在Mac和Windows运行。

Stage 3C执行冻结后，官方`yolo11n-seg.pt`仅作初始化权重；工作站获取失败必须停止，不得改用随机初始化。权重文件名、Ultralytics来源/版本、SHA256及文件时间在训练前核验，正式manifest再次记录。正式训练要求E01脚本、配置、requirements和冻结manifest均来自已提交HEAD；用户明确知道的无关工作区变化可以存在并记录，但E01相关路径有未提交内容时拒绝启动。Windows只接收固定raw ZIP并核验SHA，派生集必须现场重建。完整交接次序见[工作站清单](e01-workstation-checklist.md)。

实际验证（2026-09-20）：两次临时目录重建的派生文件逐SHA一致；Mac上的Ultralytics 8.3.220官方`YOLODataset`依次加载train/val/test **769/165/165**图、**1455/323/294**个目标，无background或corrupt记录，YAML解析类别0/1/2准确。`yolo11n-seg.yaml`在Mac Python3.12.14、Torch2.5.1下完成构建及一次64×64合成CPU前向；MPS可用，但没有运行MPS训练，也没有下载预训练权重。27项项目低成本测试通过。Mac的Python3.12 smoke只证明开发环境兼容，**不替代正式Windows Python3.11+CUDA的现场预检**。

Stage 3A Gate：A–G（冻结数据、确定性派生、格式/类别/计数、已列关系隔离、U01派生标签）已验证；H–J（唯一主模型/固定配置、训练seed、在线增强）已固定；K（Windows环境）已有版本化安装步骤和必须通过的现场CUDA检查，**实际RTX 3070尚未到手，现场结果待执行**；L–N（run manifest、命令、最终test纪律）已备。`IOR-YOLO/runs/e01_yolo11n_seg/seed_0/`将保存Ultralytics原生`weights/`、`args.yaml`、`results.csv`和图表，入口脚本另准备`metrics/`、`predictions/`、`figures/`及`run_manifest.yaml`；整个目录被Git忽略。

Stage 3A通过的含义是**E01 READY TO RUN，不是COMPLETED**：代码、数据及运行协议准备完成，工作站上线后先过[Windows预检](windows-e01-training-setup.md)，再可由用户亲自在VS Code Terminal启动。CUDA driver/版本、真实GPU与显存、预训练权重获取、派生集重建或config有任何缺项，均不得启动正式train。实际YOLO evaluator对U01普通FP语义仍需在首个真实框架评价环节核验；若不符合则停止并修正实现，不静默变更冻结annotation policy。
