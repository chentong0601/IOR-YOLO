# Ultralytics 8.3.220 defaults and E01 pretrain correction

核验日期：2026-09-20。证据是本机安装的**Ultralytics 8.3.220分发包**（包元数据版本`8.3.220`、`ultralytics/cfg/default.yaml` SHA256 `28a10d487a250d2b68d472551a306f4d7bf3f5a7be5cc7184732ab8d4dd07d97`）、同包`ultralytics/engine/trainer.py`的`_setup_train`/`build_optimizer`/`validate`/`save_model`，以及`ultralytics/utils/torch_utils.py`的`EarlyStopping`。不是从后来的网页默认值推断。E01变更发生在正式训练前，没有依据任何模型表现。

| 参数 | 8.3.220 default.yaml | E01显式请求 | 分类与实际含义 |
|---|---:|---:|---|
| epochs | 100 | 100 | 显式重复默认；训练epoch**上限** |
| batch | 16 | **8** | **冻结的E01训练预算**，不是默认或性能择优 |
| imgsz | 640 | 640 | 显式重复默认 |
| optimizer | **auto** | **auto** | 显式重复默认；实际优化器须记录trainer决策 |
| lr0 | 0.01 | 0.01 | 显式重复默认；**auto会忽略请求值** |
| lrf | 0.01 | 0.01 | 显式重复默认，调度末端比例 |
| momentum | 0.937 | 0.937 | 显式重复默认；**auto会忽略请求值** |
| weight_decay | 0.0005 | 0.0005 | 显式重复默认 |
| warmup_epochs | 3.0 | 3.0 | 显式重复默认 |
| warmup_momentum | 0.8 | 0.8 | 显式重复默认 |
| warmup_bias_lr | 0.1 | 0.1 | 默认请求值；**auto在trainer内改为0.0** |
| patience | 100 | 100 | 显式重复默认；early stopping启用，实际epoch不得假定恰100 |
| cos_lr | false | false | 显式重复默认（线性LR） |
| close_mosaic | 10 | 10 | 显式重复默认 |
| amp | true | true | 显式重复默认 |
| deterministic | true | true | 显式重复默认 |
| workers | 8 | **4** | **既有E01数据加载设置**；作为实际resolved config记录 |
| nbs | 64 | 未显式指定 | 使用包默认；影响auto所依据的预计迭代数 |

| Segmentation增强 | 8.3.220默认 | E01请求 | 来源 |
|---|---:|---:|---|
| hsv_h / hsv_s / hsv_v | .015 / .7 / .4 | 相同 | 显式重复默认 |
| degrees / translate / scale | 0 / .1 / .5 | 相同 | 显式重复默认 |
| shear / perspective | 0 / 0 | 相同 | 显式重复默认 |
| flipud / fliplr | 0 / .5 | 相同 | 显式重复默认 |
| mosaic / mixup / copy_paste | 1 / 0 / 0 | 相同 | 显式重复默认 |
| copy_paste_mode | flip | flip | 显式重复默认 |

**Optimizer A/B核验**：最初候选`SGD, lr0=.01`是有意显式指定、**不是8.3.220官方默认行为**。A方案`auto`保留官方标准机制；B方案SGD固定优化器和学习率，便于跨运行直接比较，但会改变包的标准optimizer选择。尚无论文问题或资源证据要求这种偏离，因此用户提出的中性基线原则下，E01正式采用**A：optimizer=auto**。`build_optimizer`在`auto`时依据`iterations > 10000`选择SGD(0.01, momentum .9)，否则选AdamW(`round(.002*5/(4+nc),6)`, momentum .9)；两者都忽略请求的`lr0=.01`与`momentum=.937`，并把`warmup_bias_lr`设为0。以当前1099池中769张train、batch8、nbs64、100 epochs静态推算`ceil(769/64)*100=1300`，预计AdamW、实际初始LR **.001429**。这仅是**trainer源码投影**；正式run必须记录`model.trainer.optimizer`类型、`optimizer.defaults['lr']`、`trainer.args.warmup_bias_lr`和实际`args.yaml`，不得把投影写成实验事实。

**Batch规则**：batch8已经冻结；Kaggle正式训练**之前**的短CUDA smoke若明确OOM，只允许`8→4`，在运行命令显式传`--batch 4 --oom-note "…"`，会写入预训练resolved配置和run manifest；不运行多个batch比较，更不按mAP选batch。若batch4仍失败，停止并重新设计/版本化预算，不静默改变配置。

`patience=100`与`epochs=100`意味着100是最高预算；训练中的Val fitness用于选择`best.pt`，`save_model`在当前fitness等于最佳fitness时保存。最终Test不能影响epoch或best选择。若因early stopping、异常或设备中断导致实际epoch少于100，按`results.csv`和checkpoint记录实际值；不能写成完成了100轮。

[解析工具](../IOR-YOLO/scripts/16_resolve_e01_config.py)输出四层：包默认、E01请求、训练前合并配置、auto投影。正式入口在训练**之前**保存Git忽略的`runs/e01_yolo11n_seg/seed_0_resolved_train_config.yaml`；训练后将观测到的optimizer/LR/warmup、实际epoch和`args.yaml`链接写入run内`resolved_train_config.yaml`，并记SHA。这样区别请求与执行，尤其不会把auto忽略的`.01`冒充实际学习率。
