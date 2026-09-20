# E01 结果分析协议（运行前准备）

状态：**Analysis pipeline READY；没有正式训练、Val、Test或性能数字**。只有Windows RTX 3070的正式E01结束后，才用真实输出执行下述命令。脚本和表格现在仅经合成数据单元测试。E01仍是普通instance segmentation基线，不包含ordinal训练目标。

## 产物与来源

正式运行目录为Git忽略的`IOR-YOLO/runs/e01_yolo11n_seg/seed_0/`。原生`weights/{best,last}.pt`、`args.yaml`、`results.csv`及`validation/`由固定Ultralytics产生；runner记录`run_manifest.yaml`、实际观测`resolved_train_config.yaml`及`metrics/`、`figures/`。预测导出写`predictions/predictions_val.csv`和逐实例归一化polygon JSON，字段为`image_id,pred_instance_id,pred_class,confidence,box,mask_reference,source_group_id,split_guard_cluster_id,split`。匹配后写`analysis/matched_predictions_val.csv`、`analysis/analysis_val.json`与`analysis/failure_cases/`。Test仅在一次正式最终评价完成后用相同结构、`test`后缀输出。已有预测CSV/分析JSON拒绝覆盖；原生输出不得因结果不理想而替换。

未来用户在工作站先运行正式`train`，再运行`val`；随后执行：

```powershell
python IOR-YOLO/scripts/18_export_e01_predictions.py --split val
python IOR-YOLO/scripts/17_analyze_e01_results.py --split val
```

**Test lock**：训练和Val期间禁止正式Test。唯一最终Test必须由用户在模型与分析选择结束后调用`15_e01_run.py test --final-test`，命令打印`FINAL TEST EVALUATION`，此前须有Val结果。最终Test完成且manifest标记`final_test_completed`后，才能对其预测导出和只读分析分别追加`--split test --final-test`。不得以Test结果调整阈值、epoch、方法或重新选择模型。脚本的Test入口缺少显式flag均拒绝。

## 指标、匹配和分母

从真实`results.csv`解析每epoch box与mask的Precision、Recall、mAP50、mAP50–95曲线；正式独立Val的完整汇总以及每类AP从run manifest读取。`results.csv`**最后epoch不是best checkpoint的最终Val分数**，不得混用。论文表格使用明确的独立Val/Test字段和best checkpoint对应结果；未运行记NA。类别顺序为immature=0、semi-mature=1、mature=2。

预测导出使用固定`conf=0.001`和`iou=0.7`，不依据Test选择阈值。分析从冻结派生标签读取逐实例GT，按每张图**不区分类别**的box IoU≥0.5，以IoU降序、置信度及稳定ID处理并列，贪心一对一匹配；同一GT的重复预测只能一条匹配，其他为FP。配对后计算逐mask raster IoU，输出GT/预测类及box/mask IoU。这个匹配与Ultralytics官方mAP evaluator并非同一算法；独立误差分析数字不可冒称官方mAP。未配对GT是missed detection，未配对预测是false positive，**均不进入3×3类别混淆矩阵**。

对已匹配实例计算`MASE_stage = mean(|pred_stage−gt_stage|)`、相邻错误比例`count(distance=1)/matched`、严重跨级比例`count(distance=2)/matched`、off-by-one准确率`count(distance≤1)/matched`。同时报告matched覆盖率、missed和FP数量；没有配对时阶段比例记null，不填零。这些只分析E01错误，不是ordinal方法或监督。

F1 Missed Fruit、F2 False Positive、F3 Adjacent Maturity Confusion、F4 Severe Maturity Confusion、F5 Poor Mask Localization（已匹配mask IoU<0.5）可以按上述规则自动标记；一例多类错误时以F3/F4优先，最低mask IoU列表仍单列。F6 Occlusion、F7 Illumination/Shadow、F8 Small Fruit、F9 Dense Overlap、F10 Ambiguous Visual Maturity仅作**Needs Human Review**的人工标签选项，程序不能自动归因。导出top FP、missed、最低mask IoU、高置信错误、0↔2及相邻错误的原始派生图和JSON索引，不修改raw或annotation。

图表只从真实`results.csv`、正式metric record、匹配文件生成loss与Val指标曲线、3×3混淆矩阵、每类box/mask AP和阶段错误分布；正式Val/Test产生的实际PR曲线再复制到`figures/`。不生成模拟曲线、无结果数字或伪造论文图。

## 可追溯性及约束

每次分析核对Git commit和clean标志、run ID、训练seed、Ultralytics8.3.220、D2 ZIP和冻结pool/split/protocol SHA、E01 config、best checkpoint及实际resolved config SHA；报告另记录run manifest、真实训练CSV、预测CSV与分析脚本自身SHA。任何缺失/不一致均发出显式`E01 provenance`警告并在报告中保留`incomplete`，不可把该分析当已核验论文结论。Val/Test完整评估同原生mAP结果与逐实例分析分开保存；Mac合成测试只证明代码逻辑，不证明RTX 3070端到端运行。未来尤其核验U01未知polygon区域按普通背景FP计分的真实框架语义。
