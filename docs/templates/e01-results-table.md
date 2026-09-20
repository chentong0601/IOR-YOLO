# E01 SCI result table template

状态：**尚无E01正式结果**。NA表示未运行/未核实，非零分或模型比较结论。Box与Mask分开报告，分母、split及best checkpoint在填值前另附真实run ID、commit、hash与评价协议。

| Model | Task | Params | Input | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n-seg E01 | Instance segmentation | NA | 640 | NA | NA | NA | NA | NA | NA | NA | NA |

| Split | Class | Box AP50 | Box AP50-95 | Mask AP50 | Mask AP50-95 | Matched | Missed | False positive |
|---|---|---|---|---|---|---|---|---|
| NA | immature | NA | NA | NA | NA | NA | NA | NA |
| NA | semi-mature | NA | NA | NA | NA | NA | NA | NA |
| NA | mature | NA | NA | NA | NA | NA | NA | NA |

论文使用正式Val与最终Test时须各有独立表/列，且不得以最后epoch的`results.csv`数值代替best checkpoint独立评价。
