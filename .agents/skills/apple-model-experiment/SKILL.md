---
name: apple-model-experiment
description: "当前苹果项目的 CNN、ResNet、EfficientNet、MobileNet、YOLO、Vision Transformer、attention、feature fusion、迁移学习、架构修改、训练、baseline、ablation 和指标分析。用于具体模型与实验工作；不用于无关模型任务、纯文献全景或仅检查数据来源。"
metadata:
  version: "1.0.0"
  origin: local-independent
---

# 苹果模型与实验

每次修改形成可追溯链：Problem → Hypothesis → Architectural Change（或明确训练策略变化）→ Expected Mechanism → Controlled Experiment → Evidence → Conclusion。

不得为制造创新模型而无依据叠加 Attention、Transformer、BiFPN、CBAM、ECA、SE 或其他模块。每个新增模块回答：
1. 明确解决什么错误或问题？
2. 为什么应改善苹果阶段检测，机制依据及反例是什么？
3. 相对 baseline 精确改变什么？
4. 能否通过独立消融隔离作用？
5. 参数量、FLOPs、速度代价如何？
6. 提升是否超过随机波动，证据有多强？

先确认任务类型、标签和划分。分类报告 Accuracy、Precision、Recall、F1、Confusion Matrix，ROC/AUC 仅在适用时使用并注明平均方式。检测报告 mAP@0.5、mAP@0.5:0.95、Precision、Recall，效率相关主张报告 FPS/latency。记录参数量、FLOPs（注明计数工具/约定）、有意义时的训练时间和推理延迟；速度固定硬件、输入、batch、精度、预热及预处理/NMS范围。

成熟阶段有可靠顺序依据时再使用 MAE/QWK/SER；按类别无关框匹配，固定 IoU 和验证集确定的阈值，同时报告匹配覆盖率，不能忽略漏检只报告容易样本。

对候选 IOR 路线，先检查光照导致定位还是阶段错误；辅助序数头改善是否传递到最终输出；增强是否保持标签含义。使用普通增强、无一致性但相同双视图预算等对照，避免将计算增加当作机制收益。

具体方案设计调用 experiment-design，开源复现调用 cv-reproduction-audit，涉及标签/划分先用 apple-dataset-audit。记录配置、提交、种子、环境、权重与预测，区分试跑和正式结果；最终 test 禁止调参。结论说明效应、波动、失败模式与适用范围，不把单种子上涨称为稳定改善。
