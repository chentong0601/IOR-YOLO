---
name: apple-dataset-audit
description: "当前苹果项目涉及图像 dataset、maturity labels、采集/标注、train/val/test、augmentation、类别不平衡、data leakage、domain shift、拍摄条件时使用。不用于无数据内容的架构解释或其他领域数据审计。"
metadata:
  version: "1.0.0"
  origin: local-independent
---

# 苹果数据审计

从来源页、论文、元数据、图像和标注取得证据。未提供的信息标记未知并说明影响；不能从文件名或少量样例推定全部采集条件。

逐项检查并记录证据位置：
1. 数据来源、版本、许可、引用和文件哈希。
2. Apple cultivar / variety。
3. acquisition device。
4. image resolution。
5. lighting condition。
6. background。
7. shooting distance。
8. viewpoint。
9. maturity category definition、类别 ID 和阶段顺序。
10. label ground truth 形成方法、标注人员/协议与一致性；区分视觉阶段、物候、生理成熟度。
11. class distribution：图像数、实例数、各 split 分布与小样本类。
12. train/validation/test split 的来源、比例、随机种子及分组单位。
13. 同一果实跨 split、连续帧、同一拍摄序列或近重复图像泄漏。
14. 学习型预处理及训练 augmentation 是否仅基于 training set；验证/测试的固定预处理独立记录。
15. external validation 的来源独立性、标签兼容、阈值来源与可获得性。
16. 数据能支持何种泛化 claim，哪些范围不能支持。

明确区分 image-level random split 和 fruit-level / orchard-level / session-level split。同一颗苹果的不同照片跨 train/test 必须标记潜在泄漏；结合哈希、感知近重复、元数据和人工复核。无 fruit ID 或采集分组证据时，不能声称已排除果实级泄漏。

增强在划分后进行，防止原图及增强副本跨集合。固定光照 corruption 测试是单独预定义评价，不等于训练增强泄漏；研发期只用验证集构建和调节协议，最终测试的扰动方案先冻结。

核查框越界、零面积、损坏图、空标注及类别映射。Fuji 二分类与四阶段的对应需核查定义，不能只按名称映射；同源子集不得充当独立外部域。

输出数据审计表（检查项、证据、发现、风险、行动）、统计/划分清单和主张边界。发现问题先保留原数据并提出可追溯修正，不静默删除样本。
