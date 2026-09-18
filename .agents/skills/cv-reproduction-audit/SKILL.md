---
name: cv-reproduction-audit
description: "复现计算机视觉 GitHub 项目、运行开源 baseline code、核查 pretrained weights 或 reproduce paper results 时使用。当前项目级候选通用 Skill；不用于仅浏览仓库简介、纯论文精读或自行设计新模型。"
metadata:
  version: "1.0.0"
  origin: local-independent
---

# 计算机视觉复现审计

执行所需环节：Repository Audit → Environment Reconstruction → Dataset Identification → Config Identification → Baseline Execution → Result Comparison → Deviation Analysis → Reproducibility Report。

1. 仓库审计：核验论文和仓库对应、URL、commit hash、许可证、训练/评价入口、权重和配置来源。先读脚本再运行；不执行与复现无关的安装器或外部操作。
2. 环境重建：记录操作系统、Python、框架、驱动/加速库及 dependencies；优先隔离环境和锁定版本，不污染已有环境。
3. 确认 dataset version、划分、标签、预处理、评价协议和论文是否一致；缺失项明确记录。不同数据/协议的实验只能称迁移验证，不能称原结果复现。
4. 固定 config、random seed、hardware、checkpoint 来源及校验值、完整 command。先运行最小冒烟测试，再在已授权资源范围内执行基线。
5. 在运行前记录 expected result、来源表格/配置和可接受偏差依据；运行后记录 actual result、deviation（绝对及适用时相对差异）和日志。
6. 偏差按数据、模型/权重、训练、软件、硬件、指标实现及随机性调查；保留失败命令，不使用测试数据反向调参以凑论文数字。

复现报告必须包含 repository URL、commit hash、environment、dependencies、dataset version、config、random seed、hardware、checkpoint、command、expected result、actual result、deviation。缺失不猜测。

区分状态：静态审计完成、环境可用、代码可运行、评价协议对齐、结果在事先说明的误差范围内复现。明确“代码能够运行”不等于“论文已经成功复现”。说明只验证推理还是完整训练。

本 Skill 暂存项目级、注册状态 Candidate。真实使用后记录优缺点；满足注册表晋升规则后才考虑移为用户级 research-code-reproduction，不自动复制到全局。
