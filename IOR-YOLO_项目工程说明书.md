# IOR-YOLO 项目工程说明书

> 课题：复杂果园环境下融合光照一致性与序数监督的域鲁棒苹果成熟度检测  
> 英文题目：*Domain-Robust Apple Maturity Detection via Illumination Consistency and Ordinal Supervision in Complex Orchard Environments*  
> 模型暂定名：**IOR-YOLO**  
> 文档版本：v1.0（2026-09-18）

---

## 0. 这份说明书怎么用

本说明书不是论文正文，而是整个课题的工程执行依据。实际工作严格按以下顺序推进：

1. 下载并审计数据；
2. 固定数据划分、类别顺序、软件版本和随机种子；
3. 完成完全不改模型的 YOLO11n 基线；
4. 先证明光照变化确实导致性能下降；
5. 再依次加入序数头、CPIP 和一致性约束；
6. 完成消融、对比、外部域和复杂度实验；
7. 最后统一生成论文表格和图片。

任何阶段未通过验收，不进入下一阶段。禁止同时修改多个模块后只汇报最终结果。

---

## 1. 项目目标与边界

### 1.1 最终要回答的研究问题

本项目需要用实验回答三个问题：

1. 显式利用苹果成熟阶段的顺序关系，能否减少跨越两个及以上阶段的严重误判？
2. 在尽量保留成熟度相关色度信息的情况下进行亮度扰动，能否提高模型对阴影、曝光和不均匀光照的鲁棒性？
3. 在主数据集之外的 Fuji 苹果数据上，所提方法能否比普通 YOLO11n 获得更好的零样本域外表现？

### 1.2 固定的任务定义

主任务为四阶段目标检测：

| 类别 ID | 规范类别名 | 含义 | 序数值 |
|---:|---|---|---:|
| 0 | `pre_growth` | Pre-growth | 0 |
| 1 | `young` | Young | 1 |
| 2 | `late_growth` | Late-growth | 2 |
| 3 | `ripe` | Ripe | 3 |

类别顺序一经写入数据配置，不得在训练、评价或外部映射中改变。

### 1.3 本项目不做什么

- 不以 CBAM、SE、ECA、BiFPN、GhostConv 或更换 IoU Loss 作为核心创新；
- 不同时扩展到分割、RGB-D、光谱、糖度预测或机械臂采摘；
- 不使用最终测试集选择超参数；
- 不把与主数据同源的 CDFRB 苹果子集作为独立外部域；
- 不用 Fuji 标签调参或微调模型；
- 不将视觉成熟指数 MI 宣称为 Brix、硬度或生理成熟度。

---

## 2. 推荐设备与软件环境

### 2.1 训练设备

主训练建议使用现有 **Windows 10 + NVIDIA RTX 3070** 电脑。Mac 可用于阅读、标注检查、结果整理和论文写作，不建议承担主要模型训练。

建议保留至少：

- 100 GB 可用磁盘空间；
- 16 GB 系统内存，推荐 32 GB；
- NVIDIA 驱动正常；
- Git、Anaconda/Miniconda。

### 2.2 环境建立原则

不要直接复用旧的 TensorFlow、PyMARL 或通信仿真环境。为本项目单独创建环境：

```bash
conda create -n ior_yolo python=3.11 -y
conda activate ior_yolo
```

PyTorch 按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)生成与当前驱动匹配的命令，不要凭记忆硬装 CUDA 版本。安装后检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

必须看到 `torch.cuda.is_available()` 为 `True`，才能开始正式训练。

### 2.3 获取并冻结 Ultralytics 代码

本研究需要修改检测头和训练损失，因此使用源码可编辑安装，不只安装黑盒 PyPI 包：

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
git checkout <冻结的正式版本标签或提交号>
pip install -e .
```

然后安装本项目额外依赖：

```bash
pip install opencv-python albumentations scikit-learn scipy pandas seaborn matplotlib tensorboard thop coral-pytorch
```

说明：2026 年官方仓库已经包含更新的模型系列，但本研究仍固定 **YOLO11n**，目的是与已有苹果成熟度论文形成直接可比的基线。不要在项目中途静默升级 Ultralytics。

环境确认后保存：

```bash
python -m pip freeze > requirements-lock.txt
git rev-parse HEAD > ultralytics-commit.txt
nvidia-smi > nvidia-smi.txt
```

相关官方入口：

- [Ultralytics 官方仓库](https://github.com/ultralytics/ultralytics)
- [YOLO11 官方说明](https://docs.ultralytics.com/models/yolo11/)
- [Ultralytics 训练文档](https://docs.ultralytics.com/modes/train/)
- [CORAL/CORN PyTorch 参考实现](https://github.com/Raschka-research-group/coral-pytorch)

> 许可提醒：Ultralytics 仓库采用其当前公布的开源许可。科研实验和论文复现通常可行；如果后续做商业交付，应重新核查当时的许可要求。

---

## 3. 推荐工程目录

```text
IOR-YOLO/
├── README.md
├── requirements-lock.txt
├── ultralytics-commit.txt
├── configs/
│   ├── data/
│   │   ├── orchard_apple.yaml
│   │   └── fuji_external.yaml
│   ├── experiments/
│   │   ├── E01_baseline.yaml
│   │   ├── E04_ordinal.yaml
│   │   ├── E05_cpip.yaml
│   │   └── E06_full.yaml
│   └── corruption/
│       └── illumination_benchmark.yaml
├── data/
│   ├── raw/
│   │   ├── orchard_apple/
│   │   └── fuji_ripeness/
│   ├── interim/
│   ├── processed/
│   │   ├── orchard_apple/
│   │   │   ├── images/{train,val,test}/
│   │   │   └── labels/{train,val,test}/
│   │   └── fuji_external/
│   ├── splits/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   └── test.txt
│   └── manifests/
│       ├── files_sha256.csv
│       ├── class_counts.csv
│       └── duplicate_report.csv
├── ior_yolo/
│   ├── augment/
│   │   ├── cpip.py
│   │   └── illumination_corruptions.py
│   ├── models/
│   │   ├── ordinal_head.py
│   │   └── ior_detect.py
│   ├── losses/
│   │   ├── ordinal_loss.py
│   │   └── consistency_loss.py
│   ├── trainers/
│   │   └── ior_trainer.py
│   ├── evaluators/
│   │   ├── detection_metrics.py
│   │   ├── ordinal_metrics.py
│   │   ├── robustness_metrics.py
│   │   └── external_fuji.py
│   └── utils/
│       ├── seed.py
│       ├── io.py
│       └── run_manifest.py
├── scripts/
│   ├── 00_check_environment.py
│   ├── 01_inspect_dataset.py
│   ├── 02_make_splits.py
│   ├── 03_check_leakage.py
│   ├── 04_train_baseline.py
│   ├── 05_build_corruption_benchmark.py
│   ├── 06_train_ordinal.py
│   ├── 07_train_cpip.py
│   ├── 08_train_full.py
│   ├── 09_eval_internal.py
│   ├── 10_eval_corruptions.py
│   ├── 11_eval_fuji.py
│   ├── 12_benchmark_speed.py
│   └── 13_make_paper_figures.py
├── tests/
│   ├── test_class_order.py
│   ├── test_cpip_geometry.py
│   ├── test_ordinal_monotonicity.py
│   ├── test_loss_finite.py
│   └── test_fuji_mapping.py
├── runs/
│   ├── E00/
│   ├── E01/
│   └── ...
├── reports/
│   ├── tables/
│   ├── figures/
│   ├── predictions/
│   └── failure_cases/
└── third_party/
    └── NOTICE.md
```

`data/`、`runs/`、模型权重和大体积预测文件不要提交到 Git。代码、配置、数据划分清单、环境锁定文件和最终结果表需要提交。

---

## 4. 数据下载与数据协议

### 4.1 主数据：Orchard Apple Maturity Dataset

下载入口：

- DOI：[10.6084/m9.figshare.29533841.v1](https://doi.org/10.6084/m9.figshare.29533841.v1)

下载后先不要训练，完成以下审计：

- 统计图片数、标注框数、每类实例数；
- 检查空标注、越界框、零面积框和损坏图片；
- 检查原始标注究竟是 YOLO、VOC 还是 COCO 格式；
- 检查图像文件名和内容哈希是否重复；
- 人工抽查每类至少 50 个框，确认类别语义和类别 ID；
- 保存数据集原始许可、引用信息和下载日期。

推荐划分：

- 原作者的 421 张 validation：冻结为最终内部 `test`；
- 原作者的 1618 张 train：划分为本项目 `train` 和 `val`；
- 建议按约 8:2 划分，并尽量保持类别实例分布；
- 固定 `seed=42`，将确切文件名保存到 `data/splits/*.txt`；
- 若同一场景存在连续帧或近重复图像，必须按场景分组后再划分，不能让近重复帧跨集合。

数据配置示例：

```yaml
path: data/processed/orchard_apple
train: images/train
val: images/val
test: images/test
names:
  0: pre_growth
  1: young
  2: late_growth
  3: ripe
```

### 4.2 外部域：Fuji-Ripeness-Size Dataset

下载与代码入口：

- [Fuji Ripeness and Size Estimation](https://github.com/zhukeyi-stan/Fuji_Ripeness_And_Size_Estimation)

使用规则：

- 只使用 RGB 数据完成本研究的域外检测与成熟/未成熟评价；
- 保留其原始测试划分；若仓库没有明确测试划分，先根据论文/README 固定一个一次性划分并记录；
- 不用 Fuji 数据训练、微调、早停、选择置信度阈值或选择模型；
- 四阶段预测统一映射为：`ripe → Ripe`，其余三个阶段 → `Unripe`；
- 所有阈值只能在 Orchard 内部验证集上决定；
- 转换标注后抽样人工核验，记录转换脚本版本。

### 4.3 仅作为辅助或参考的数据

- [CDFRB](https://github.com/zxr0826/CDFRB)：苹果部分可能与主数据同源，不作为独立外部域；
- [MinneApple](https://github.com/nicolaihaeni/MinneApple)：可用于定位泛化补充实验，但没有四级成熟度标签；
- [AppleGrowthVision](https://github.com/fraunhoferhhi/AppleGrowthVision)：可用于物候或定位补充分析，不直接替代四级成熟度任务。

### 4.4 数据泄漏检查

至少执行三种检查：

1. 文件级 SHA-256 完全重复；
2. 感知哈希寻找缩放、压缩或轻微裁剪后的近重复图；
3. 文件名、拍摄序列和元数据检查连续帧。

最终在论文中报告：划分原则、随机种子、各集合图像数、各类实例数，以及去重/近重复处理规则。

---

## 5. 基线训练规范

### 5.1 固定训练配置

第一轮建议配置：

| 参数 | 建议值 | 说明 |
|---|---:|---|
| model | `yolo11n.pt` | COCO 预训练 |
| imgsz | 640 | 所有主实验一致 |
| epochs | 200 | 配合早停 |
| patience | 50 | 只看内部 val |
| batch | 16 | RTX 3070 起始值，显存不足改 8 |
| device | 0 | 单 GPU |
| workers | 4 | Windows 稳定优先 |
| optimizer | `auto` 或固定后记录 | 一旦基线冻结不得变 |
| amp | true | 所有同类实验一致 |
| seed | 0, 1, 2 | 核心结果至少 3 次独立运行 |
| deterministic | true | 尽量保证可重复 |
| cache | false | 避免内存不足 |

训练命令示例：

```bash
yolo detect train model=yolo11n.pt data=configs/data/orchard_apple.yaml imgsz=640 epochs=200 patience=50 batch=16 device=0 workers=4 seed=0 deterministic=True project=runs name=E01_baseline_s0
```

这只是模板。正式训练前先执行 2 个 epoch 的冒烟测试，确认数据、Loss、保存路径和验证流程全部正常。

### 5.2 每次实验必须保存的内容

每个 `runs/E*/` 目录至少包含：

- 完整配置 YAML；
- 随机种子；
- Git commit；
- Python、PyTorch、CUDA、Ultralytics 和 GPU 信息；
- `best.pt`、`last.pt`；
- 每 epoch 的 loss 和指标 CSV；
- Precision、Recall、mAP@0.5、mAP@0.5:0.95；
- 四分类混淆矩阵；
- 测试预测明细；
- 运行时间、峰值显存和异常日志。

只复制一张结果图而不保留配置和权重，视为实验无效。

---

## 6. 方法实现说明

### 6.1 总体数据流

```text
原始图像 x ───────────────┐
                           ├─ 共享 YOLO11n ─ 检测分支 ─ 框与普通分类
CPIP 图像 T_L(x) ─────────┘              └─ 序数分支 ─ 阶段概率与 MI
                                               │
                                  原图/增强图的一致性约束
```

CPIP 只改变光照，不执行几何变换，因此原图和增强图共用相同的检测标注。

### 6.2 序数头：工程冻结方案

对每个由 YOLO 目标分配器匹配到真实苹果的**正样本位置**，增加三个序数输出。负样本不计算成熟度序数损失。

设阶段 (y\in\{0,1,2,3\})，定义阈值标签：

\[
t_k=\mathbb{1}(y>k),\qquad k=0,1,2.
\]

为避免出现 (P(y>1)>P(y>0)) 这种违反顺序的结果，工程实现采用单调累计概率。令三个条件概率为：

\[
r_k=\sigma(z_k),
\]

累计序数概率为：

\[
p_0=r_0,\qquad p_1=r_0r_1,\qquad p_2=r_0r_1r_2.
\]

因此天然满足：

\[
p_0\ge p_1\ge p_2.
\]

四阶段分布可由累计概率恢复：

\[
q_0=1-p_0,\quad q_1=p_0-p_1,\quad q_2=p_1-p_2,\quad q_3=p_2.
\]

序数损失采用 CORN 风格的条件二元交叉熵；代码实现和论文公式必须保持一致。第一版中，序数头作为辅助监督，普通 YOLO 分类分支仍负责主检测类别。另做一个“序数分布解码”消融，比较：

- `Cls decode`：使用普通分类分支输出阶段；
- `Ordinal decode`：使用 \(\arg\max q_c\) 输出阶段；
- 可选融合只作为额外消融，不默认写入主方法。

连续视觉成熟指数：

\[
MI=\frac{p_0+p_1+p_2}{3}\in[0,1].
\]

必须编写单元测试，随机输入下始终满足累计概率单调、四类概率非负且和为 1。

### 6.3 CPIP：色度保持的光照扰动

流程：

1. 将 RGB 图像转换到 Lab；
2. 仅对亮度通道 (L) 进行随机变换；
3. 尽量保持 (a,b) 色度通道不变；
4. 转回 RGB，并记录色域裁剪比例。

第一版支持以下操作：

| 操作 | 训练随机范围（起始建议） | 说明 |
|---|---|---|
| brightness | 0.7–1.3 | 全局亮暗 |
| gamma | 0.7–1.5 | 非线性曝光 |
| local shadow | 强度 0.15–0.45 | 平滑局部阴影掩膜 |
| local exposure | 强度 0.10–0.35 | 局部强光 |
| contrast | 0.75–1.25 | 亮度对比度 |
| uneven illumination | 随机平滑场 | 模拟冠层不均匀光照 |

约束：

- 不改变 Hue/Saturation 或直接把青苹果变红；
- 不改变图像尺寸和目标框；
- 每张图随机使用 0–2 种光照变化；
- 增强参数必须保存到日志；
- 可视化抽查每类至少 100 个目标；
- 若出现大量过曝纯白、全黑或明显色偏，立即缩小参数范围。

训练增强和测试 corruption 使用不同的配置文件。测试集变换固定随机种子并一次生成，后续所有模型使用同一批测试图。

### 6.4 光照一致性约束

由于 CPIP 不改变几何位置，原图和增强图使用同一真实框。对同一真实目标对应的正样本序数分布 (q(x)) 与 (q(T_L(x))) 计算 Jensen–Shannon 散度：

\[
L_{con}=JS\big(q(x)\parallel q(T_L(x))\big).
\]

不要直接对两幅图中未经匹配的全部检测框按数组索引计算一致性。推荐做法是：

1. 使用真实框和干净分支的目标分配结果确定正样本；
2. 在相同尺度和空间位置提取两分支的序数输出；
3. 先对属于同一真实目标的正样本概率求加权平均；
4. 再计算目标级 JS 散度。

总损失：

\[
L_{total}=L_{box}+\lambda_{dfl}L_{dfl}+\lambda_{cls}L_{cls}
+\lambda_{ord}L_{ord}+\lambda_{con}L_{con}.
\]

YOLO 原有三项权重先保持官方默认值。只在内部验证集上搜索：

- \(\lambda_{ord}\in\{0.25,0.5,1.0\}\)；
- \(\lambda_{con}\in\{0.1,0.25,0.5\}\)。

为控制计算量，先用单个种子和较短训练筛选，再对最终配置运行 3 个种子。筛选规则：清洁集 mAP@0.5:0.95 相对基线下降不超过 0.5 个百分点，在满足该条件的模型中优先选择验证集 SER 更低者，再以 MAE 作为次级判据。

---

## 7. 光照鲁棒性基准

### 7.1 基准组成

冻结的 Orchard `test` 生成以下 18 个版本：

| Corruption | Level 1 | Level 2 | Level 3 |
|---|---|---|---|
| low brightness | 轻 | 中 | 重 |
| over exposure | 轻 | 中 | 重 |
| gamma | 轻 | 中 | 重 |
| local shadow | 轻 | 中 | 重 |
| uneven illumination | 轻 | 中 | 重 |
| contrast reduction | 轻 | 中 | 重 |

具体参数写入 `configs/corruption/illumination_benchmark.yaml`，一旦 E03 完成即冻结。所有模型必须使用完全相同的 corruption 图像与标注。

### 7.2 鲁棒性指标

除每种条件的 mAP 外，至少报告：

\[
\Delta mAP=mAP_{clean}-mAP_{corrupted},
\]

\[
R=\frac{mAP_{corrupted}}{mAP_{clean}},
\]

以及 18 个条件的平均 corrupted mAP。结果同时按 corruption 类型和 severity 分组，不能只汇报一个最好结果。

---

## 8. 评价协议

### 8.1 常规检测指标

- Precision；
- Recall；
- F1；
- mAP@0.5；
- mAP@0.5:0.95；
- 每类 AP；
- 混淆矩阵。

### 8.2 序数指标的匹配规则

Stage MAE、QWK 和 SER 不能直接对两组未对齐框计算。统一使用以下协议：

1. 先忽略成熟类别，按目标框做一对一匹配；
2. IoU 阈值固定为 0.5；
3. 置信度阈值在内部验证集上一次确定；
4. 同一真实框最多匹配一个预测框；
5. 在匹配成功的实例上计算 Stage MAE、QWK 和 SER；
6. 同时报告 `matched GT / all GT`，避免只在容易检测的样本上计算而造成虚高。

指标定义：

\[
MAE=\frac{1}{N_m}\sum_{i=1}^{N_m}|\hat y_i-y_i|,
\]

\[
SER=\frac{1}{N_m}\sum_{i=1}^{N_m}\mathbb{1}(|\hat y_i-y_i|\ge2).
\]

QWK 使用 0–3 的固定类别顺序。论文表格同时给出均值和 3 次运行的标准差。

### 8.3 外部 Fuji 评价

将四级预测映射为二级后报告：

- binary AP；
- Precision、Recall、F1；
- binary confusion matrix；
- class-agnostic detection AP；
- Ripe 与 Unripe 的分别结果；
- 典型成功与失败案例。

严禁根据 Fuji 结果返回去改阈值、调增强范围或选择权重。若确需设计新方法，必须把 Fuji 视为已经“看过”，另找新的最终外部测试数据。

### 8.4 统计报告

核心对比至少运行 3 个随机种子：

- YOLO11n baseline；
- Ordinal only；
- CPIP only；
- 完整 IOR-YOLO。

报告 `mean ± std`。若声称显著提升，优先对逐图指标或配对预测做 bootstrap 95% 置信区间，而不是只比较单次训练的小数点。

---

## 9. 实验编号与执行清单

| ID | 实验 | 主要任务 | 必须产出 | 通过条件 |
|---|---|---|---|---|
| E00 | 环境检查 | GPU、依赖、代码版本、2 epoch 冒烟测试 | 环境快照、日志 | CUDA 可用，训练/验证/保存无报错 |
| E01 | YOLO11n 基线 | 3 seeds 正式训练 | 权重、曲线、检测指标 | 结果稳定，无数据泄漏 |
| E02 | 数据质量审计 | 类别、框、重复和近重复检查 | manifest、审计报告 | 类别 ID 正确，坏标注已处理 |
| E03 | 光照问题验证 | 构建并测试 18 个 corruption 条件 | 鲁棒性表和曲线 | 至少部分条件有明确性能下降 |
| E04 | Ordinal only | 增加序数头与损失 | MAE/QWK/SER、单调性测试 | mAP 基本保持，至少一个序数指标改善 |
| E05 | CPIP only | 只加入 CPIP | clean/corrupted 指标 | 平均鲁棒性优于 E01 |
| E06 | Full model | Ordinal + CPIP + consistency | 完整指标和权重 | 达到预设选择规则 |
| E07 | 消融实验 | 逐模块、解码方式、权重敏感性 | 消融表 | 每个结论有对应证据 |
| E08 | 主流模型对比 | YOLOv8n、YOLO11n、RT-DETR 等 | 对比表 | 同数据、同划分、同评价 |
| E09 | Fuji 外部测试 | 完全零微调外部评价 | 二分类结果、可视化 | 全流程无 Fuji 调参 |
| E10 | 复杂度与论文图 | Params、GFLOPs、FPS、8 幅核心图 | 最终表图包 | 同硬件、同输入、同测试脚本 |

### 9.1 推荐的最小消融矩阵

| 模型 | CPIP | Ordinal | Consistency | 目的 |
|---|:---:|:---:|:---:|---|
| Baseline | × | × | × | 参照 |
| A | ✓ | × | × | CPIP 单独贡献 |
| B | × | ✓ | × | 序数监督单独贡献 |
| C | ✓ | ✓ | × | 两模块组合 |
| Proposed | ✓ | ✓ | ✓ | 完整方法 |

额外小型消融：

- 普通分类解码 vs 序数解码；
- 普通 ColorJitter vs CPIP；
- 无一致性 vs KL vs JS（只在确有必要时做）；
- 不同 \(\lambda_{ord}\)、\(\lambda_{con}\) 的敏感性。

### 9.2 公平对比规则

- 相同 train/val/test 文件清单；
- 相同分辨率和训练 epoch 上限；
- 相同预训练策略；
- 不给 Proposed 额外使用测试信息；
- FPS 和 latency 在同一台 RTX 3070、同一 batch、同一精度下测量；
- 首先预热，再统计至少 500 次推理；
- 明确 latency 是否包含预处理和 NMS。

AugMix 与 MixStyle 可作为鲁棒学习参考实现：

- [AugMix](https://github.com/google-research/augmix)
- [MixStyle](https://github.com/KaiyangZhou/mixstyle-release)

它们不能直接照搬分类代码并声称已完成检测对比，需要明确插入位置和适配方式。

---

## 10. 每阶段具体待办

### 阶段 A：下载与准备

- [ ] 建立独立 Conda 环境；
- [ ] CUDA 检查通过；
- [ ] 克隆并冻结 Ultralytics commit；
- [ ] 下载 Orchard 数据集；
- [ ] 下载 Fuji 数据和官方代码；
- [ ] 保存数据许可与引用信息；
- [ ] 完成图片、框和类别统计；
- [ ] 完成重复和近重复检查；
- [ ] 固定 `train/val/test` 清单；
- [ ] 将标注统一转换为 YOLO 格式；
- [ ] 人工可视化抽查。

### 阶段 B：基线

- [ ] 2 epoch 冒烟测试；
- [ ] 运行 E01 seed 0/1/2；
- [ ] 保存所有配置、权重和预测；
- [ ] 生成 clean test 主结果；
- [ ] 检查是否接近合理的公开基线；
- [ ] 若差距很大，先排查数据划分、类别映射和训练配置。

### 阶段 C：证明问题存在

- [ ] 实现 6 类 × 3 级光照 corruption；
- [ ] 固定生成参数与随机种子；
- [ ] 用 E01 权重测试全部条件；
- [ ] 生成 mAP–severity 曲线；
- [ ] 确认论文要解决的光照退化真实可观测。

### 阶段 D：方法开发

- [ ] 序数头只作用于正样本；
- [ ] 完成累计概率单调性测试；
- [ ] 完成 q 分布非负与归一化测试；
- [ ] 跑 E04 并先看 MAE/SER；
- [ ] 实现 CPIP，确保框不变；
- [ ] 做 CPIP 色偏和过曝人工检查；
- [ ] 跑 E05；
- [ ] 实现目标级一致性匹配；
- [ ] 跑 E06；
- [ ] 检查所有 loss 是否有限、是否出现塌缩。

### 阶段 E：论文证据链

- [ ] 完成消融矩阵；
- [ ] 完成主流检测器对比；
- [ ] 完成三个种子的均值与标准差；
- [ ] 完成 Fuji 零微调测试；
- [ ] 完成参数量、GFLOPs、FPS 和 latency；
- [ ] 整理成功与失败案例；
- [ ] 生成论文主表和 8 幅图；
- [ ] 检查所有论文数字是否能追溯到具体 run。

---

## 11. 论文结果文件映射

| 论文内容 | 工程来源 |
|---|---|
| 主检测结果表 | E01、E06、E08 |
| 消融表 | E04–E07 |
| 序数 MAE/QWK/SER | `reports/tables/ordinal_metrics.csv` |
| 光照鲁棒性曲线 | E03、E05、E06 |
| Fuji 外部结果 | E09 |
| 复杂度表 | E10 |
| Figure 1 阶段顺序示意 | 手工矢量绘制 |
| Figure 2 网络结构 | 根据最终代码绘制 |
| Figure 3 CPIP 样例 | E05 固定样例 |
| Figure 4 混淆矩阵 | E06 |
| Figure 5 错误距离分布 | E01 vs E06 |
| Figure 6 severity 曲线 | E03/E06 |
| Figure 7 Fuji 可视化 | E09 |
| Figure 8 成功与失败案例 | E06/E09 |

每张图和每个表都应有生成脚本，避免手工复制数字造成错误。

---

## 12. 成功判据与停止规则

### 12.1 预期成功判据

这些是项目管理目标，不是可以预先写进论文的结果：

- Clean mAP@0.5:0.95 不低于基线，或下降不超过 0.5 个百分点；
- Stage MAE 相对下降约 10% 或以上；
- SER 相对下降约 20% 或以上；
- corrupted mean mAP 和相对鲁棒性优于 YOLO11n；
- Fuji Ripe/Unripe F1 稳定优于 YOLO11n；
- 增加的推理延迟和参数量可接受。

### 12.2 Go/No-Go 决策

- 如果 E03 显示光照变化几乎不影响性能：重新检查 corruption 是否合理，仍无差异则削弱“光照鲁棒性”主张；
- 如果 E04 的 mAP 正常但 MAE/SER 不改善：先检查正样本映射、类别顺序和序数解码，不继续叠加模块；
- 如果 E05 只提高 corrupted 结果却明显损害 clean：缩小 CPIP 强度；
- 如果 E06 一致性 loss 导致置信度塌缩：降低 \(\lambda_{con}\)，检查目标匹配，不立即更换 backbone；
- 如果内部提升但 Fuji 无提升：如实报告域差异和失败模式，不使用 Fuji 反向调参；
- 如果只有单次随机种子提升：不得声称稳定有效。

---

## 13. 风险清单

| 风险 | 典型表现 | 应对方式 |
|---|---|---|
| 数据同源 | 外部集与主数据重复 | 哈希、近重复和来源核查 |
| 类别顺序错误 | MAE/SER 异常 | 固定映射并写单元测试 |
| 序数概率不单调 | 阶段概率为负 | 累计条件概率实现 |
| 一致性对象未对齐 | loss 波动或无意义 | 按真实目标聚合正样本 |
| CPIP 改变成熟语义 | 颜色明显改变 | 只改 L，人工与数值检查 |
| 测试集参与调参 | 结果虚高 | 只用内部 val 选模型 |
| Ultralytics 更新破坏代码 | 同命令结果不同 | 冻结 commit 与环境 |
| 只追求 mAP | 创新证据不足 | 强制报告 MAE/QWK/SER/OOD |
| 小数据高方差 | 单次提升不稳定 | 3 seeds + bootstrap CI |

---

## 14. 推荐进度（8 周）

| 周次 | 任务 | 里程碑 |
|---:|---|---|
| 1 | 环境、下载、数据审计 | E00/E02 完成 |
| 2 | 数据划分、基线冒烟和正式训练 | E01 完成 |
| 3 | 光照 corruption benchmark | E03 完成 |
| 4 | 序数头与评价代码 | E04 完成 |
| 5 | CPIP 与鲁棒训练 | E05 完成 |
| 6 | 一致性学习与完整模型 | E06 完成 |
| 7 | 消融、对比、Fuji 外部测试 | E07–E09 完成 |
| 8 | 复杂度、图表、失败案例和论文方法稿 | E10 完成 |

如果每周可投入时间较少，可将每一周扩展为两周，但不要打乱先后依赖关系。

---

## 15. 第一次开工只做这 7 件事

1. 在 RTX 3070 Windows 电脑创建 `ior_yolo` Conda 环境；
2. 确认 PyTorch 能识别 GPU；
3. 克隆 Ultralytics 并记录 commit；
4. 下载 Orchard Apple Maturity Dataset；
5. 下载 Fuji 仓库与数据说明，但暂不用于训练；
6. 建立上述目录，运行数据审计；
7. 只跑 YOLO11n 的 2 epoch 冒烟测试。

在这 7 项完成前，不修改网络结构，也不编写论文结果章节。

---

## 16. 官方资源清单

| 用途 | 资源 |
|---|---|
| 主数据集 | [Orchard Apple Maturity Dataset DOI](https://doi.org/10.6084/m9.figshare.29533841.v1) |
| 主检测框架 | [Ultralytics](https://github.com/ultralytics/ultralytics) |
| YOLO11 文档 | [YOLO11](https://docs.ultralytics.com/models/yolo11/) |
| 训练文档 | [Ultralytics Train Mode](https://docs.ultralytics.com/modes/train/) |
| 序数学习参考 | [coral-pytorch](https://github.com/Raschka-research-group/coral-pytorch) |
| 鲁棒增强参考 | [AugMix](https://github.com/google-research/augmix) |
| 域泛化参考 | [MixStyle](https://github.com/KaiyangZhou/mixstyle-release) |
| 外部域数据 | [Fuji Ripeness and Size Estimation](https://github.com/zhukeyi-stan/Fuji_Ripeness_And_Size_Estimation) |
| 定位辅助数据 | [MinneApple](https://github.com/nicolaihaeni/MinneApple) |
| 生长阶段辅助数据 | [AppleGrowthVision](https://github.com/fraunhoferhhi/AppleGrowthVision) |
| 综合成熟度数据参考 | [CDFRB](https://github.com/zxr0826/CDFRB) |

---

## 17. 最终验收清单

课题达到“可以开始写完整 SCI 论文”的最低条件：

- [ ] 数据来源、许可、划分和类别统计完整；
- [ ] 主数据无明显泄漏；
- [ ] Baseline 可重复；
- [ ] 序数头概率单调且只监督正样本；
- [ ] CPIP 不改变框和主要色度语义；
- [ ] 一致性约束完成目标级对齐；
- [ ] clean、corruption、external 三层评价全部完成；
- [ ] 主结果至少 3 个随机种子；
- [ ] MAE、QWK、SER 有严格匹配协议；
- [ ] 所有对比使用同一数据划分；
- [ ] 复杂度在同一硬件和设置下测量；
- [ ] 每个论文数字可回溯到配置、权重和预测文件；
- [ ] 失败案例和研究限制已整理；
- [ ] 论文图表由脚本生成并复核。

完成以上项目后，论文的证据链才真正闭合：

> **标准检测性能 + 成熟阶段顺序合理性 + 人工光照退化鲁棒性 + 真实外部域泛化 + 计算代价。**

