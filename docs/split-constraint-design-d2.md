# D2 Stage 2B-5P — Split Constraint Design（候选，不冻结）

历史候选记录更新：2026-09-20。基于SHA256为`049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`的D2 v4 ZIP、Stage 2B-4H人工裁决后的[1112组清单](../IOR-YOLO/data/manifests/source_groups.csv)和[关系边](../IOR-YOLO/data/manifests/source_group_relations.csv)。本页记载2B-5P当时的约束设计与**单次模拟**；“NOT FROZEN”仅为当时状态，用户随后已确认并完成[正式Stage 2B-5冻结](dataset-freeze-d2.md)，其逐图归属、比例和seed以正式协议为准。

## 两层组概念

| 概念 | 含义 | 可以声称什么 |
|---|---|---|
| `source_group_id` | 由Confirmed/Strongly Supported/Supported原图关系得到的证据支持候选同源组 | 1112个可复算的工作组；**不等于**独立果实、采集事件或物理原图数 |
| `split_guard_cluster_id` | 对source groups之间仍为Candidate的边做传递闭包；只用于同split约束 | 相关组不得跨未来split；**不证明**Candidate已同源，guard cluster数也不是独立采集数 |

生成器[10_prepare_d2_freeze_candidate.py](../IOR-YOLO/scripts/10_prepare_d2_freeze_candidate.py)在1406个非resize表示上，从现有关系清单找到**29条跨source-group Candidate原图边**：23条dHash残余候选，以及6条同名/增强家族候选（2条`172`跨官方split、4条当前处于同一官方split）。把这些边两端的source groups作为不可分配约束并做确定性传递闭包；**source_group_id完全不改**。输出[split_guard_clusters.csv](../IOR-YOLO/data/manifests/split_guard_clusters.csv)为1406行，显式写入文件、官方split、source group、guard cluster、关系原因与候选置信度。

重建结果：**1112 source groups → 1096 guard clusters**；16个cluster含跨组Candidate边，最大cluster为**2个source groups、6张非resize表示**。23条dHash及额外6条命名Candidate均在同一个guard cluster内，未覆盖的已列跨组Candidate边为**0**。`highest_risk_relation`只是每个cluster内按最小dHash距离、再按最高64×64相关度选出的**展示代表**，并非风险概率或物理同源判定。未被现有哈希/命名/视觉审计发现的同果、同树、同场次关系仍可能存在。

## Representation policy（候选）

| 层 | 规则 |
|---|---|
| Raw package | 原始ZIP/JSON只读、保留版本和SHA256；任何派生文件由manifest追溯。 |
| Dataset-provided `_resize` | 1406张全部不进入正式实验候选池；这与Stage 2B-4H用户决定一致。训练管线以后动态resize输入。 |
| Dataset-provided offline augmentation | 1406张非resize中，明显`_brightness`为**170**、`_noise`为**117**、其他识别后缀为**0**，共**287**；不作为验证/测试独立样本。本页研究更保守的训练候选：也不将它们当独立训练样本，未来仅在train内在线增强。 |
| Plain/base | 文件名无上述后缀的表示为**1119**；每个有base的source group按`(official_split, filename)`字典序确定一个候选canonical表示，不按未来模型结果挑选。 |
| Exact-image duplicate | plain表示经SHA去重为1112个不同字节图像；其中6张虽字节不同但已有其他base处于同一source group，本候选再按组保留一个。得到**1106**个有base组的候选代表。 |
| Annotation conflict | 发现**10个**同字节但polygon不同的SHA关系组，落在**7个source groups**；3个跨官方split（D01–D03），另7个在同一官方split。当前候选把这7个组整体暂存于训练与评价池之外，不替它们选annotation。故本轮`include_candidate=true`为**1099张图、1099个source groups**。 |

另有**6个source groups没有明显plain/base表示**：`sg-0285`、`sg-0341`、`sg-0397`、`sg-0468`、`sg-0482`、`sg-0495`。它们不是被静默丢弃。[候选池manifest](../IOR-YOLO/data/manifests/d2_experiment_pool_candidate.csv)把相关非resize表示标为`no_base_fallback_requires_review`；如果以后决定纳入，确定性fallback是取组内`(official_split, filename)`字典序最小的非resize表示，并且**仅作为train候选**，不能当作已证实的采集base进入val/test。此fallback目前未启用。`sg-0285`还含R01/R02异常resize的原图伴随关系，其处理需维持独立审查。

候选池CSV覆盖全部**2812个包内图像表示**，每行有variant、representation/augmentation type、include_candidate、exclusion reason与annotation policy；它是**候选manifest，不是最终split或导出的派生图像**。U01 `IMG_54350.jpg`在候选池中，派生三类目标只包含3个有效region，#3标为invalid/unknown；真实派生标注尚未生成。

## 确定性seed与透明分配

候选tie-break seed按 `first_64_bits(SHA256(bytes.fromhex(raw_zip_sha256) || b"IOR-YOLO/D2/Stage2B-5P/guard-greedy-v1"))` 得到，本包值为 **13436313853456744620**。同一原包版本与协议版本产生同一seed；它只用于同大小cluster的次序，不依据模型结果、mAP或模拟分布挑选。改变协议文本须显式变更context/version，不能暗中换seed。**此seed尚未冻结为正式seed。**

算法：对当前候选池中有图像的guard cluster，先按ID排序、用上述seed打乱，再稳定按图像数降序排列；逐cluster试放train/val/test，按全局目标相对误差的字典序选择：①图像比例L1，②三类实例比例L1均值，③多类别图像比例L1，④guard-cluster比例L1，最后以train/val/test固定顺序破同分。guard cluster永远整组赋值；同一source group自然不拆。8个仅有已排除表示的guard cluster标为`excluded`，不计入任何模拟split。**未保存逐图/逐组的分配文件。**

## Post-adjudication单次模拟（只作可行性检查）

候选池总计**1099张图、1099个有图像source groups、1088个有图像guard clusters**；三类有效实例 **915 / 425 / 732**，多类别图**105**。下表的source-group数与图像数恰好相同，是本候选一组一表示策略的结果，不是物理独立性证据。另有8个空池guard clusters及13个空池source groups，仍保留在完整约束图中。

| 比例 | split | 图像 / source groups / guard clusters | immature / semi-mature / mature | 多类别图 | 图像目标偏差 |
|---|---|---:|---:|---:|---:|
| 70/15/15 | train | 769 / 769 / 758 | 646 / 297 / 512 | 71 | −0.027 pp |
| | val | 165 / 165 / 165 | 143 / 65 / 115 | 16 | +0.014 pp |
| | test | 165 / 165 / 165 | 126 / 63 / 105 | 18 | +0.014 pp |
| 70/10/20 | train | 769 / 769 / 758 | 642 / 298 / 515 | 71 | −0.027 pp |
| | val | 110 / 110 / 110 | 101 / 42 / 76 | 10 | +0.009 pp |
| | test | 220 / 220 / 220 | 172 / 85 / 141 | 24 | +0.018 pp |

70/15/15的最大三类实例比例偏差为**1.230 pp**，最大多类别图比例偏差为**2.381 pp**；70/10/20分别为**1.202 pp**与**2.857 pp**。两次模拟中，已列原图关系边跨模拟split均为**0**，其中Confirmed、Strongly Supported、Supported和guarded Candidate各自都是0。这只是对**已列边**的机械检查；没有证据证明未知采集关系亦为0。此次未跑多seed搜索，也未根据目标偏差选择“最漂亮”的方案。旧1114组模拟数字不适用于本轮1112组及1099图候选池。

## 复现与停止点

在项目根目录运行下面的只读原包、写候选CSV命令；stdout只有汇总统计，不输出样本split成员：

```zsh
python3 IOR-YOLO/scripts/10_prepare_d2_freeze_candidate.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/data/manifests
```

复算本地ZIP SHA256、关系数、类别数和CSV LF行尾的单元测试见`IOR-YOLO/tests/test_d2_freeze_candidate.py`。本轮**不创建**`train.txt`、`val.txt`、`test.txt`，不选择正式比例/seed，不宣布Dataset Frozen。
