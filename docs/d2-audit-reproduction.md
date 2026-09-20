# D2 v4 数据审计复现入口

本入口供新研究者从本地 `dataset-20260508.zip` 复核证据链。**只读原包**；命令不解压、修改或删除 Raw Data，也不生成正式 train/val/test。请在项目根目录运行。Pillow 可由 `uv run --with pillow` 在临时缓存运行环境提供；不需要 PyTorch、GPU 或项目虚拟环境。

## 1. 原包身份

确认原包位于 `IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip`，先核对字节与 SHA256。预期为 **290131787 bytes**、`049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`。若不匹配，停止并核对下载版本；不要把其他 ZIP 与现有 manifest 混用。

```zsh
stat -f %z IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip
```

```zsh
shasum -a 256 IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip
```

## 2. 结构、标注和全图解码

中央目录结构检查预期 2826 ZIP members，其中 original JPG 为 train 1041、val 126、test 239；resize 目录各一一对应。成员数不等于图像或实例数。

```zsh
python3 IOR-YOLO/scripts/01_inspect_dataset.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --structure-only
```

读取六份 JSON 与 JPEG 头并检查标注对应关系：

```zsh
python3 IOR-YOLO/scripts/04_audit_d2_zip.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip
```

建立**隔离的复现目录**，避免覆盖项目已有 Stage 2B-3 manifests；下条命令会向该目录写 3 个小型 CSV。

```zsh
mkdir -p IOR-YOLO/reports/dataset_audit/reproduction_manifests
```

```zsh
uv run --with pillow python IOR-YOLO/scripts/05_audit_d2_integrity.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/reports/dataset_audit/reproduction_manifests
```

核验 2812/2812 完整解码、original 1406、三类有效实例 **921/814/839 = 2574**，另有 1 个 Unknown region。该脚本输出 `files_sha256.csv`、`duplicate_report.csv`、`class_counts.csv`；其中精确跨官方 split SHA 重复为 3 组。脚本不修改原包。

## 3. 分组与模拟

对上一步隔离目录重建候选同源组；输出 `source_groups.csv`、`source_group_relations.csv`、`source_family_review.csv`。生成器会读取已记录的Stage 2B-4H[人工裁决输入](../IOR-YOLO/configs/data/d2_source_group_adjudications.json)，当前预期 **1112 groups、67 组跨官方 split、67 个命名跨集家族均为 Strongly Supported**。裁决前的1114/65/65+2是历史结果。`source_group_id` 仅为稳定排序的候选分量号，不是采集 ID。

```zsh
uv run --with pillow python IOR-YOLO/scripts/06_build_source_groups.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/reports/dataset_audit/reproduction_manifests
```

下面的独立验证器是**Stage 2B-4V裁决前**的自动阈值复核器，没有加载Stage 2B-4H人工裁决；它重建旧的1114组图。不要将其对新1112组清单的`membership_exact_match`或`source_group_id_exact_match`预期设为true，也不要将旧图的500×2模拟当作新图结果。当前组图的确定性由`test_d2_source_groups.py`在隔离目录重复生成并比较哈希。

```zsh
uv run --with pillow python IOR-YOLO/scripts/08_verify_d2_groups.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/reports/dataset_audit/reproduction_manifests --out-dir IOR-YOLO/reports/dataset_audit/reproduction_verification --seeds 500
```

旧图复核的source/version/hash和阈值敏感性仍可参考；新图的membership与组号比较要以包含人工裁决的重建测试为准。另可用原模拟器单独重现固定 seed 序列；它只打印汇总 JSON，**不写样本归属**：

```zsh
python3 IOR-YOLO/scripts/07_simulate_group_split.py --archive IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip --manifest-dir IOR-YOLO/reports/dataset_audit/reproduction_manifests --seeds 500
```

## 4. 验证边界

审计确认的是包内文件数、标签计数、字节重复、规则化候选组及模拟图约束；**不是**真实独立果实/树/场次数量，也不能证明所有近重复都被找到。两种模拟不冻结比例或 seed。`08` 的 8×8 块 SSIM 是轻量诊断实现，不等同特定库的标准 SSIM；阈值敏感性必须与所用代码版本一起记录。论文中的来源、视觉阶段语义与许可仍应核对官方数据页面和原始论文；本地 CSV 不能替代那些来源。

运行低成本测试：

```zsh
uv run --with pillow python -m unittest discover -s IOR-YOLO/tests -v
```
