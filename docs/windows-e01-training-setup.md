# E01 Windows RTX 3070 frozen handoff

目标平台为Windows工作站RTX 3070。以下命令在**PowerShell**中从仓库根目录执行。机器执行清单见[工作站 checklist](e01-workstation-checklist.md)。本页顺序已经冻结；任一步失败即停止，不自动越过第10步。

1. 在工作站安装/更新NVIDIA驱动、Conda、Git和GitHub CLI；在PowerShell执行`nvidia-smi`，确认RTX 3070与可用显存。PyTorch CUDA 12.1 wheel自带对应运行时，不要求单独安装完整CUDA Toolkit；驱动仍须兼容。安装来源及对应torch/vision版本见[PyTorch官方历史安装表](https://docs.pytorch.org/get-started/previous-versions/)。如果`nvidia-smi`或后续CUDA检查失败，先解决驱动/环境，不运行E01。

2. **Step 1 — Git verification。** 用户通过GitHub官方交互授权并clone/pull已审核提交。实验相关脚本、配置、requirements和冻结manifest必须来自提交版本；允许用户已知的无关本地修改，但它们不能影响E01。runner会阻止E01相关路径存在未提交变化，并把HEAD及完整`git status --porcelain`写入manifest。不要把密码或Token放入项目或聊天。新机器可执行：

```powershell
gh auth login
gh repo clone chentong0601/IOR-YOLO
cd IOR-YOLO
```

已有本地克隆则在仓库根目录执行：

```powershell
git pull
git status --short
git rev-parse HEAD
```

3. **Step 2 — Python environment。** 创建Python 3.11 Conda环境并安装固定Windows CUDA wheel及依赖：

```powershell
conda create -n ior-e01 python=3.11 -y
conda activate ior-e01
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r IOR-YOLO/requirements-e01.txt
python -m pip check
```

4. **Step 3 — CUDA/GPU verification。** 验证Windows RTX 3070、CUDA及固定版本；检查不跑benchmark：

```powershell
python IOR-YOLO/scripts/14_check_training_environment.py --require-cuda
```

5. **Step 4 — Raw ZIP verification。** 将同一D2原包放到`IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip`，不要提交Git。Raw可以从Mac复制；必须验证固定SHA：

```powershell
(Get-FileHash IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip -Algorithm SHA256).Hash.ToLower()
```

必须等于`049591afd4fc3529aedbd31ef9119f5ec0601ebcb8cbbd4a25b7e883fde147ce`，否则停止。

6. **Step 5–6 — Derived build and validation。** Windows必须从raw ZIP和frozen manifest重新生成派生集；不把Mac的processed目录当可信来源：

```powershell
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py build
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py validate
```

7. **Step 7 — Resolved config。** 核对配置、正式池及设备，输出须为`e01_yolo11n_seg`和`cuda:0`；正式run目录必须尚不存在。解析固定8.3.220的请求/default/有效预训练参数：

```powershell
python IOR-YOLO/scripts/15_e01_run.py preflight --require-cuda
```

```powershell
python IOR-YOLO/scripts/16_resolve_e01_config.py
```

8. **Step 8 — Pretrained weight verification。** 官方`yolo11n-seg.pt`只作初始化；命令可能可见下载，并输出文件名、Ultralytics 8.3.220来源说明、SHA256及文件时间。获取失败就停止，禁止改为随机初始化：

```powershell
python IOR-YOLO/scripts/15_e01_run.py verify-weights
```

正式run manifest会再次保存权重文件名、来源、SHA256和文件时间。

9. **Step 9 — Tiny CUDA smoke。** 使用同一派生数据及配置做一次可丢弃的1 epoch、2%数据CUDA可行性检查；它不是正式结果：

```powershell
python IOR-YOLO/scripts/15_e01_run.py smoke
```

batch8明确CUDA OOM时，保留报错证据并只允许8→4：

```powershell
python IOR-YOLO/scripts/15_e01_run.py smoke --batch 4 --oom-note "Pre-training RTX 3070 smoke produced CUDA OOM at batch 8"
```

batch变化只用于硬件可行性，不属于性能优化；不比较多个batch，不在正式训练后改变batch。

10. **Step 10 — User confirmation。** 用户亲自检查Git、环境、数据、resolved config、权重和smoke输出，决定是否启动。任何脚本不得自动从smoke进入正式训练。

11. **Step 11 — Formal train。** 仅在第10步通过后，由用户在Windows VS Code Terminal可见启动一次：

```powershell
python IOR-YOLO/scripts/15_e01_run.py train
```

batch8是预先声明的显存安全选择。若**正式训练前**的极小GPU smoke明确CUDA OOM，可以仅按预定义`8→4`执行，并把真实OOM证据写入命令参数；不做多batch性能比较、不静默改配置：

```powershell
python IOR-YOLO/scripts/15_e01_run.py train --batch 4 --oom-note "Pre-training RTX 3070 smoke produced CUDA OOM at batch 8"
```

以上训练命令按batch8或已有OOM证据的batch4**二选一**，只启动一次正式E01。入口保存Git HEAD/status、预训练权重来源与SHA、训练前resolved配置，并在训练后保存实际auto optimizer/LR、`args.yaml`和checkpoint SHA。

训练完成后必须同时存在`best.pt`、`last.pt`、`results.csv`、`args.yaml`、`run_manifest.yaml`、`resolved_train_config.yaml`及Ultralytics标准图表；缺一不能标记COMPLETED。固定`best.pt`后先正式Val：

```powershell
python IOR-YOLO/scripts/15_e01_run.py val
```

```powershell
python IOR-YOLO/scripts/18_export_e01_predictions.py --split val
python IOR-YOLO/scripts/17_analyze_e01_results.py --split val
```

只有training完成、best固定、Val完成、配置不再变化且分析协议固定后，才执行一次最终Test：

```powershell
python IOR-YOLO/scripts/15_e01_run.py test --final-test
python IOR-YOLO/scripts/18_export_e01_predictions.py --split test --final-test
python IOR-YOLO/scripts/17_analyze_e01_results.py --split test --final-test
```

Test不得用于修改模型、augmentation、optimizer、epoch预算或checkpoint选择。只有training、固定best、Val、final Test、完整manifest、provenance核验和分析产物全部完成，才能写`E01 = COMPLETED`。
