# E01 Windows RTX 3070 setup and visible runbook

目标平台为Windows工作站RTX 3070。以下命令在**PowerShell**中执行；从仓库根目录运行（根目录中有`PROJECT.md`与子目录`IOR-YOLO`）。当前Mac M5不安装CUDA、不执行正式训练；无独显联想电脑不作为正式平台。

1. 在工作站安装/更新NVIDIA驱动、Conda、Git和GitHub CLI；在PowerShell执行`nvidia-smi`，确认RTX 3070与可用显存。PyTorch CUDA 12.1 wheel自带对应运行时，不要求单独安装完整CUDA Toolkit；驱动仍须兼容。安装来源及对应torch/vision版本见[PyTorch官方历史安装表](https://docs.pytorch.org/get-started/previous-versions/)。如果`nvidia-smi`或后续CUDA检查失败，先解决驱动/环境，不运行E01。

2. 用户在本机通过GitHub官方交互授权，并克隆/拉取项目。仓库是Private，Stage 3A文件需先由用户自行纳入其Git工作流；不要把GitHub密码或Token放入项目或聊天。新机器可执行：

```powershell
gh auth login
gh repo clone chentong0601/IOR-YOLO
cd IOR-YOLO
```

已有本地克隆则在仓库根目录执行：

```powershell
git pull
```

3. 创建Python 3.11 Conda环境并安装**固定**Windows CUDA wheel及项目依赖。为防`requirements-e01.txt`重装Torch，先装官方GPU wheels，再装其余固定依赖并复核版本：

```powershell
conda create -n ior-e01 python=3.11 -y
conda activate ior-e01
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r IOR-YOLO/requirements-e01.txt
python -m pip check
```

4. 验证当前设备确为Windows RTX 3070，PyTorch显示CUDA可用且版本吻合。检查脚本不跑benchmark：

```powershell
python IOR-YOLO/scripts/14_check_training_environment.py --require-cuda
```

5. 用户将**同一固定D2原包**放在仓库内被Git忽略的`IOR-YOLO/data/raw/multistage_apple_v4/dataset-20260508.zip`。可由自己的本地存储介质传递；不要把raw包提交Git。脚本会先比较ZIP SHA256、冻结manifest及协议SHA，不匹配就停止。

6. 从冻结manifest重建并检查derived segmentation dataset；这会产生被Git忽略的图像和标签目录，绝不更改raw。`dataset.yaml`省略绝对path，以文件位置解析数据根目录，适配Windows和Mac：

```powershell
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py build
python IOR-YOLO/scripts/13_build_e01_ultralytics_dataset.py validate
```

7. 核对E01配置、正式池及设备，确认命令输出为`experiment_id=e01_yolo11n_seg`和`cuda:0`；核对`IOR-YOLO/configs/experiments/e01_yolo11n_seg.yaml`及输出目录尚不存在。正式训练入口还要求Git工作区干净，以将配置和代码精确对应到一个提交。可先在终端查看从**固定8.3.220包**解析的请求/default/有效预训练参数；`auto`的实际optimizer要以trainer运行记录为准：

```powershell
python IOR-YOLO/scripts/15_e01_run.py preflight --require-cuda
```

```powershell
python IOR-YOLO/scripts/16_resolve_e01_config.py
```

8. **仅在上述检查全部通过、用户决定正式开始时**，由用户在工作站VS Code Terminal可见启动一次E01训练。官方`yolo11n-seg.pt`首次加载可能触发可见的Ultralytics权重下载；记下获取来源及脚本自动记录的文件SHA。训练日志、权重、图片和运行记录位于Git忽略的`IOR-YOLO/runs/e01_yolo11n_seg/seed_0/`：

```powershell
python IOR-YOLO/scripts/15_e01_run.py train
```

batch8是预先声明的显存安全选择。若**正式训练前**的极小GPU smoke明确CUDA OOM，可以仅按预定义`8→4`执行，并把真实OOM证据写入命令参数；不做多batch性能比较、不静默改配置：

```powershell
python IOR-YOLO/scripts/15_e01_run.py train --batch 4 --oom-note "Pre-training RTX 3070 smoke produced CUDA OOM at batch 8"
```

以上两条训练命令**二选一**，只启动一次正式E01。入口在正式启动前生成Git忽略的`seed_0_resolved_train_config.yaml`，训练后将实际auto optimizer/LR写入run目录的resolved配置和manifest。

9. 训练完成后，用户先查看`best.pt`、`last.pt`、`args.yaml`和`run_manifest.yaml`，再运行正式Val并检查box/mask指标。只有模型已固定且Val已记录后，才运行一次最终Test；不要利用Test改配置或择模：

```powershell
python IOR-YOLO/scripts/15_e01_run.py val
```

```powershell
python IOR-YOLO/scripts/15_e01_run.py test --final-test
```

`train`、`val`和`test`均要求Windows RTX 3070 CUDA。若显存不足、版本不合、标签异常或实际evaluator不符合U01普通背景FP语义，停止并记录；不要悄悄改冻结数据、batch、seed或协议。Mac可用CPU/MPS做极小smoke，但产出的指标不得当作正式E01结果。

Val指标记录后，可运行`18_export_e01_predictions.py --split val`及`17_analyze_e01_results.py --split val`生成实例匹配与误差图。Test只有最终模型确定、上面的`test --final-test`已执行并记录后，才可为只读导出与分析各自显式传`--split test --final-test`。分析输出位于忽略的`runs/e01_yolo11n_seg/seed_0/analysis/`及`figures/`；操作说明见[分析协议](e01-analysis-protocol.md)。
