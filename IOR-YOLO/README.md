# IOR-YOLO 工程工作区

依据上一级 `IOR-YOLO_项目工程说明书.md` 第 3 节建立。IOR-YOLO 为候选工程名称，不代表方法已冻结。研究与 Skill 文档统一保留在上一级；后续 Git 根目录应覆盖上一级研究项目，避免将本子目录单独设为仓库而隔离研究记录与项目技能。

```text
IOR-YOLO/
├── configs/{data,experiments,corruption}/
├── data/
│   ├── raw/{orchard_apple,fuji_ripeness}/
│   ├── interim/
│   ├── processed/
│   │   ├── orchard_apple/{images,labels}/{train,val,test}/
│   │   └── fuji_external/
│   ├── splits/
│   └── manifests/
├── ior_yolo/{augment,models,losses,trainers,evaluators,utils}/
├── scripts/
├── tests/
├── runs/E00 … E10/
├── reports/{tables,figures,predictions,failure_cases}/
└── third_party/
```

当前只建立目录和说明，未创建训练代码、配置或测试占位实现。原说明书规定的脚本名称在实际开发时采用；`requirements-lock.txt` 与 `ultralytics-commit.txt` 在环境安装和版本冻结后生成；split、manifest、权重及报告须来自真实操作。

从本目录运行工程脚本；从上一级管理研究档案及版本控制。根 `.gitignore` 排除原始/处理中数据、训练运行、权重和大体积预测，同时保留 `data/splits/` 与 `data/manifests/` 的可追溯清单。
