# Anti-DETR 代码库概览

本目录包含对 Anti-DETR 项目的分模块代码整理文档。

## 文档索引

| 文档 | 内容 |
|------|------|
| [01-project-overview.md](01-project-overview.md) | 项目总览：目录结构、技术栈、核心创新点 |
| [02-anti-detr-modules.md](02-anti-detr-modules.md) | Anti-DETR 自定义模块详解（anti_detr.py 全部类） |
| [03-loss-functions.md](03-loss-functions.md) | 损失函数体系：DETRLoss、YOLO Loss、Inner-IoU |
| [04-metrics-and-iou.md](04-metrics-and-iou.md) | bbox_iou 计算与 Inner-IoU 增强 |
| [05-training-scripts.md](05-training-scripts.md) | 训练入口脚本：train_detr.py、train_yolo.py |
| [06-configuration.md](06-configuration.md) | YAML 配置文件体系（exp_cfg/、cfg/、datasets.yaml/） |
| [07-dataset-utilities.md](07-dataset-utilities.md) | 数据集工具：CST→YOLO 转换、采样、可视化 |
| [08-experiment-tracking.md](08-experiment-tracking.md) | 实验记录与基准测试结果 |

## 快速导航

- **想了解项目整体架构？** → [01-project-overview.md](01-project-overview.md)
- **想查某个自定义模块的实现？** → [02-anti-detr-modules.md](02-anti-detr-modules.md)
- **想了解损失函数怎么计算？** → [03-loss-functions.md](03-loss-functions.md)
- **想了解 Inner-IoU 如何集成？** → [04-metrics-and-iou.md](04-metrics-and-iou.md)
- **想运行训练？** → [05-training-scripts.md](05-training-scripts.md)
- **想添加新的实验配置？** → [06-configuration.md](06-configuration.md)
- **想处理数据集？** → [07-dataset-utilities.md](07-dataset-utilities.md)
- **想查看实验结果？** → [08-experiment-tracking.md](08-experiment-tracking.md)

## 核心文件速查

```
anti-detr/
├── model.py                          # 早期 Anti-DETR 模块原型（已废弃，参考用）
├── default.yaml                      # 自定义训练默认配置
├── scripts/
│   ├── train_detr.py                 # RT-DETR 训练入口
│   └── train_yolo.py                 # YOLO 训练入口
├── exp_cfg/
│   ├── detr/                         # RT-DETR 实验配置（~50个YAML）
│   └── yolo/                         # YOLO 实验配置
├── cfg/                              # 早期 RT-DETR 配置（5个基线）
├── ultralytics-git/
│   └── ultralytics/
│       ├── nn/modules/anti_detr.py   # ★ 核心模块实现
│       ├── nn/modules/__init__.py    # 模块导出
│       ├── nn/tasks.py               # ★ 模型构建 & criterion 初始化
│       ├── utils/loss.py             # YOLO 系列 Loss
│       ├── utils/metrics.py          # ★ bbox_iou（含 Inner-IoU）
│       ├── models/utils/loss.py      # ★ DETR 系列 Loss
│       └── cfg/default.yaml          # Ultralytics 默认配置
├── utils/dataset/                    # 数据集处理工具
└── docs/                             # 文档
    ├── overview/                     # ★ 当前目录：代码整理文档
    ├── inner_iou_integration.md      # Inner-IoU 集成说明
    ├── exps_custom_modules.md        # 自定义模块实验说明
    ├── CST2YOLO.md                   # CST 数据集转换说明
    └── rtdetr-nan_repair.md          # RT-DETR NaN 修复说明
```
