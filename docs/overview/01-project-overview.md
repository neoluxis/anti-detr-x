# 01 - 项目总览

## 项目定位

**Anti-DETR** 是一个面向**空对空无人机小目标检测**的深度学习项目。基于 Ultralytics RT-DETR 框架，集成了四大核心创新模块（小波残差网络、自适应全局校准注意力、直方图尺度交互特征、端到端检测），解决空对空场景中的小目标特征缺失、多尺度融合弱、背景抗干扰差等问题。

## 目录结构

```
anti-detr/
├── model.py                     # 早期独立模块原型（WTConv2d, HIFI, MPA等）
├── default.yaml                 # 自定义训练默认超参数配置
├── pyproject.toml               # Python 项目配置（uv 管理）
├── plan.md                      # 研究计划与创新方向（中文）
├── cst20k_benchmark.md          # CST20k 数据集各模型最优表现
│
├── scripts/                     # 训练 & 分析脚本
│   ├── train_detr.py            # ★ RT-DETR 训练入口
│   ├── train_yolo.py            # ★ YOLO 训练入口
│   ├── report_best_curves.py    # 验证结果曲线整理
│   ├── visualize_det_heatmap.py # 检测热力图可视化
│   ├── batch_det_heatmaps.py    # 批量热力图
│   ├── convert_irdst_to_yolo.py # IRDST → YOLO 格式转换
│   ├── generate_*.py            # 各类架构图/实验图表生成脚本
│   ├── compare_feature_responses.py  # 特征响应对比
│   ├── search_cam_candidates.py      # CAM 候选搜索
│   └── run_*_experiments.py     # 批量实验运行脚本
│
├── exp_cfg/                     # 实验配置文件
│   ├── detr/                    # RT-DETR 系列（~50个YAML）
│   └── yolo/                    # YOLO 系列（~10个YAML）
│
├── cfg/                         # 早期 RT-DETR 基线配置（5个）
│   ├── rtdetr-p2.yaml
│   ├── rtdetr-r18.yaml
│   ├── rtdetr-HIFI.yaml
│   ├── rtdetr-MRFPN.yaml
│   └── rtdetr-WTConv.yaml
│
├── datasets.yaml/               # 数据集配置
│   ├── Anti-UAV.yaml
│   ├── Hazy.yaml
│   └── UAV.yaml
│
├── ultralytics-git/             # Ultralytics 框架（修改版）
│   └── ultralytics/
│       ├── nn/
│       │   ├── modules/
│       │   │   ├── anti_detr.py     # ★ 自定义模块核心实现
│       │   │   ├── __init__.py      # 模块导出注册
│       │   │   ├── block.py         # 基础 block（Bottleneck, C2f等）
│       │   │   ├── conv.py          # 卷积模块
│       │   │   ├── head.py          # 检测/分割/姿态头
│       │   │   └── transformer.py   # Transformer 组件
│       │   └── tasks.py             # ★ 模型构建/解析 YAML/criterion 初始化
│       ├── utils/
│       │   ├── loss.py              # ★ YOLO 系列 Loss（v8DetectionLoss 等）
│       │   ├── metrics.py           # ★ bbox_iou 计算（含 Inner-IoU）
│       │   └── ...
│       ├── models/utils/
│       │   └── loss.py              # ★ DETR 系列 Loss（DETRLoss, RTDETRDetectionLoss）
│       └── cfg/
│           ├── default.yaml         # Ultralytics 默认配置
│           └── __init__.py          # 配置解析逻辑
│
├── utils/dataset/               # 数据集工具
│   ├── cst2yolo.py              # CST → YOLO 格式转换（核心工具）
│   ├── dataset_sampling.py      # 数据集采样
│   ├── visualize_dataset.py     # 数据集可视化
│   ├── compare_cst_yolo.py      # CST/YOLO 格式对比
│   ├── yolo_mc2sc.py            # 多类别→单类别转换
│   ├── invert_images.py         # 图像反转
│   └── cst2frhybrid.py          # CST → FR-Hybrid 格式
│
├── tools/
│   └── export_preds.py          # 模型预测导出
│
├── docs/                        # 文档
│   ├── overview/                # ★ 当前：分模块代码整理
│   ├── inner_iou_integration.md # Inner-IoU 集成文档
│   ├── exps_custom_modules.md   # 自定义模块实验说明
│   ├── CST2YOLO.md              # CST 数据集转换文档
│   └── rtdetr-nan_repair.md     # NaN 修复记录
│
└── assets/                      # 图表资源
    ├── figures/                 # 各类分析图表
    └── detr0705/                # RT-DETR 查询分配流程图
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch |
| 检测框架 | Ultralytics（修改版） |
| 包管理 | uv |
| 任务调度 | tsp (Task Spooler) |
| 数据集格式 | YOLO（训练）、CST（原始标注） |
| 检测算法 | YOLOv5/v8/v11/v26、RT-DETR |

## 核心创新模块

项目围绕 Anti-DETR 论文的四大创新方向，在代码中实现了以下模块：

### 1. 动态小波残差网络 (DGWRN)
- **基础实现**: `WTConv2d` — 小波卷积（db1 小波 + 多级分解/重构）
- **动态升级**: `DynamicWTConv2d` — 输入自适应的门控小波卷积
- **Stage 集成**: `BasicBlock_DGWRN` — 将动态小波卷积 + 稀疏全局注意力集成到残差块
- **文件**: `ultralytics-git/ultralytics/nn/modules/anti_detr.py`

### 2. 双向自适应全局校准注意力 (BiAGCAU)
- **基础单元**: `_AGCAUUnit` — 局部卷积 + 全局上下文 + 对齐支路
- **双向块**: `BiAGCAUBlock` — Top-Down/Bottom-Up 双路径 + 交叉门控
- **文件**: 同上

### 3. 多维直方图尺度交互特征 (MDHIFI)
- **基础实现**: `HIFI` — 直方图自注意力 + FFN
- **多维升级**: `MDHIFI` — 直方图 + 梯度(Sobel) + 纹理 + 噪声门控
- **轻量版**: `LiteMDHIFI` — 去除直方图分支，仅梯度+纹理
- **辅助模块**: `Attention_histogram` — 排序式直方图自注意力
- **文件**: 同上

### 4. 多尺度特征对齐 (HFSCC)
- **`HFSCC`** — 高分辨率特征空间一致性校准（平滑残差 + 偏移对齐 + 补偿）
- **文件**: 同上

### 5. 辅助增强模块
- **`SparseGlobalAttention`** — 稀疏全局注意力（通道门控 + 空间门控）
- **`DWGConv`** — 深度可分离门控卷积（P2 特征降噪）
- **文件**: 同上

## 训练流程

```
train_detr.py / train_yolo.py
    │
    ├── 解析 CLI 参数（--model, --epochs, --iou_type, --inner_ratio, ...）
    │
    ├── RTDETR(model_path) / YOLO(model_path)
    │   └── model.train(**kwargs)
    │       ├── 构建/加载模型（tasks.py: DetectionModel / RTDETRDetectionModel）
    │       ├── 初始化 Loss Criterion（init_criterion）
    │       │   ├── YOLO: v8DetectionLoss → BboxLoss(iou_type, inner_ratio)
    │       │   └── DETR: RTDETRDetectionLoss → DETRLoss(iou_type, inner_ratio)
    │       └── 训练循环
    │
    └── 输出: runs/<project>/<name>/weights/best.pt
```

## 数据集

主要使用 **CST-AntiUAV** 数据集（空对空无人机检测）：

- 原始格式: CST（每个场景一个文件夹，包含连续帧 + exist.txt + gt.txt）
- 训练格式: YOLO（images/ + labels/ + data.yaml）
- 转换工具: `utils/dataset/cst2yolo.py`
- 常用抽样: 训练集 5000 张，验证集 1000 张，测试集 1000 张，连续抽帧 seq-100

## 修改的 Ultralytics 文件清单

相对于上游 Ultralytics，本项目修改了以下文件：

| 文件 | 改动 |
|------|------|
| `nn/modules/anti_detr.py` | **新增** — 全部自定义模块 |
| `nn/modules/__init__.py` | 导入并导出 anti_detr 模块 |
| `nn/tasks.py` | 导入新模块；`RTDETRDetectionModel.init_criterion()` 透传 `iou_type`/`inner_ratio` |
| `utils/loss.py` | `BboxLoss` 支持 `iou_type`/`inner_ratio`；`v8DetectionLoss` 读取并透传 |
| `utils/metrics.py` | `bbox_iou()` 新增 SIoU/InnerIoU/InnerSIoU 计算分支 |
| `models/utils/loss.py` | `DETRLoss` 支持 `iou_type`/`inner_ratio`；`_get_loss_bbox()` dispatch |
| `cfg/default.yaml` | 新增 `iou_type: ciou` 和 `inner_ratio: 0.75` |
| `cfg/__init__.py` | `CFG_FRACTION_KEYS` 加入 `"inner_ratio"` |
