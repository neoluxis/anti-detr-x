# Inner-IoU 损失函数集成文档

## 概述

将 **Inner-CIoU** 和 **Inner-SIoU** 损失函数集成到 YOLO 和 RTDETR 训练流程中，通过 `--iou_type` 配置项统一控制。

Inner-IoU 的核心思想：在计算 IoU 时，使用一个按 `ratio` 缩放的**辅助内框（auxiliary inner box）**替代原始框，从而使损失函数对小目标和高精度定位更加敏感。

- `ratio = 1.0`：内框退化为原始框，等价于标准 CIoU/SIoU
- `ratio < 1.0`：内框缩小，损失更关注框中心区域的对齐（对小目标检测友好）

参考论文：*Inner-IoU: More Effective Intersection over Union Loss with Auxiliary Bounding Box*

---

## 支持的 IoU 类型

| `iou_type` 值 | 损失类型 | 说明 |
|---------------|---------|------|
| `ciou` | Complete IoU | **YOLO 默认**，含中心距离 + 长宽比惩罚 |
| `diou` | Distance IoU | 含中心距离惩罚 |
| `giou` | Generalized IoU | **RTDETR 默认**，含最小闭包区域惩罚 |
| `siou` | SCYLLA IoU | 含角度 + 距离 + 形状惩罚 |
| `inner_ciou` | Inner Complete IoU | 内框 CIoU，需配合 `inner_ratio` |
| `inner_siou` | Inner SCYLLA IoU | 内框 SIoU，需配合 `inner_ratio` |

---

## 使用方法

### YOLO 训练

```bash
# 默认 CIoU（无需改动）
uv run scripts/train_yolo.py \
    --model exp_cfg/yolo/yolo11.yaml \
    --dataset_path datasets/xxx/data.yaml \
    --epochs 100 --batch_size 32

# 使用 Inner-CIoU（推荐 ratio: 0.5~0.8）
uv run scripts/train_yolo.py \
    --model exp_cfg/yolo/yolo11.yaml \
    --dataset_path datasets/xxx/data.yaml \
    --epochs 100 --batch_size 32 \
    --iou_type inner_ciou --inner_ratio 0.75

# 使用 Inner-SIoU
uv run scripts/train_yolo.py \
    --model exp_cfg/yolo/yolo11.yaml \
    --dataset_path datasets/xxx/data.yaml \
    --epochs 100 --batch_size 32 \
    --iou_type inner_siou --inner_ratio 0.65
```

### RTDETR 训练

```bash
# 默认 GIoU（无需改动）
uv run scripts/train_detr.py \
    --model exp_cfg/detr/rtdetr-l.yaml \
    --dataset_path datasets/xxx/data.yaml \
    --epochs 100 --batch_size 2

# 使用 Inner-CIoU
uv run scripts/train_detr.py \
    --model exp_cfg/detr/rtdetr-l.yaml \
    --dataset_path datasets/xxx/data.yaml \
    --epochs 100 --batch_size 2 \
    --iou_type inner_ciou --inner_ratio 0.75

# 使用 SIoU（标准 SIoU，ratio 不生效）
uv run scripts/train_detr.py \
    --model exp_cfg/detr/rtdetr-l.yaml \
    --dataset_path datasets/xxx/data.yaml \
    --epochs 100 --batch_size 2 \
    --iou_type siou
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 取值范围 | 说明 |
|------|------|--------|---------|------|
| `--iou_type` | str | YOLO: `ciou`, DETR: `giou` | `ciou`, `diou`, `giou`, `siou`, `inner_ciou`, `inner_siou` | IoU 损失类型 |
| `--inner_ratio` | float | `0.75` | `0.0 ~ 1.0` | 内框缩放因子，仅在 `inner_ciou`/`inner_siou` 时生效 |

---

## 参数流

```
train_yolo.py / train_detr.py --iou_type inner_ciou --inner_ratio 0.75
    │
    ▼
YOLO().train(**kwargs) / RTDETR().train(**kwargs)
    │
    ▼
get_cfg(DEFAULT_CFG, overrides) → model.args
    │
    ├── YOLO 路径
    │   v8DetectionLoss(model)
    │     → BboxLoss(iou_type="inner_ciou", inner_ratio=0.75)
    │       → bbox_iou(..., InnerIoU=True, ratio=0.75)
    │
    └── RTDETR 路径
        RTDETRDetectionModel.init_criterion()
          → RTDETRDetectionLoss(iou_type="inner_ciou", inner_ratio=0.75)
            → DETRLoss._get_loss_bbox()
              → bbox_iou(..., InnerIoU=True, ratio=0.75)
```

---

## 修改的文件清单

| 文件 | 改动内容 |
|------|---------|
| `ultralytics-git/ultralytics/utils/metrics.py` | `bbox_iou()` 新增 `SIoU`, `InnerIoU`, `InnerSIoU`, `ratio` 参数，添加 Inner-CIoU / Inner-SIoU / SIoU 三个计算分支 |
| `ultralytics-git/ultralytics/cfg/default.yaml` | 新增 `iou_type: ciou` 和 `inner_ratio: 0.75` 配置项 |
| `ultralytics-git/ultralytics/cfg/__init__.py` | `CFG_FRACTION_KEYS` 加入 `"inner_ratio"` |
| `ultralytics-git/ultralytics/utils/loss.py` | `BboxLoss.__init__` 接受 `iou_type`/`inner_ratio`；`forward()` 用 dispatch 替代硬编码 `CIoU=True`；`v8DetectionLoss` 从 `model.args` 读取并透传 |
| `ultralytics-git/ultralytics/models/utils/loss.py` | `DETRLoss.__init__` 接受 `iou_type`/`inner_ratio`（默认 `giou`）；`_get_loss_bbox()` 用 dispatch 替代硬编码 `GIoU=True` |
| `ultralytics-git/ultralytics/nn/tasks.py` | `RTDETRDetectionModel.init_criterion()` 从 `self.args` 读取并透传 |
| `scripts/train_yolo.py` | 新增 `--iou_type`（默认 `ciou`）和 `--inner_ratio` CLI 参数 |
| `scripts/train_detr.py` | 新增 `--iou_type`（默认 `giou`）和 `--inner_ratio` CLI 参数 |

---

## 向后兼容性

- **YOLO**：不传 `--iou_type` 时默认 `ciou`，与原有行为完全一致
- **RTDETR**：不传 `--iou_type` 时默认 `giou`，与原有行为完全一致
- **TAL 分配阶段**保持不变（始终使用 CIoU）
- 从旧 checkpoint 恢复训练时，`iou_type` 会从 `args.yaml` 读取，行为一致

---

## 注意事项

1. `inner_ratio` 仅对 `inner_ciou` 和 `inner_siou` 生效，其他类型忽略此参数
2. `inner_ratio` 建议范围 0.5~0.8，过小会导致内框面积过小，数值不稳定
3. TAL（Task-Aligned Assigner）分配阶段始终使用标准 CIoU，不受 `iou_type` 影响
4. `ratio=1.0` 时 Inner-CIoU/SIoU 完全等价于标准 CIoU/SIoU（可验证）
