# 04 - bbox_iou 与 Inner-IoU 增强

> **文件位置**: `ultralytics-git/ultralytics/utils/metrics.py#L105`

## 函数签名

```python
def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    SIoU: bool = False,         # ★ 新增
    InnerIoU: bool = False,     # ★ 新增
    InnerSIoU: bool = False,    # ★ 新增
    ratio: float = 1.0,         # ★ 新增 (Inner-IoU 缩放因子)
    eps: float = 1e-7,
) -> torch.Tensor:
```

## 支持的 IoU 变体

| 标志 | 计算内容 | 公式 |
|------|---------|------|
| (none) | Standard IoU | `inter / union` |
| `GIoU=True` | Generalized IoU | `IoU - (C - union) / C` |
| `DIoU=True` | Distance IoU | `IoU - rho²/c²` |
| `CIoU=True` | Complete IoU | `IoU - (rho²/c² + αv)` |
| `SIoU=True` | SCYLLA IoU | `IoU - 0.5*(distance + shape)` |
| `InnerIoU=True` | Inner CIoU | `inner_IoU - (rho²/c² + αv)` |
| `InnerSIoU=True` | Inner SIoU | `inner_IoU - 0.5*(distance + shape)` |

## 执行顺序

```
1. 坐标归一化 (xywh → xyxy)

2. 如果 InnerIoU 或 InnerSIoU:
   计算内框坐标 (ratio 缩放):
     inner_b1_x1 = cx1 - iw1 * ratio
     inner_b1_x2 = cx1 + iw1 * ratio
     inner_b1_y1 = cy1 - ih1 * ratio
     inner_b1_y2 = cy1 + ih1 * ratio
     (对 box2 同样)

3. 计算标准 inter / union / IoU

4. 按优先级匹配 (从上到下第一个 True):
   InnerIoU → Inner-CIoU (内框 IoU + 外框 CIoU 惩罚)
   InnerSIoU → Inner-SIoU (内框 IoU + 外框 SIoU 惩罚)
   SIoU → Standard SIoU
   CIoU/DIoU/GIoU → 对应的标准变体
   否则 → Standard IoU
```

## Inner-IoU 详解

### 核心思想

Inner-IoU 使用**按 ratio 缩放的辅助内框**替代原始框计算 IoU，使损失函数对小目标和高精度定位更加敏感。

```
ratio = 1.0: 内框 = 外框，等价于标准 IoU
ratio < 1.0: 内框缩小，损失更关注中心区域对齐
```

### Inner-CIoU 计算

```python
# 1. 内框 IoU
inner_inter = intersect(inner_box1, inner_box2)
inner_union = area1*ratio² + area2*ratio² - inner_inter
inner_iou = inner_inter / inner_union

# 2. 外框 CIoU 惩罚项 (与原版相同)
cw, ch = enclosing_box_size
c2 = cw² + ch²         # 闭包对角线平方
rho2 = center_dist²    # 中心距离平方
v = (4/π²) * (atan(w2/h2) - atan(w1/h1))²  # 长宽比一致性
alpha = v / (v - iou + 1 + eps)

# 3. 返回值
return inner_iou - (rho2/c² + v*alpha)  # 值域约 [-1.5, 1.0]
```

### Inner-SIoU 计算

与 Inner-CIoU 类似，但惩罚项使用 SIoU 的角度+距离+形状代价：

```python
# SIoU 惩罚:
angle_cost = cos(2*arcsin(sin_alpha) - π/2)
distance_cost = 2 - exp(γ*ρ_x) - exp(γ*ρ_y)
shape_cost = Σ (1-exp(-ω))⁴

return inner_iou - 0.5 * (distance_cost + shape_cost)
```

## 标准 SIoU 详解

SIoU (SCYLLA IoU) 的惩罚项由三部分组成：

1. **角度代价** (`angle_cost`): 衡量两框中心连线的角度，引导预测框向 GT 方向对齐
2. **距离代价** (`distance_cost`): 结合角度信息后的加权中心距离
3. **形状代价** (`shape_cost`): 宽高差异的指数惩罚

## 使用方式

### YOLO 训练
```bash
# Inner-CIoU (推荐 ratio: 0.5-0.8)
uv run scripts/train_yolo.py --iou_type inner_ciou --inner_ratio 0.75

# Standard SIoU
uv run scripts/train_yolo.py --iou_type siou
```

### RTDETR 训练
```bash
# Inner-CIoU
uv run scripts/train_detr.py --iou_type inner_ciou --inner_ratio 0.75

# Inner-SIoU
uv run scripts/train_detr.py --iou_type inner_siou --inner_ratio 0.65
```

## 修改涉及的文件

| 文件 | 改动 |
|------|------|
| `ultralytics-git/ultralytics/utils/metrics.py` | `bbox_iou()` 新增 `SIoU`, `InnerIoU`, `InnerSIoU`, `ratio` 参数及计算分支 |
| `ultralytics-git/ultralytics/cfg/default.yaml` | 新增 `iou_type: ciou` 和 `inner_ratio: 0.75` |
| `ultralytics-git/ultralytics/cfg/__init__.py` | `CFG_FRACTION_KEYS` 加入 `"inner_ratio"` |
| `ultralytics-git/ultralytics/utils/loss.py` | `BboxLoss` 支持 `iou_type`/`inner_ratio` dispatch |
| `ultralytics-git/ultralytics/models/utils/loss.py` | `DETRLoss` 支持 `iou_type`/`inner_ratio` dispatch |
| `ultralytics-git/ultralytics/nn/tasks.py` | `RTDETRDetectionModel.init_criterion()` 透传参数 |
| `scripts/train_yolo.py` | CLI 新增 `--iou_type`, `--inner_ratio` |
| `scripts/train_detr.py` | CLI 新增 `--iou_type`, `--inner_ratio` |

## 向后兼容性

- YOLO 默认 `iou_type="ciou"` → 行为不变
- DETR 默认 `iou_type="giou"` → 行为不变
- TAL 分配器始终使用标准 CIoU，不受影响
- `ratio=1.0` 时 Inner-IoU 等价于标准 IoU（可验证）

## 注意事项

1. `inner_ratio` 仅对 `inner_ciou` / `inner_siou` 生效，对其他类型忽略
2. `inner_ratio` 建议范围 0.5-0.8，过小导致内框面积过小、数值不稳定
3. 从旧 checkpoint 恢复训练时，`iou_type` 从 `args.yaml` 读取
