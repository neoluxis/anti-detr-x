# 03 - 损失函数体系

## 文件分布

| 文件 | 内容 |
|------|------|
| `ultralytics-git/ultralytics/utils/loss.py` | YOLO 系列 Loss（v8DetectionLoss, BboxLoss 等） |
| `ultralytics-git/ultralytics/models/utils/loss.py` | DETR 系列 Loss（DETRLoss, RTDETRDetectionLoss） |
| `ultralytics-git/ultralytics/utils/metrics.py` | `bbox_iou()` 底层 IoU 计算 |

---

## 一、底层 Loss 组件

### VarifocalLoss
**文件**: `ultralytics-git/ultralytics/utils/loss.py`

```python
VarifocalLoss(gamma=2.0, alpha=0.75)
```

Varifocal Loss，用于处理类别不平衡。对正样本（label=1）使用 gt_score 加权，对负样本使用降权因子。

### FocalLoss
```python
FocalLoss(gamma=1.5, alpha=0.25)
```

标准 Focal Loss。通过 modulating_factor `(1 - p_t)^gamma` 降低易分样本的权重。

### DFLoss
```python
DFLoss(reg_max=16)
```

Distribution Focal Loss。将边界框回归建模为离散概率分布，计算左右相邻 bin 的交叉熵损失。

### BboxLoss
**文件**: `ultralytics-git/ultralytics/utils/loss.py#L110`

```python
BboxLoss(reg_max=16, iou_type="ciou", inner_ratio=1.0)
```

边界框损失。组合 IoU Loss + DFL Loss：

```
loss_iou = (1 - iou) * weight / target_scores_sum
loss_dfl = DFL(pred_dist, target_ltrb) * weight / target_scores_sum
```

**IoU 类型 dispatch**（本项目扩展）:

```python
iou = bbox_iou(
    pred_bboxes, target_bboxes,
    xywh=False,
    CIoU=(iou_type == "ciou"),
    DIoU=(iou_type == "diou"),
    GIoU=(iou_type == "giou"),
    SIoU=(iou_type == "siou"),
    InnerIoU=(iou_type == "inner_ciou"),
    InnerSIoU=(iou_type == "inner_siou"),
    ratio=inner_ratio,
)
```

### MultiChannelDiceLoss / BCEDiceLoss
用于分割任务的 Dice Loss 和 BCE+Dice 组合 Loss。

### KeypointLoss
用于姿态估计的关键点回归损失（OKS-based）。

### RLELoss
Residual Log-Likelihood Estimation Loss，用于姿态估计的概率分布建模。

---

## 二、YOLO 系列 Loss

### v8DetectionLoss
**文件**: `ultralytics-git/ultralytics/utils/loss.py#L347`

YOLOv8/v11/v26 检测任务的损失计算类。

**初始化**:
```python
v8DetectionLoss(model, tal_topk=10, tal_topk2=None)
```
- 从 `model.args` 读取 `iou_type` 和 `inner_ratio` 并传入 `BboxLoss`
- 使用 `TaskAlignedAssigner` 进行标签分配（topk=10）

**核心流程** (`get_assigned_targets_and_loss`):

```
1. 预处理: pred_bboxes = bbox_decode(anchor_points, pred_distri)
2. TaskAlignedAssigner 分配:
   target_bboxes, target_scores, fg_mask = assigner(
       pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes
   )
3. 分类损失:
   loss_cls = BCE(pred_scores, target_scores) / target_scores_sum
4. 边界框损失:
   loss_box, loss_dfl = BboxLoss(pred_distri, pred_bboxes, ...)
5. 加权求和:
   loss = box_gain*loss_box + cls_gain*loss_cls + dfl_gain*loss_dfl
```

**返回**: `(fg_mask, target_indices, ...), loss_tensor, detached_loss`

### v8SegmentationLoss
继承 `v8DetectionLoss`，新增 mask loss（BCE + Dice）。

### v8PoseLoss / PoseLoss26
继承 `v8DetectionLoss`，新增关键点 loss（KeypointLoss + RLE Loss）。

### v8OBBLoss
继承 `v8DetectionLoss`，面向旋转框检测（Rotated BBox），新增角度 loss。

### v8ClassificationLoss
简单的交叉熵分类损失。

### E2ELoss / E2EDetectLoss
端到端（One-to-Many + One-to-One）检测的训练损失。

- `E2EDetectLoss`: 两个独立 `v8DetectionLoss` 的简单求和
- `E2ELoss`: 带衰减策略的 o2m/o2o 加权（o2m 从 0.8 衰减到 0.1）

---

## 三、DETR 系列 Loss

### DETRLoss
**文件**: `ultralytics-git/ultralytics/models/utils/loss.py#L17`

DETR 风格的端到端检测损失。

**初始化**:
```python
DETRLoss(
    nc=80,
    loss_gain={"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1},
    aux_loss=True,
    use_fl=True,       # 使用 Focal Loss 做分类
    use_vfl=False,     # 是否使用 Varifocal Loss
    use_uni_match=False,
    uni_match_ind=0,
    gamma=1.5, alpha=0.25,
    iou_type="giou",       # ★ 本项目扩展
    inner_ratio=1.0,       # ★ 本项目扩展
)
```

**核心流程**:

```
1. HungarianMatcher 二分图匹配:
   match_indices = matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

2. 分类损失 (_get_loss_class):
   - 支持 VFL / FL / BCE 三种模式
   - 正样本使用 IoU 作为 soft label (gt_scores)

3. 边界框损失 (_get_loss_bbox):
   - L1 Loss (bbox)
   - IoU Loss (GIoU/CIoU/DIoU/SIoU/InnerIoU — dispatch 到 bbox_iou)

4. 辅助损失 (_get_loss_aux):
   - 对每个中间 decoder 层计算同样的 loss
   - 可选使用统一匹配 (uni_match) 或各自匹配
```

### RTDETRDetectionLoss
**文件**: `ultralytics-git/ultralytics/models/utils/loss.py#L410`

继承 `DETRLoss`，新增**去噪（Denoising）训练**支持：

```python
class RTDETRDetectionLoss(DETRLoss):
    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        # 1. 标准 DETR loss
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # 2. 去噪 loss (如果有 dn_meta)
        if dn_meta is not None:
            dn_loss = super().forward(dn_bboxes, dn_scores, batch,
                                      postfix="_dn", match_indices=dn_match_indices)
            total_loss.update(dn_loss)
        return total_loss
```

### init_criterion 调用链

**YOLO 路径** (`tasks.py: DetectionModel`):
```python
self.criterion = v8DetectionLoss(self)  # 内部读取 self.args.iou_type / self.args.inner_ratio
```

**DETR 路径** (`tasks.py: RTDETRDetectionModel`):
```python
def init_criterion(self):
    return RTDETRDetectionLoss(
        self,
        nc=self.nc,
        iou_type=getattr(self.args, "iou_type", "giou"),      # 默认 "giou"
        inner_ratio=getattr(self.args, "inner_ratio", 1.0),   # 默认 1.0
    )
```

---

## 四、损失函数配置流

```
CLI: --iou_type inner_ciou --inner_ratio 0.75
  │
  ▼
train_detr.py / train_yolo.py
  → model.train(iou_type="inner_ciou", inner_ratio=0.75)
    │
    ▼
get_cfg(DEFAULT_CFG, overrides) → model.args
  │
  ├── YOLO 路径
  │   DetectionModel.__init__ → v8DetectionLoss(self)
  │     → BboxLoss(reg_max, iou_type="inner_ciou", inner_ratio=0.75)
  │       → bbox_iou(..., InnerIoU=True, ratio=0.75)
  │
  └── RTDETR 路径
      RTDETRDetectionModel.init_criterion()
        → RTDETRDetectionLoss(..., iou_type="inner_ciou", inner_ratio=0.75)
          → DETRLoss._get_loss_bbox()
            → bbox_iou(..., InnerIoU=True, ratio=0.75)
```

---

## 五、TAL (Task-Aligned Assigner)

**文件**: `ultralytics-git/ultralytics/utils/tal.py`

YOLO 系列使用的标签分配策略。基于分类分数和定位质量的联合对齐度量：

```
align_metric = scores^alpha * iou^beta
```

- `alpha=0.5, beta=6.0`: 偏向定位质量
- 选择 topk 个 anchor 作为正样本
- **注意**: TAL 中的 IoU 计算始终使用标准 CIoU，不受 `iou_type` 参数影响

---

## 六、HungarianMatcher

**文件**: `ultralytics-git/ultralytics/models/utils/ops.py`

DETR 系列使用的匈牙利算法二分图匹配。

成本矩阵:
```
cost = class_cost * cost_gain["class"]
     + bbox_cost  * cost_gain["bbox"]
     + giou_cost  * cost_gain["giou"]
```

其中 `class_cost` 使用预测分数取负，`bbox_cost` 使用 L1 距离，`giou_cost` 使用 1-GIoU。
