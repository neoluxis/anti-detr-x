#!/bin/bash
# IRDST20k 批量训练脚本
# 使用 tsp (Task Spooler) 顺序排队，每次只跑一个 GPU 任务
# 参数与 CST20k 训练保持一致
#
# 用法: bash scripts/irdst20k_batch_train.sh

set -euo pipefail

PROJECT="irdst20k"
DATA="datasets/IRDST_real_yolo/data.yaml"
DEVICE="0"
EPOCHS=100
PATIENCE=10
BATCH=4
IMSZ=640
WORKERS=8
OPTIM="AdamW"
LR0=0.001
LRF=0.01
MOMENTUM=0.937
WD=0.0001
WARMUP_BIAS_LR=0.0
SEED=42

# CRITICAL: COMMON_ARGS must be a SINGLE LINE — multi-line strings break bash -c command parsing.
# Each newline would terminate the uv run command, leaving --project etc. as orphan shell commands.
COMMON_ARGS="--project $PROJECT --dataset_path $DATA --epochs $EPOCHS --patience $PATIENCE --batch_size $BATCH --imgsz $IMSZ --device $DEVICE --workers $WORKERS --optim $OPTIM --lr0 $LR0 --lrf $LRF --momentum $MOMENTUM --weight_decay $WD --warmup_bias_lr $WARMUP_BIAS_LR --seed $SEED --cache --exist-ok"

# Help reduce CUDA memory fragmentation (helps with OOM on large models)
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================"
echo " IRDST20k 批量训练 (CST 参数)"
echo " Project: $PROJECT"
echo " Data: $DATA"
echo " Device: GPU $DEVICE"
echo "============================================"
echo ""

# =============================================
# YOLO 系列 (使用 train_yolo.py)
# =============================================

queue_yolo() {
    local name="$1"
    local model="$2"
    local extra_args="${3:-}"

    tsp -L "$name" bash -c "
        mkdir -p 'runs/detect/$PROJECT' &&
        echo '[\$(date)] Starting $name (model=$model)' &&
        uv run scripts/train_yolo.py \
            --name '$name' \
            --model '$model' \
            --pretrained \
            --close_mosaic 10 \
            $COMMON_ARGS \
            $extra_args \
            2>&1 | tee 'runs/detect/$PROJECT/${name}_train.log' &&
        echo '[\$(date)] $name finished' ||
        { echo '[\$(date)] $name FAILED'; exit 1; }
    "
    echo "  [tsp] $name queued"
}

echo "=== YOLO 系列 ==="

# 1. YOLOv5s (标准 P3-P5)
queue_yolo "yolov5s" "yolov5s.pt"

# 2. YOLOv5s-P2
queue_yolo "yolov5s-p2" "exp_cfg/yolo/yolov5-p2.yaml"

# 3. YOLOv8s (标准 P3-P5)
queue_yolo "yolov8s" "yolov8s.pt"

# 4. YOLOv8s-P2
queue_yolo "yolov8s-p2" "exp_cfg/yolo/yolov8-p2.yaml"

# 5. YOLO11s (标准 P3-P5)
queue_yolo "yolo11s" "yolo11s.pt"

# 6. YOLO11s-P2
queue_yolo "yolo11s-p2" "exp_cfg/yolo/yolo11-p2.yaml"

# 7. YOLO26s (标准 P3-P5, 自定义模型)
queue_yolo "yolo26s" "exp_cfg/yolo/yolo26.yaml"

# 8. YOLO26s-P2
queue_yolo "yolo26s-p2" "exp_cfg/yolo/yolo26-p2.yaml"

# =============================================
# RT-DETR 系列 (使用 train_detr.py)
# =============================================

queue_detr() {
    local name="$1"
    local model="$2"
    local extra_args="${3:-}"

    tsp -L "$name" bash -c "
        mkdir -p 'runs/detect/$PROJECT' &&
        echo '[\$(date)] Starting $name (model=$model)' &&
        uv run scripts/train_detr.py \
            --name '$name' \
            --model '$model' \
            --pretrained true \
            --freeze 0 \
            $COMMON_ARGS \
            $extra_args \
            2>&1 | tee 'runs/detect/$PROJECT/${name}_train.log' &&
        echo '[\$(date)] $name finished' ||
        { echo '[\$(date)] $name FAILED'; exit 1; }
    "
    echo "  [tsp] $name queued"
}

echo ""
echo "=== RT-DETR 系列 ==="

# 9. RT-DETR-L (标准 P3-P5)
queue_detr "rtdetr-l" "rtdetr-l.pt"

# 10. RT-DETR-R18 (标准 P3-P5) — 使用 YAML 配置 (rtdetr-r18.pt 不存在)
queue_detr "rtdetr-r18" "exp_cfg/detr/rtdetr-r18.yaml"

# 11. RT-DETR-R18-P2
queue_detr "rtdetr-r18-p2" "exp_cfg/detr/rtdetr-p2.yaml"

# 12. RT-DETR-R50 (ResNet50 backbone, P3-P5) — 模型较大，降低 batch size 防止 OOM
queue_detr "rtdetr-r50" "exp_cfg/detr/rtdetr-resnet50-last3-pretrained.yaml" "--batch_size 2"

# 13. RT-DETR-R101 (ResNet101 backbone, P3-P5) — 模型较大，降低 batch size 防止 OOM
queue_detr "rtdetr-r101" "exp_cfg/detr/rtdetr-resnet101-last3-pretrained.yaml" "--batch_size 2"

# =============================================
# B3 (自研模型, 使用 train_detr.py)
# =============================================

echo ""
echo "=== B3 ==="

# 14. B3 (SwinV2-Tiny + P2 + HFSCC + MDHIFI) — 模型较大，降低 batch size 防止 OOM
queue_detr "b3" "exp_cfg/0704/b3.yaml" "--batch_size 2"

echo ""
echo "============================================"
echo " 全部 14 个训练任务已加入 tsp 队列"
echo " 查看队列: tsp"
echo " 查看运行日志: tail -f runs/detect/$PROJECT/*_train.log"
echo "============================================"
