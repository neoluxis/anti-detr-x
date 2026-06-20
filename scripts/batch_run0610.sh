#!/bin/bash

export PROJECT=cst-sample-12k4k4k-s100
export TRAIN_NUM=12000
export VAL_NUM=4000
export TEST_NUM=4000
export SEQ_X=100
export ID=0
export DATASET_DIR=datasets/CST_AntiUAV/cst-sample_train-${TRAIN_NUM}_val-${VAL_NUM}_test-${TEST_NUM}_seq-${SEQ_X}_id-${ID}
export DATASET_YAML=${DATASET_DIR}/data.yaml
export HEATMAP_SOURCE=${DATASET_DIR}/val/images


tsp uv run scripts/train_detr.py \
  --project "${PROJECT}" \
  --name rtdetr-resnet50-unfreeze-last3-12k4k4k \
  --model exp_cfg/detr/rtdetr-resnet50-last3-pretrained.yaml \
  --dataset_path "${DATASET_YAML}" \
  --batch_size 16 \
  --epochs 30 \
  --device 0 \
  --freeze 0 \
  --pretrained true

tsp uv run scripts/visualize_det_heatmap.py \
  --model "runs/detect/${PROJECT}/rtdetr-resnet50-unfreeze-last3-12k4k4k/weights/best.pt" \
  --source "${HEATMAP_SOURCE}" \
  --arch detr \
  --device cuda:0 \
  --cam-method eigencam \
  --max-images 50 \
  --outdir runs/heatmap_compare/resnet_last3_20k/rtdetr-resnet50-unfreeze-last3-12k4k4k

tsp uv run scripts/train_detr.py \
  --project "${PROJECT}" \
  --name rtdetr-resnet50-unfreeze-last3-ema-12k4k4k \
  --model exp_cfg/detr/rtdetr-resnet50-last3-pretrained-ema.yaml \
  --dataset_path "${DATASET_YAML}" \
  --batch_size 16 \
  --epochs 30 \
  --device 0 \
  --freeze 0 \
  --pretrained true

tsp uv run scripts/visualize_det_heatmap.py \
  --model "runs/detect/${PROJECT}/rtdetr-resnet50-unfreeze-last3-ema-12k4k4k/weights/best.pt" \
  --source "${HEATMAP_SOURCE}" \
  --arch detr \
  --device cuda:0 \
  --cam-method eigencam \
  --max-images 50 \
  --outdir runs/heatmap_compare/resnet_last3_20k/rtdetr-resnet50-unfreeze-last3-ema-12k4k4k

tsp uv run scripts/train_detr.py \
  --project "${PROJECT}" \
  --name rtdetr-resnet101-unfreeze-last3-12k4k4k \
  --model exp_cfg/detr/rtdetr-resnet101-last3-pretrained.yaml \
  --dataset_path "${DATASET_YAML}" \
  --batch_size 16 \
  --epochs 30 \
  --device 0 \
  --freeze 0 \
  --pretrained true

tsp uv run scripts/visualize_det_heatmap.py \
  --model "runs/detect/${PROJECT}/rtdetr-resnet101-unfreeze-last3-12k4k4k/weights/best.pt" \
  --source "${HEATMAP_SOURCE}" \
  --arch detr \
  --device cuda:0 \
  --cam-method eigencam \
  --max-images 50 \
  --outdir runs/heatmap_compare/resnet_last3_20k/rtdetr-resnet101-unfreeze-last3-12k4k4k

tsp uv run scripts/train_detr.py \
  --project "${PROJECT}" \
  --name rtdetr-resnet101-unfreeze-last3-ema-12k4k4k \
  --model exp_cfg/detr/rtdetr-resnet101-last3-pretrained-ema.yaml \
  --dataset_path "${DATASET_YAML}" \
  --batch_size 16 \
  --epochs 30 \
  --device 0 \
  --freeze 0 \
  --pretrained true

tsp uv run scripts/visualize_det_heatmap.py \
  --model "runs/detect/${PROJECT}/rtdetr-resnet101-unfreeze-last3-ema-12k4k4k/weights/best.pt" \
  --source "${HEATMAP_SOURCE}" \
  --arch detr \
  --device cuda:0 \
  --cam-method eigencam \
  --max-images 50 \
  --outdir runs/heatmap_compare/resnet_last3_20k/rtdetr-resnet101-unfreeze-last3-ema-12k4k4k
  
