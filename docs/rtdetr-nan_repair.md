# RT-DETR NaN Repair

## Problem

The original CST smoke runs under `runs/detect/cst-sample-5k1k1k-s100` show repeated NaN/Inf losses in RT-DETR variants, so those runs cannot be used as completed training evaluations.

| run                  | rows | last epoch | first NaN epoch | NaN epochs |
| -------------------- | ---: | ---------: | --------------: | ---------: |
| `redetr-p2-init`     |   82 |         82 |               4 |         47 |
| `redetr-WTConv-init` |   34 |         34 |               7 |         13 |
| `redetr-HIFI-init`   |    8 |          8 |               8 |          1 |
| `redetr-r18-init`    |   40 |         40 |               8 |         33 |
| `redetr-MRFPN-init`  |   88 |         88 |              13 |         70 |
| `redetr-l-init`      |  112 |        112 |              15 |         98 |

The original dataset size, `train=5000`, `val=1000`, `test=1000`, makes each RT-DETR epoch slow. For repair validation, use a smaller CST sample first and treat it only as a NaN smoke dataset, not as a formal benchmark.

## Small Smoke Dataset

Generated dataset:

```text
datasets/CST_AntiUAV/cst-sample_train-500_val-100_test-100_seq-20_id-0
```

Generation command:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run utils/dataset/cst2yolo.py \
  --cst_dir datasets/CST_AntiUAV/CST-AntiUAV \
  --output_dir datasets/CST_AntiUAV/cst-sample_train-500_val-100_test-100_seq-20_id-0 \
  --type train val test \
  --number 500 100 100 \
  --selection seq-20 \
  --id 0 \
  --force
```

Dataset check:

| split | images | labels | missing labels | empty labels | boxes | invalid boxes |
| ----- | -----: | -----: | -------------: | -----------: | ----: | ------------: |
| train |    500 |    500 |              0 |           94 |   406 |             0 |
| val   |    100 |    100 |              0 |           14 |    86 |             0 |
| test  |    100 |    100 |              0 |           17 |    83 |             0 |

`data.yaml` contains `train`, `val`, and `test`, and all labels are valid YOLO boxes. Empty labels are expected because `seq-20` preserves continuous CST frames, including frames without visible UAV targets.

## Repair

Use more conservative RT-DETR defaults in `scripts/train_detr.py`:

| parameter        | repaired default                                                                   |
| ---------------- | ---------------------------------------------------------------------------------- |
| `dataset_path`   | `datasets/CST_AntiUAV/cst-sample_train-500_val-100_test-100_seq-20_id-0/data.yaml` |
| `optimizer`      | `AdamW`                                                                            |
| `lr0`            | `0.001`                                                                            |
| `warmup_bias_lr` | `0.0`                                                                              |
| `amp`            | `False`                                                                            |
| `weight_decay`   | `0.0001`                                                                           |

The trainer now also checks `loss` and `loss_items` before backward. If either is NaN/Inf, it skips the optimizer step for that batch and clears gradients, preventing non-finite gradients from corrupting model weights. The existing epoch-level NaN recovery remains as a second line of defense.

## Smoke Validation Commands

Use GPU when available. In this environment, CUDA was visible in the training process as:

```text
CUDA:0 (NVIDIA GeForce RTX 5080, 15840MiB)
```

Start with the baseline and P2 RT-DETR variants:

```bash
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run scripts/train_detr.py \
  --model rtdetr-l.yaml \
  --epochs 10 \
  --batch_size 4 \
  --workers 8 \
  --device 0 \
  --project cst-rtdetr-nan-smoke \
  --name rtdetr-l-nanfix

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run scripts/train_detr.py \
  --model rtdetr-p2.yaml \
  --epochs 10 \
  --batch_size 2 \
  --workers 8 \
  --device 0 \
  --project cst-rtdetr-nan-smoke \
  --name rtdetr-p2-nanfix
```

If both complete without NaN/Inf in `results.csv`, repeat with:

```bash
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run scripts/train_detr.py \
  --model rtdetr-r18.yaml \
  --epochs 10 \
  --batch_size 8 \
  --workers 8 \
  --device 0 \
  --project cst-rtdetr-nan-smoke \
  --name rtdetr-r18-nanfix

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run scripts/train_detr.py \
  --model rtdetr-HIFI.yaml \
  --epochs 10 \
  --batch_size 8 \
  --workers 8 \
  --device 0 \
  --project cst-rtdetr-nan-smoke \
  --name rtdetr-HIFI-nanfix

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run scripts/train_detr.py \
  --model rtdetr-MRFPN.yaml \
  --epochs 10 \
  --batch_size 8 \
  --workers 8 \
  --device 0 \
  --project cst-rtdetr-nan-smoke \
  --name rtdetr-MRFPN-nanfix

env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib uv run scripts/train_detr.py \
  --model rtdetr-WTConv.yaml \
  --epochs 10 \
  --batch_size 8 \
  --workers 8 \
  --device 0 \
  --project cst-rtdetr-nan-smoke \
  --name rtdetr-WTConv-nanfix
```

Acceptance criteria:

- `results.csv` has no NaN/Inf in train or val losses.
- `weights/last.pt` is saved.
- Any skipped non-finite batch is visible as a warning and does not continue into an optimizer step.

## Smoke Validation Results

Syntax checks passed:

```bash
python -m py_compile scripts/train_detr.py ultralytics-git/ultralytics/engine/trainer.py
```

A tiny CPU startup run also completed for `rtdetr-l.yaml` with `epochs=1`, `batch_size=1`, `workers=0`, `device=cpu`, `fraction=0.02`, and `imgsz=320`. This was only a startup sanity check, not the acceptance run.

GPU smoke runs on the 500/100/100 `seq-20` dataset:

| run                    | rows | last epoch | NaN/Inf cells | `last.pt` | `best.pt` |    P(B) |    R(B) | mAP50(B) | mAP50-95(B) |
| ---------------------- | ---: | ---------: | ------------: | :-------- | :-------- | ------: | ------: | -------: | ----------: |
| `rtdetr-l-nanfix`      |   10 |         10 |             0 | yes       | yes       | 0.46473 | 0.23256 |  0.17045 |     0.03436 |
| `rtdetr-p2-nanfix`     |   10 |         10 |             0 | yes       | yes       | 0.35883 | 0.18605 |  0.19653 |     0.11437 |
| `rtdetr-r18-nanfix`    |   10 |         10 |             0 | yes       | yes       | 0.25337 | 0.13953 |  0.04886 |     0.00489 |
| `rtdetr-HIFI-nanfix`   |   10 |         10 |             0 | yes       | yes       | 0.00097 | 0.33721 |  0.00112 |     0.00033 |
| `rtdetr-MRFPN-nanfix`  |   10 |         10 |             0 | yes       | yes       | 0.67986 | 0.23256 |  0.23500 |     0.10626 |
| `rtdetr-WTConv-nanfix` |   10 |         10 |             0 | yes       | yes       | 0.26532 | 0.23256 |  0.12112 |     0.03966 |

Conclusion: with `lr0=0.001`, `warmup_bias_lr=0.0`, `amp=False`, `AdamW`, `weight_decay=0.0001`, and batch-level finite-loss guarding, all six RT-DETR variants completed the 10-epoch smoke validation without NaN/Inf in `results.csv`. Checkpoints were saved normally, including the WTConv variant.
