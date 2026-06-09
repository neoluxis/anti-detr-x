# Scripts for copy2run


## 后台任务管理

使用 `tsp` (Task Spooler) 来管理后台任务，安装后直接使用 `tsp` 命令即可。

```bash
export TS_MAILTO=
# 列出所有任务
tsp 
# 运行一个命令 (-m 完成后发送邮件通知)
tsp -m "uv run your_command_here"
# 查看某个任务的输出
tsp -c <job_id>
# 杀死某个任务
tsp -k <job_id>
```


## CST-AntiUAV 抽样测试

训练集和验证集必须使用相同的 `output_dir`，才能组成一个 YOLO 数据集。

```bash
export TRAIN_NUM=5000
export VAL_NUM=1000
export TEST_NUM=1000
export SEQ_X=100 # 每次从一个场景连续抽取 100 帧
export ID=0
export OUTPUT_DIR=datasets/CST_AntiUAV/cst-sample_train-${TRAIN_NUM}_val-${VAL_NUM}_test-${TEST_NUM}_seq-${SEQ_X}_id-${ID}

# 一条命令直接生成 train / val / test
uv run util/dataset/cst2yolo.py --cst_dir datasets/CST_AntiUAV/CST-AntiUAV \
    --output_dir "$OUTPUT_DIR" \
    --type train val test \
    --number "$TRAIN_NUM" "$VAL_NUM" "$TEST_NUM" \
    --selection seq-$SEQ_X \
    --id "$ID" \
    --force
```

## YOLO 系列训练

修改 `scripts/train_yolo.py` 后运行

```bash
uv run scripts/train_yolo.py
```

也可以直接命令行传参，或者放在后台任务运行

## DETR 系列训练

修改 `scripts/train_detr.py` 后运行

```bash
uv run scripts/train_detr.py
```
也可以直接命令行传参，或者放在后台任务运行

## 验证结果整理

修改 `scripts/report_best_curves.py` 

```python
# <display name>: <run name>
DEFAULT_MODELS = {
    "rtdetr-l": "redetr-l-init-2",
    "rtdetr-p2": "rtdetr-p2-init",
    "rtdetr-r18": "rtdetr-r18-init",
    "rtdetr-HIFI": "rtdetr-hifi-init-2",
    "rtdetr-WTConv": "rtdetr-wtconv-init",
    "rtdetr-MRFPN": "rtdetr-mrfpn-init",
    "yolo26s-p2": "yolo26s-p2-init",
    "yolo11s-p2": "yolo11s-p2-init",
    "yolov8s-p2": "yolov8s-p2-init",
    "yolov5s-p2": "yolov5s-p2-init",
    "yolo26s": "yolo26s-pretrained",
    "yolo11s": "yolo11s-pretrained",
    "yolov8s": "yolov8s-pretrained",
    "yolov5s": "yolov5s-pretrained-2",
}
```

key 为图表中展示的名称，value 为对应的 run name。

```bash
uv run scripts/report_best_curves.py
```
