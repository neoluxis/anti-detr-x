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

