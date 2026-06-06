# CST to YOLO Format Conversion

脚本：`cst2yolo.py`

该脚本将 CST 格式的注释转换为 YOLO 格式。它读取 CST 注释文件，提取目标边界框信息，并将其转换为 YOLO 格式的注释文件。

参数：

| 参数         | 默认值   | 描述                                             |
| ------------ | -------- | ------------------------------------------------ |
| `cst_dir`    | `datasets/CST_AntiUAV/CST-AntiUAV` | CST 数据集的路径，包含 train、val、test 文件夹。 |
| `output_dir` | `datasets/CST_AntiUAV/yolo` | 输出 YOLO 格式数据集的路径。                     |
| `type`       | `train`  | 要转换的数据集类型，可传多个，如 `train val test` |
| `number`     | `0`      | 要转换的图像数量，可传多个；单个值会广播到所有 `type` |
| `ratio`      | 未设置   | 转换图像的比例，范围 1-100；可传多个；单个值会广播 |
| `selection`  | `global` | `global` 全局随机抽样；`seq-X` 连续抽帧；可传多个 |
| `seed`       | `42`     | 随机种子，用于随机选择图像。                     |
| `id`         | `0`      | 类别 ID，可传多个；单个值会广播到所有 `type`     |
| `--force` / `-f` | 关闭 | `output_dir` 非空时不交互，直接清空后覆盖。      |
| `--noforce` / `-nf` | 关闭 | `output_dir` 非空时不交互，直接退出。            |


CST 格式：

```
<cst_dir>/
    train/
        scene1/ (每个场景一个文件夹，场景内的图像是连续帧)
            000001.jpg
            ...
            exist.txt (CST annotation files)
            gt.txt (CST annotation files)
            IR_label.json (CST annotation files)
        scene2/
            ...
        ...
    val/
        ...
    test/
        ...
```

其中 exist.txt 包含每个图像是否存在目标的信息，即JSON 文件中的 exist 数组
gt.txt 包含每个图像的目标边界框信息，浮点数，转换为整数即为[x_ltc, y_ltc, w, h]
即JSON 文件中的 gt 数组，exist为0则 gt 数组为 [0, 0, 0, 0]

YOLO 文件夹布局：

`selection=global` 时，抽中的图片全局打乱，统一保存到 `images/` 和
`labels/`；文件名增加 scene 前缀以避免重名：

```
<output_dir>/
    train/
        images/
            scene1_image1.jpg (symlink to original image)
            ...
        labels/
            scene1_image1.txt (YOLO annotation file)
            ...
    val/
        ...
    test/
        ...
```

`selection=seq-X` 时，`number` 或 `ratio` 仍表示总抽样图片数。脚本每次随机
选择一个 scene，并从中抽取一段连续的、至多 X 帧的序列，直到达到总抽样数；
最后一段可能少于 X 帧。输出保留 scene 结构：

```
<output_dir>/
    train/
        images/
            scene1/
                image1.jpg
        labels/
            scene1/
                image1.txt
```

`number` 与 `ratio` 是互斥参数，不能同时设置；不设置 `ratio` 且 `number=0`
时转换全部图片。抽样总数基于全部图片计算。`exist=0` 或 bbox 无效的帧仍会被
抽取并生成空标签，因此 `seq-X` 保留 scene 内原始视频的真实连续性。

多 split 模式下，`type` 必须传成数组；`number`、`ratio`、`selection`、`id`
可以：

- 只传 1 个值：广播到所有 `type`
- 传与 `type` 相同数量的值：按位置一一对应

例如：

```bash
uv run util/dataset/cst2yolo.py \
  --output_dir datasets/CST_AntiUAV/sample-all \
  --type train val test \
  --number 5000 1000 1000 \
  --selection seq-100 \
  --id 0 \
  --force
```

脚本每次运行后都会在 `<output_dir>/data.yaml` 自动生成或刷新 Ultralytics
数据集配置文件，并写入当前已经存在的 `train`、`val`、`test` split。可直接用于：

```bash
uv run yolo detect train model=cfg/rtdetr-p2.yaml data=<output_dir>/data.yaml
```

如果 `output_dir` 已存在且非空：

- 默认行为：命令行交互确认是否清空，默认 `N`
- `--force` 或 `-f`：不交互，直接清空并覆盖
- `--noforce` 或 `-nf`：不交互，直接退出
