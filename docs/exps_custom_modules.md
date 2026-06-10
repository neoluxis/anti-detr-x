# RT-DETR Custom Module Experiments

本文档说明当前仓库在 Ultralytics `RT-DETR` 主线上新增的实验模块，包含设计动机、代码落地方式、具体替换位置，以及三组实验配置 `Exp1/Exp2/Exp3` 的差异。

## 1. 背景与目标

原始实验思路来自 Anti-DETR 方向上的三类增强：

- `A`: 动态自适应 GWRN
- `B`: 双向递归多尺度 AGCAU
- `C`: 多维降噪 HIFI

这次实现没有切到 YOLO 主线，而是继续沿用当前仓库已经打通的 Ultralytics `RT-DETR` 分支，在下列基础配置上继续做增强：

- `exp_cfg/detr/rtdetr-WTConv.yaml`
- `exp_cfg/detr/rtdetr-MRFPN.yaml`
- `exp_cfg/detr/rtdetr-HIFI.yaml`

对应新增的三组实验配置：

- `exp_cfg/detr/rtdetr-exp1-dgwrn.yaml`
- `exp_cfg/detr/rtdetr-exp2-dgwrn-biagcau.yaml`
- `exp_cfg/detr/rtdetr-exp3-dgwrn-biagcau-mdhifi.yaml`

## 2. 修改入口

核心改动集中在以下位置：

- `ultralytics-git/ultralytics/nn/modules/anti_detr.py`
  - 新增实验模块实现
  - 保留原 `WTConv2d` / `HIFI` 作为对照基线
- `ultralytics-git/ultralytics/nn/modules/__init__.py`
  - 导出新模块给 Ultralytics 解析器使用
- `ultralytics-git/ultralytics/nn/tasks.py`
  - 让 YAML 中的模块名能被正确解析、补全通道参数并实例化

## 3. 新增模块说明

### 3.1 `SparseGlobalAttention`

目标：给动态 GWRN 和 BiAGCAU 提供一个轻量的“背景抑制”能力，尽量压掉天空、云层等大面积无效区域。

实现方式：

- 输入特征先做两种统计：
  - 通道均值图 `avg_map`
  - 通道最大值图 `max_map`
- 将两者拼接后通过一个 `7x7` 卷积生成空间 mask
- 同时使用全局池化和两层 `1x1` 卷积生成通道 gate
- 最终输出是 `通道 gate * 空间 mask * 1x1 投影`

它不是完整的全局自注意力，而是一个更便宜的稀疏门控近似版，优先保证：

- 可以稳定接到现有 RT-DETR 主干/neck
- 不显著拉高显存和训练复杂度

### 3.2 `DynamicWTConv2d`

目标：把原始 `WTConv2d` 从固定小波残差卷积升级为输入自适应版本。

原始 `WTConv2d` 的结构：

- 一条 base depthwise conv 支路
- 一条多层 wavelet decomposition / reconstruction 支路
- 两条支路最后直接相加

`DynamicWTConv2d` 的新增逻辑：

- 保留原始 wavelet 分解、卷积、逆变换流程
- 新增一套全局池化门控网络
- 从输入特征生成两组 gate：
  - `wavelet_gate`
  - `base_gate`
- 输出不再是简单 `base + wavelet`
- 改为 `wavelet_out * wavelet_gate + base_out * base_gate`

这样做的含义是：

- 高频信息更强时，模型可以提升 wavelet 分支占比
- 语义更稳定时，模型可以更依赖普通卷积分支

额外处理：

- 把 `wt_filter`、`iwt_filter`、`stride_filter` 改成 `buffer`
- 这样可以避免训练器把这些固定滤波器错误识别为可训练参数
- 同时保留 `torch.save` 可序列化能力

### 3.3 `BasicBlock_DGWRN`

目标：把动态 GWRN 真正接入 backbone stage，而不是只单独提供一个卷积层。

实现方式基于原 `BasicBlock`：

- `branch2a` 仍是普通 `ConvNormLayer`
- `branch2b` 从普通卷积替换为 `DynamicWTConv2d`
- 在残差相加前再串接一个 `SparseGlobalAttention`

结构顺序：

1. `3x3 ConvNormLayer`
2. `DynamicWTConv2d`
3. `SparseGlobalAttention`
4. shortcut 相加
5. 激活

这样 `BasicBlock_DGWRN` 就成为动态 GWRN 的 stage 级实现单元。

### 3.4 `BiAGCAUBlock`

目标：在不重写 RT-DETR 整个 neck 拓扑的前提下，把“单向融合”升级为“带双向校准的融合块”。

限制条件：

- Ultralytics YAML 里当前 neck 是标准串行图结构
- 不适合一次性改成复杂的 list-input / multi-branch neck

因此这里采用的是“单输入特征上的双路径校准实现”。

实现方式：

- 输入先走两条 `1x1` 投影分支：
  - `top_down`
  - `bottom_up`
- 两条路径内部都叠加多个 `_AGCAUUnit`
- `_AGCAUUnit` 包含：
  - 一个局部 `3x3` 卷积
  - 一个全局上下文 gate
  - 一个 depthwise + pointwise 的轻量对齐支路
  - 残差回写
- `bottom_up` 路径会先做一次平均池化，再插值回原分辨率
- 两条路径输出拼接后，通过 `cross_gate` 生成双向权重
- 最后经 `fuse` 融合并再走一次 `SparseGlobalAttention`

它不是严格意义上的显式 FPN<->PAN 循环递归，而是：

- 在每个融合块内部显式模拟 top-down / bottom-up 两种尺度校准信号
- 再与 RT-DETR 原有多层 neck 拓扑叠加

这样能在工程成本和想法表达之间取得平衡。

### 3.5 `MDHIFI`

目标：把原 `HIFI` 从单一直方图注意增强，扩展为多特征联合增强。

原始 `HIFI` 只做两件事：

- `Attention_histogram`
- FFN + LayerNorm

`MDHIFI` 在此基础上新增两条显式分支：

- 梯度分支
  - 用 Sobel 核计算 `grad_x` / `grad_y`
  - 再合成为梯度幅值图
- 纹理分支
  - 用局部平均池化构造平滑版本
  - 取 `abs(x - pooled)` 作为局部纹理响应

然后将三类特征联合：

- 直方图注意输出 `hist`
- 梯度特征 `grad`
- 纹理特征 `texture`

再通过 `noise_gate` 估计增强强度：

- `src2 = hist + gate * (grad + texture)`

最后仍保留原 HIFI 的残差 + FFN + LayerNorm 形式。

实现意图：

- `hist` 保持原 HIFI 的全局/排序式增强能力
- `grad` 强化边缘与轮廓
- `texture` 强化局部变化
- `noise_gate` 避免对整幅图无差别增强

## 4. 实验配置如何映射到模块

### 4.1 Exp1(A): `rtdetr-exp1-dgwrn.yaml`

目标：只验证动态 GWRN。

具体修改：

- backbone 四个 stage 的 block 类型统一替换为 `BasicBlock_DGWRN`
- neck 仍保持 `RepC3`
- 两个下采样卷积替换为 `DynamicWTConv2d`
- 仍保留原始 `AIFI`

效果上，这一组主要观察：

- 动态小波卷积是否比固定 WTConv 更适合 CST-AntiUAV 小目标
- 稀疏全局门控是否能减少复杂天空背景干扰

### 4.2 Exp2(A+B): `rtdetr-exp2-dgwrn-biagcau.yaml`

目标：在 Exp1 上继续验证双向 AGCAU。

具体修改：

- 保留 Exp1 中全部 DGWRN 改动
- neck 中 4 个 `RepC3` 全部换成 `BiAGCAUBlock`
  - 两个 FPN 融合块
  - 两个 PAN 融合块

这组主要观察：

- backbone 增强后，双向校准式 neck 能否进一步提升多尺度融合质量
- 对位置偏移、弱目标、畸变目标是否更友好

### 4.3 Exp3(A+B+C): `rtdetr-exp3-dgwrn-biagcau-mdhifi.yaml`

目标：在 Exp2 上再叠加多维 HIFI。

具体修改：

- 保留 Exp1 + Exp2 的所有改动
- `AIFI` 后新增一层 `MDHIFI`
- 不删原始 `AIFI`，而是使用 `AIFI + MDHIFI` 串联

这样做的原因：

- `AIFI` 本身是 RT-DETR 高层语义交互的一部分
- 直接替换掉会让改动语义过大
- 串联更符合“进一步增强”的实验逻辑，也更稳定

## 5. 与原始想法相比，工程上做了哪些取舍

这次实现不是论文公式级复刻，而是仓库可训练版 `v1`。主要取舍如下：

### 5.1 动态 GWRN

没有做的部分：

- 没有动态改变小波基类型
- 没有动态改变 wavelet level 数量

实际采用：

- 固定 `db1`
- 固定 `wt_levels`
- 只对 wavelet/base 两条支路做输入自适应加权

原因：

- 保持与现有 `WTConv2d` 兼容
- 降低实现和训练风险

### 5.2 双向 AGCAU

没有做的部分：

- 没有引入 DCNv2
- 没有重写 RT-DETR 为显式循环递归 neck

实际采用：

- 用 `_AGCAUUnit + top/down 双路径 + cross gate`
- 在单模块内部表达双向校准

原因：

- 不引入额外依赖
- 避免大改 Ultralytics YAML 图结构

### 5.3 多维 HIFI

没有做的部分：

- 没有额外引入频域纹理描述子
- 没有做更复杂的噪声估计器

实际采用：

- Sobel 梯度
- 局部平滑差分纹理
- 一层轻量 `noise_gate`

原因：

- 计算代价可控
- 与现有 `HIFI` 接口完全兼容

## 6. 兼容性与验证

当前这些模块已经通过以下检查：

- YAML 可被 Ultralytics 正常解析
- 随机前向可跑通
- 新模块可序列化 `torch.save`
- `Exp1` 已完成最小训练烟测并成功落 checkpoint

为了避免 WTConv 老问题回归，还额外处理了：

- 小波滤波器不再作为 `Parameter`
- 防止训练器错误把固定滤波器加入优化器

## 7. 后续可继续优化的方向

如果后续要继续往论文版靠近，可以优先考虑：

- 给 `DynamicWTConv2d` 增加更细粒度的频带级 gate，而不是只做两支路 gate
- 给 `BiAGCAUBlock` 增加显式 offset / deformable alignment
- 给 `MDHIFI` 加入更强的噪声建模或频域统计分支
- 单独补充消融：
  - 仅 `SparseGlobalAttention`
  - 仅 `DynamicWTConv2d`
  - `AIFI -> MDHIFI` 替换版 vs 串联版

## 8. 相关文件

- 模块实现：
  - `ultralytics-git/ultralytics/nn/modules/anti_detr.py`
- 模块导出：
  - `ultralytics-git/ultralytics/nn/modules/__init__.py`
- YAML 解析接线：
  - `ultralytics-git/ultralytics/nn/tasks.py`
- 实验配置：
  - `exp_cfg/detr/rtdetr-exp1-dgwrn.yaml`
  - `exp_cfg/detr/rtdetr-exp2-dgwrn-biagcau.yaml`
  - `exp_cfg/detr/rtdetr-exp3-dgwrn-biagcau-mdhifi.yaml`
