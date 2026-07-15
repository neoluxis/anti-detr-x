# 02 - Anti-DETR 自定义模块详解

> **文件位置**: `ultralytics-git/ultralytics/nn/modules/anti_detr.py`

## 模块总览

```python
__all__ = (
    "Attention_histogram", "BasicBlock", "BasicBlock_DGWRN", "BasicBlock_WTConv",
    "BiAGCAUBlock", "BottleNeck", "BottleNeck_WTConv", "Blocks", "Conv2d_BN",
    "ConvNormLayer", "DWGConv", "DynamicWTConv2d", "HFSCC", "HIFI", "LayerNorm",
    "LiteMDHIFI", "MDHIFI", "SparseGlobalAttention", "WTConv2d",
)
```

---

## 一、基础组件

### ConvNormLayer
**用途**: 标准 Conv + BatchNorm + Activation 组合，是其他模块的基础构建块。

```python
ConvNormLayer(ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None)
```

### BasicBlock
**用途**: ResNet 风格的基础残差块（expansion=1）。两个 3×3 卷积 + shortcut 连接。

```
branch_2a (3x3 ConvNorm) → branch2b (3x3 ConvNorm) → +shortcut → Act
```

### BottleNeck
**用途**: ResNet 风格的瓶颈残差块（expansion=4）。1×1 → 3×3 → 1×1 + shortcut。

### Blocks
**用途**: 将多个 BasicBlock/BottleNeck 串联为一个 stage。第一个 block 负责 stride=2 下采样和维度变换，后续 block 为 stride=1 的恒等映射。

### LayerNorm
**用途**: 支持 channels_first 和 channels_last 两种数据格式的 LayerNorm 实现。

### Conv2d_BN
**用途**: Conv2d + BatchNorm2d 的顺序组合，支持部署时的融合（`switch_to_deploy`）。

### autopad / get_activation
**用途**: 工具函数。`autopad` 计算 'same' padding；`get_activation` 通过名称字符串获取激活函数。

---

## 二、小波相关组件

### create_wavelet_filter
**用途**: 使用 PyWavelets 创建 db1 小波的分解/重构滤波器组。

- 分解滤波器 (dec_filters): 形状 `[in_size, 1, 2, 2]` → 重复为 `[in_size, 4, 2, 2]`
- 重构滤波器 (rec_filters): 同理，用于逆变换

### wavelet_transform / inverse_wavelet_transform
**用途**: 2D 离散小波变换（DWT）和逆变换（IDWT）的前向计算。

- 使用 group convolution 实现（groups=c），每个通道独立进行小波变换
- 输出形状: `[B, C, 4, H/2, W/2]`（4 个频带: LL, LH, HL, HH）

### WaveletTransform / InverseWaveletTransform
**用途**: `torch.autograd.Function` 子类，将 wavelet transform 包装为可微操作。

- forward: 执行 DWT/IDWT（使用 `no_grad`）
- backward: 使用对应的逆变换传递梯度

---

## 三、WTConv2d — 小波卷积

```
输入 x
  ├── base_conv (DW Conv 5x5) → base_scale → x1
  │
  └── 多级小波分解:
        for level in wt_levels:
          ├── DWT → [LL, LH, HL, HH]
          ├── LL → 下一级分解
          └── wavelet_conv(4C → 4C) → scale
        ───────────────────────
        逆序重建:
          for level in reversed:
            LL + next_x_ll → concat([LL, LH, HL, HH])
            → IDWT → next_x_ll
        → x2
  ─────────────────────────────
  x1 + x2 → (可选 stride) → 输出
```

**关键参数**:
- `in_channels == out_channels`（仅支持同通道）
- `wt_levels`: 小波分解级数（默认 1）
- `wt_type`: 小波基类型（仅支持 "db1"）
- `kernel_size`: base_conv 和 wavelet_conv 的卷积核大小（默认 5）

**设计意图**: 在标准 depthwise 卷积的基础上叠加小波域的多尺度频率分解，增强对不同频率特征的感知能力。

### _ScaleModule
**用途**: 可学习的标量缩放模块。`output = weight * input`，权重初始化为 `init_scale`。

---

## 四、DynamicWTConv2d — 动态自适应小波卷积

**继承**: `WTConv2d`

**新增**: 输入自适应的门控机制

```
输入 x
  ├── 小波分支 (同 WTConv2d)
  │     → wavelet_out
  │
  ├── base 分支 (同 WTConv2d)
  │     → base_out
  │
  └── Gate 网络 (全局池化 → FC → ReLU → FC → Sigmoid → 2C)
        → [wavelet_gate, base_gate]
  ─────────────────────────────
  wavelet_out * wavelet_gate + base_out * base_gate → (可选 stride) → 输出
```

与 WTConv2d 的差异：输出不再是简单相加，而是由输入内容动态决定两支路的权重比例。高频信息丰富时提升小波分支权重，语义稳定时偏重普通卷积分支。

---

## 五、SparseGlobalAttention — 稀疏全局注意力

**用途**: 轻量级背景抑制模块。用于抑制天空、云层等大面积无效区域。

```
输入 x
  ├── avg_map = mean(x, dim=1)        # 通道均值
  ├── max_map = max(x, dim=1)         # 通道最大值
  ├── spatial_mask = Conv([avg, max])  # 7x7 卷积 → Sigmoid
  ├── channel_gate = SE(x)            # 全局池化 → FC → ReLU → FC → Sigmoid
  └── proj(channel_gate * x * spatial_mask)  # 1x1 投影
```

**关键参数**:
- `reduction`: SE 模块的压缩比（默认 4）
- `mask_kernel`: 空间 mask 的卷积核大小（默认 7）

---

## 六、DWGConv — 深度可分离门控卷积

**用途**: 轻量级 P2 特征降噪块。用于 FPN 中 P2 特征投影后抑制高频噪声。

```
输入 x
  ├── DWConv(3x3, groups=C) → SE-Gate → Proj(1x1) → out
  └── Short(1x1) ──────────────────────────────────→ +
```

---

## 七、BasicBlock 变体

### BasicBlock_WTConv
**继承**: `BasicBlock`，将 `branch2b` 替换为 `WTConv2d`。

### BottleNeck_WTConv
**继承**: `BottleNeck`，将 `branch2b` 替换为 `WTConv2d`。

### BasicBlock_DGWRN — ★ 动态 GWRN Stage 单元
**继承**: `BasicBlock`，组合使用动态小波卷积 + 稀疏注意力：

```
branch2a: 3x3 ConvNormLayer
  → branch2b: DynamicWTConv2d
  → SparseGlobalAttention
  → + shortcut → Act
```

---

## 八、HIFI 系列 — 直方图尺度交互特征

### Attention_histogram — 排序式直方图自注意力

核心创新：对特征图进行**空间排序**后再做注意力计算。

```
输入 x [B, C, H, W]
  ├── 对前 C/2 通道: 按行排序 → 按列排序
  ├── QKV = DWConv(QKV_proj(x))
  ├── Q1, K1, Q2, K2, V = chunk(5)
  ├── V 按值排序
  ├── Q1, K1, Q2, K2 按 V 的排序索引 gather
  ├── reshape_attn(Q1, K1, V, ifBox=True)   → out1
  ├── reshape_attn(Q2, K2, V, ifBox=False)  → out2
  ├── out = out1 * out2 → project_out
  └── 对前 C/2 通道: 恢复原始排序 → 输出
```

**设计意图**: 排序操作使注意力在"有序空间"中进行，对像素强度分布更敏感，有利于低对比度目标的特征增强。

### HIFI — 基础直方图交互特征

```
输入 x
  → Attention_histogram(x)
  → x + Dropout(hist_out) → LayerNorm
  → FFN(Conv 1x1 → Act → Conv 1x1)
  → x + Dropout(ffn_out) → LayerNorm → 输出
```

### MDHIFI — ★ 多维特征增强 HIFI

在 HIFI 基础上新增**梯度分支**和**纹理分支**：

```
输入 x
  ├── hist = Attention_histogram(x)
  ├── grad = Sobel(x) → edge_proj       # 梯度幅值
  ├── texture = |x - AvgPool(x)| → texture_proj  # 纹理响应
  ├── gate = noise_gate([hist, grad, texture])   # 噪声门控
  ├── enhanced = hist + gate * (grad + texture)
  ├── x + Dropout(enhanced) → LayerNorm
  └── FFN → x + Dropout → LayerNorm → 输出
```

**三个分支的含义**:
- `hist`: 保持原 HIFI 的全局/排序式增强能力
- `grad` (Sobel): 强化边缘与轮廓特征
- `texture` (局部差分): 强化局部纹理变化
- `noise_gate`: 自适应抑制噪声区域的增强

### LiteMDHIFI — 轻量多维 HIFI

去除 Attention_histogram 分支，仅保留梯度 + 纹理 + 噪声门控 + FFN。计算量更小。

---

## 九、BiAGCAUBlock — 双向自适应全局校准注意力

**设计思想**: 在单个融合块内部模拟 Top-Down / Bottom-Up 两种尺度校准信号。

```
输入 x [B, C1, H, W]
  ├── top_down: 1x1 Conv → _AGCAUUnit × N
  │     → td_features
  │
  └── bottom_up: 1x1 Conv → AvgPool(2x) → _AGCAUUnit × N
        → Interpolate(up) → bu_features
  ─────────────────────────────────────
  cross_gate([td, bu]) → [td_gate, bu_gate]
  fused = Conv([td*td_gate, bu*bu_gate]) → 融合
  → SparseGlobalAttention → + shortcut → 输出
```

### _AGCAUUnit（内部单元）

```
输入 x
  ├── local: 3x3 ConvNorm (局部特征)
  ├── context: AvgPool → 1x1 Conv → Sigmoid (全局上下文门控)
  ├── align: DWConv 3x3 → 1x1 Conv → BN (对齐支路)
  └── x + Dropout(align(local) * context(local)) → 输出
```

**关键参数**:
- `n`: 每条路径的 _AGCAUUnit 堆叠数量（默认 3）
- `e`: 内部隐藏通道扩展比例
- `num_heads`: SparseGlobalAttention 的 reduction 参数

---

## 十、HFSCC — 高分辨率特征空间一致性校准

**用途**: 对 FPN 的 P3/P4/P5 三层输出进行平滑残差提取 + 空间偏移对齐 + 一致性补偿。

```
输入 [f3, f4, f5]  # P3, P4, P5 特征图

1. 平滑残差提取:
   r3 = f3 - smooth(f3)
   r4 = f4 - smooth(f4)
   r5 = f5 - smooth(f5)

2. 多尺度对齐:
   r34 = Down(r3) → ShiftAlign(r34, r4)  # 找到 r3→r4 的最佳偏移
   r45 = Down(r4) → ShiftAlign(r45, r5)  # 同理

3. 一致性估计:
   c4 = consistency4([r34_aligned, r4, |diff|, product])
   c5 = consistency5([r45_aligned, r5, |diff|, product])
   c3 = Interpolate(c4)

4. 补偿:
   d3 = delta3([f3, r3, c3])
   d4 = delta4(pair_features4)
   d5 = delta5(pair_features5)

5. 输出:
   p3 = f3 + gamma3 * c3 * d3
   p4 = f4 + gamma4 * c4 * d4  (- beta4*(1-c4)*noise4)
   p5 = f5 + gamma5 * c5 * d5  (- beta5*(1-c5)*noise5)
```

**关键参数**:
- `cm`: 隐藏通道压缩比
- `smooth_kernel`: 平滑卷积核大小
- `use_noise_suppression`: 是否启用噪声抑制
- `use_p3_gate`: P3 是否使用一致性门控

---

## 十一、模块依赖关系图

```
BasicBlock / BottleNeck (基础残差块)
  ├── BasicBlock_WTConv / BottleNeck_WTConv (小波残差)
  └── BasicBlock_DGWRN (动态小波 + 稀疏注意力)

WTConv2d (小波卷积)
  └── DynamicWTConv2d (动态自适应小波卷积)
        ├── create_wavelet_filter
        ├── WaveletTransform / InverseWaveletTransform
        └── _ScaleModule

HIFI → Attention_histogram → HIFI (基础)
  ├── MDHIFI (+Sobel梯度 +纹理 +噪声门控)
  └── LiteMDHIFI (仅梯度+纹理，无直方图注意力)

BiAGCAUBlock → _AGCAUUnit + SparseGlobalAttention

HFSCC → _DepthwiseSmoothing + _ShiftAlign + _CompensationHead

SparseGlobalAttention → _ChannelGate
DWGConv (独立使用的轻量降噪块)
```

## 注意事项

1. **WTConv2d 仅支持 db1 小波**，且要求 `in_channels == out_channels`。
2. **小波滤波器**在 `DynamicWTConv2d` 中使用 `register_buffer`，而在 `WTConv2d` 中使用 `nn.Parameter(requires_grad=False)`。
3. **BiAGCAUBlock** 不是严格意义上的显式 FPN↔PAN 循环递归，而是在单模块内部用双路径模拟双向校准。
4. **MDHIFI** 的 Sobel 核注册为 `buffer(persistent=False)`，不会保存到 checkpoint。
5. **Attention_histogram** 使用 `rearrange`（einops）进行张量重塑，在 ultralytics 版本中已改为原生 PyTorch reshape/permute 实现以避免依赖。
