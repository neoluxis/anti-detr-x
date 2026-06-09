"""Anti-DETR custom modules for Ultralytics RT-DETR variants."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = (
    "Attention_histogram",
    "BasicBlock",
    "BasicBlock_DGWRN",
    "BasicBlock_WTConv",
    "BiAGCAUBlock",
    "BottleNeck",
    "BottleNeck_WTConv",
    "Blocks",
    "Conv2d_BN",
    "ConvNormLayer",
    "DynamicWTConv2d",
    "HIFI",
    "LayerNorm",
    "MDHIFI",
    "SparseGlobalAttention",
    "WTConv2d",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def get_activation(act: str, inplace: bool = True):
    """Get an activation module by name."""
    act = act.lower()

    if act == "silu":
        module = nn.SiLU()
    elif act == "relu":
        module = nn.ReLU()
    elif act == "leaky_relu":
        module = nn.LeakyReLU()
    elif act == "gelu":
        module = nn.GELU()
    elif act is None:
        module = nn.Identity()
    elif isinstance(act, nn.Module):
        module = act
    else:
        raise RuntimeError(f"Unsupported activation: {act}")

    if hasattr(module, "inplace"):
        module.inplace = inplace

    return module


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        out = self.act(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__()

        if variant == "a":
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == "d" and stride == 2:
                self.short = nn.Sequential(
                    OrderedDict(
                        [
                            ("pool", nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                            ("conv", ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1)),
                        ]
                    )
                )
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        short = x if self.shortcut else self.short(x)
        out = out + short
        out = self.act(out)

        return out


class BasicBlock_WTConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.branch2b = WTConv2d(ch_out, ch_out)


class BottleNeck_WTConv(BottleNeck):
    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.branch2b = WTConv2d(ch_out, ch_out, stride=stride)


class Blocks(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        block,
        count,
        stage_num,
        act="relu",
        input_resolution=None,
        sr_ratio=None,
        kernel_size=None,
        kan_name=None,
        variant="d",
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            stride = 2 if i == 0 and stage_num != 2 else 1
            shortcut = False if i == 0 else True
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=stride,
                        shortcut=shortcut,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio,
                    )
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=stride,
                        shortcut=shortcut,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size,
                    )
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=stride,
                        shortcut=shortcut,
                        variant=variant,
                        act=act,
                        kan_name=kan_name,
                    )
                )
            else:
                self.blocks.append(
                    block(
                        ch_in,
                        ch_out,
                        stride=stride,
                        shortcut=shortcut,
                        variant=variant,
                        act=act,
                    )
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def switch_to_deploy(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, ifBox=True):
        super().__init__()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 5, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 5, dim * 5, kernel_size=3, stride=1, padding=1, groups=dim * 5, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, "constant", 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0] : hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        channels = c // self.num_heads
        if ifBox:
            q = q.reshape(b, self.num_heads, channels, self.factor, hw)
            k = k.reshape(b, self.num_heads, channels, self.factor, hw)
            v = v.reshape(b, self.num_heads, channels, self.factor, hw)
        else:
            q = q.reshape(b, self.num_heads, channels, hw, self.factor).permute(0, 1, 2, 4, 3)
            k = k.reshape(b, self.num_heads, channels, hw, self.factor).permute(0, 1, 2, 4, 3)
            v = v.reshape(b, self.num_heads, channels, hw, self.factor).permute(0, 1, 2, 4, 3)

        q = q.reshape(b, self.num_heads, channels * self.factor, hw)
        k = k.reshape(b, self.num_heads, channels * self.factor, hw)
        v = v.reshape(b, self.num_heads, channels * self.factor, hw)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = attn @ v
        out = out.reshape(b, self.num_heads, channels, self.factor, hw)
        if ifBox:
            out = out.reshape(b, self.num_heads * channels, self.factor * hw)
        else:
            out = out.permute(0, 1, 2, 4, 3).reshape(b, self.num_heads * channels, hw * self.factor)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x[:, : c // 2].sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x[:, : c // 2] = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, q2, k2, v = qkv.chunk(5, dim=1)

        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        out1 = self.reshape_attn(q1, k1, v, True)
        out2 = self.reshape_attn(q2, k2, v, False)

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)
        out = out1 * out2
        out = self.project_out(out)
        out_replace = out[:, : c // 2]
        out_replace = torch.scatter(out_replace, -1, idx_w, out_replace)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        out[:, : c // 2] = out_replace
        return out


class HIFI(nn.Module):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        super().__init__()
        self.additivetoken = Attention_histogram(c1, num_heads)
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.additivetoken(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class _ChannelGate(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(self.pool(x))


class SparseGlobalAttention(nn.Module):
    def __init__(self, c1, reduction=4, mask_kernel=7):
        super().__init__()
        padding = mask_kernel // 2
        self.channel_gate = _ChannelGate(c1, reduction=reduction)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, mask_kernel, padding=padding, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(c1, c1, 1, bias=False)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        spatial_mask = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        x = self.channel_gate(x)
        return self.proj(x * spatial_mask)


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    if wave != "db1":
        raise NotImplementedError("Anti-DETR WTConv2d currently supports only the db1 wavelet.")

    scale = 2.0**-0.5
    dec_lo = torch.tensor([scale, scale], dtype=type)
    dec_hi = torch.tensor([-scale, scale], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_lo = torch.tensor([scale, scale], dtype=type)
    rec_hi = torch.tensor([scale, -scale], dtype=type)
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters.to(device=x.device, dtype=x.dtype), stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters.to(device=x.device, dtype=x.dtype), stride=2, groups=c, padding=pad)
    return x


class WaveletTransform(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.filters = filters
        with torch.no_grad():
            x = wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = inverse_wavelet_transform(grad_output, ctx.filters)
        return grad, None


class InverseWaveletTransform(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.filters = filters
        with torch.no_grad():
            x = inverse_wavelet_transform(input, filters)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = wavelet_transform(grad_output, ctx.filters)
        return grad, None


def wavelet_transform_init(filters):
    def apply(input):
        return WaveletTransform.apply(input, filters)

    return apply


def inverse_wavelet_transform_init(filters):
    def apply(input):
        return InverseWaveletTransform.apply(input, filters)

    return apply


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return x * self.scale


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type="db1"):
        super().__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        wt_filter, iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.register_buffer("wt_filter", wt_filter, persistent=True)
        self.register_buffer("iwt_filter", iwt_filter, persistent=True)

        self.base_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding="same",
            stride=1,
            dilation=1,
            groups=in_channels,
            bias=bias,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.register_buffer("stride_filter", torch.ones(in_channels, 1, 1, 1), persistent=True)
        else:
            self.stride_filter = None

    def _do_stride(self, x):
        return F.conv2d(
            x,
            self.stride_filter.to(device=x.device, dtype=x.dtype),
            bias=None,
            stride=self.stride,
            groups=self.in_channels,
        )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for level in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = WaveletTransform.apply(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[level](self.wavelet_convs[level](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for level in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = InverseWaveletTransform.apply(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.stride > 1:
            x = self._do_stride(x)

        return x


class DynamicWTConv2d(WTConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type="db1",
        gate_reduction=4,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            wt_levels=wt_levels,
            wt_type=wt_type,
        )
        hidden = max(in_channels // gate_reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels * 2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for level in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = WaveletTransform.apply(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[level](self.wavelet_convs[level](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for level in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = InverseWaveletTransform.apply(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        wavelet_out = next_x_ll
        base_out = self.base_scale(self.base_conv(x))
        wavelet_gate, base_gate = self.gate(x).chunk(2, dim=1)
        x = wavelet_out * wavelet_gate + base_out * base_gate

        if self.stride > 1:
            x = self._do_stride(x)

        return x


class BasicBlock_DGWRN(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act="relu", variant="d"):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.branch2b = DynamicWTConv2d(ch_out, ch_out)
        self.sparse_attn = SparseGlobalAttention(ch_out)

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.sparse_attn(out)
        short = x if self.shortcut else self.short(x)
        out = out + short
        out = self.act(out)
        return out


class _AGCAUUnit(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.local = ConvNormLayer(channels, channels, 3, 1, act="relu")
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.align = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        local = self.local(x)
        aligned = self.align(local)
        gated = aligned * self.context(local)
        return x + self.dropout(gated)


class BiAGCAUBlock(nn.Module):
    def __init__(self, c1, c2, n=3, e=1.0, num_heads=4, dropout=0.0):
        super().__init__()
        hidden = int(c2 * e)
        self.top_down = ConvNormLayer(c1, hidden, 1, 1, act="relu")
        self.bottom_up = ConvNormLayer(c1, hidden, 1, 1, act="relu")
        self.top_path = nn.Sequential(*(_AGCAUUnit(hidden, dropout=dropout) for _ in range(n)))
        self.bottom_path = nn.Sequential(*(_AGCAUUnit(hidden, dropout=dropout) for _ in range(n)))
        self.cross_gate = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden * 2, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = ConvNormLayer(hidden * 2, c2, 1, 1, act="relu")
        self.short = ConvNormLayer(c1, c2, 1, 1, act=None) if c1 != c2 else nn.Identity()
        self.out_gate = SparseGlobalAttention(c2, reduction=max(num_heads, 1))

    def forward(self, x):
        td = self.top_path(self.top_down(x))
        pooled = F.avg_pool2d(self.bottom_up(x), kernel_size=2, stride=2, ceil_mode=True)
        bu = F.interpolate(self.bottom_path(pooled), size=td.shape[-2:], mode="nearest")
        cross = self.cross_gate(torch.cat([td, bu], dim=1))
        td_gate, bu_gate = cross.chunk(2, dim=1)
        fused = torch.cat([td * td_gate, bu * bu_gate], dim=1)
        out = self.fuse(fused)
        return self.out_gate(out) + self.short(x)


class MDHIFI(nn.Module):
    def __init__(
        self,
        c1,
        cm=2048,
        num_heads=8,
        grad_kernel=3,
        texture_kernel=5,
        noise_reduction=4,
        dropout=0.0,
        act=nn.GELU(),
        normalize_before=False,
    ):
        super().__init__()
        self.additivetoken = Attention_histogram(c1, num_heads)
        self.edge_proj = nn.Conv2d(c1, c1, 1, bias=False)
        self.texture_proj = nn.Conv2d(c1, c1, 1, bias=False)
        self.texture_pool = nn.AvgPool2d(texture_kernel, stride=1, padding=texture_kernel // 2)
        hidden = max(c1 // noise_reduction, 8)
        self.noise_gate = nn.Sequential(
            nn.Conv2d(c1 * 3, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)
        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = act
        self.normalize_before = normalize_before

        sobel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, grad_kernel, grad_kernel), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, grad_kernel, grad_kernel), persistent=False)

    def _gradient_branch(self, x):
        channels = x.shape[1]
        weight_x = self.sobel_x.expand(channels, 1, -1, -1).to(dtype=x.dtype, device=x.device)
        weight_y = self.sobel_y.expand(channels, 1, -1, -1).to(dtype=x.dtype, device=x.device)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=channels)
        grad = torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)
        return self.edge_proj(grad)

    def _texture_branch(self, x):
        pooled = self.texture_pool(x)
        texture = torch.abs(x - pooled)
        return self.texture_proj(texture)

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        hist = self.additivetoken(src)
        grad = self._gradient_branch(src)
        texture = self._texture_branch(src)
        gate = self.noise_gate(torch.cat([hist, grad, texture], dim=1))
        src2 = hist + gate * (grad + texture)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)
