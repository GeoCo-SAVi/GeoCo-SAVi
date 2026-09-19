from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton as tr
import triton.language as tl


def _odd_kernel(value: int) -> int:
    value = int(value)
    if value < 1 or value % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return value


class ScaleSteeredConv2d(nn.Module):


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        s_ref: float,
        gain_max: float = 2.0,
        learn_gain: bool = True,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _odd_kernel(kernel_size)
        self.s_ref = float(s_ref)
        self.gain_max = float(gain_max)
        self.padding_mode = str(padding_mode)
        if self.padding_mode != "zeros":
            raise ValueError("padding_mode must be zeros")
        if self.s_ref <= 0.0:
            raise ValueError("s_ref must be positive")
        if self.gain_max <= 1.0:
            raise ValueError("gain_max must be greater than one")

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else None
        theta = torch.zeros(())
        if learn_gain:
            self.theta = nn.Parameter(theta)
        else:
            self.register_buffer("theta", theta)

        radius = self.kernel_size // 2
        dy, dx = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer(
            "kernel_offsets_xy",
            torch.stack([dx.flatten(), dy.flatten()], dim=-1),
        )


        self.register_buffer(
            "_effective_s_ref",
            torch.tensor(self.s_ref, dtype=torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size * self.kernel_size
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def gain(self) -> torch.Tensor:


        return torch.exp(math.log(self.gain_max) * torch.tanh(self.theta))

    def set_effective_s_ref(self, value: float | None) -> None:


        value = self.s_ref if value is None else float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("effective s_ref must be finite and positive")
        self._effective_s_ref.fill_(value)

    def forward(self, x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape (batch_slots, channels, H, W)")
        batch, channels, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError("input channel count does not match the layer")
        scales = scales.reshape(-1).to(device=x.device, dtype=torch.float32)
        if scales.numel() != batch:
            raise ValueError("one positive scale is required per rendered slot")
        if not torch._dynamo.is_compiling():
            if not torch.isfinite(scales).all() or bool((scales <= 0).any()):
                raise ValueError("all scales must be finite and positive")

        dilation = (
            scales / self._effective_s_ref.to(device=x.device)
        ) * self.gain.float()
        if self.kernel_size == 1:
            return F.conv2d(x, self.weight, self.bias)

        low = torch.is_autocast_enabled("cuda") and (
            torch.get_autocast_dtype("cuda") == torch.bfloat16
        )
        sampled = sample(
            x.contiguous(memory_format=torch.channels_last),
            dilation.contiguous(),
            self.kernel_size,
            low,
        )
        weight = self.weight.permute(0, 2, 3, 1).reshape(self.out_channels, -1)
        if self.kernel_size == 3 and self.out_channels == 64:
            flat = sampled.permute(0, 2, 3, 1).reshape(batch * height * width, -1)
            output = F.linear(flat, weight, self.bias)
            output = output.reshape(batch, height, width, self.out_channels).permute(0, 3, 1, 2)
        else:
            output = F.conv2d(sampled, weight[:, :, None, None], self.bias)
        return output.to(dtype=x.dtype)


class ScaleSteeredConvBlock(nn.Module):


    def __init__(self, *args, activation: str = "relu", **kwargs) -> None:
        super().__init__()
        self.conv = ScaleSteeredConv2d(*args, **kwargs)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

    def forward(self, x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x, scales))


@tr.jit
def forward_kernel(
    X,
    D,
    Y,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    SWAP: tl.constexpr = False,
):
    n = tl.program_id(1 if SWAP else 0)
    i = tl.program_id(0 if SWAP else 1) * B + tl.arange(0, B)
    m = i < H * W * C * K * K
    c = i % C
    p = (i // C) % (K * K)
    pix = i // (C * K * K)
    dilation = tl.load(D + n)
    sx = dilation * (p % K - K // 2)
    sy = dilation * (p // K - K // 2)
    fx = tl.floor(sx).to(tl.int32)
    fy = tl.floor(sy).to(tl.int32)
    wx = sx - fx
    wy = sy - fy
    xx = pix % W + fx
    yy = pix // W + fy
    base = n * H * W * C + c
    a = tl.load(
        X + base + (yy * W + xx) * C,
        m & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H),
        other=0,
    ).to(tl.float32)
    b = tl.load(
        X + base + (yy * W + xx + 1) * C,
        m & (xx + 1 >= 0) & (xx + 1 < W) & (yy >= 0) & (yy < H),
        other=0,
    ).to(tl.float32)
    d = tl.load(
        X + base + ((yy + 1) * W + xx) * C,
        m & (xx >= 0) & (xx < W) & (yy + 1 >= 0) & (yy + 1 < H),
        other=0,
    ).to(tl.float32)
    e = tl.load(
        X + base + ((yy + 1) * W + xx + 1) * C,
        m & (xx + 1 >= 0) & (xx + 1 < W) & (yy + 1 >= 0) & (yy + 1 < H),
        other=0,
    ).to(tl.float32)
    tl.store(
        Y + n * H * W * C * K * K + i,
        a * (1 - wx) * (1 - wy) + b * wx * (1 - wy) + d * (1 - wx) * wy + e * wx * wy,
        m,
    )


@tr.jit
def backward_kernel(
    X,
    D,
    DY,
    DX,
    PART,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
    TILES: tl.constexpr,
    SWAP: tl.constexpr = False,
):
    n = tl.program_id(1 if SWAP else 0)
    tile = tl.program_id(0 if SWAP else 1)
    i = tile * B + tl.arange(0, B)
    m = i < H * W * C
    c = i % C
    pix = i // C
    dilation = tl.load(D + n)
    result = tl.full((B,), 0, tl.float32)
    rd = tl.full((B,), 0, tl.float32)
    for p in range(K * K):
        ox = p % K - K // 2
        oy = p // K - K // 2
        sx = dilation * ox
        sy = dilation * oy
        fx = tl.floor(sx).to(tl.int32)
        fy = tl.floor(sy).to(tl.int32)
        wx = sx - fx
        wy = sy - fy
        xx = pix % W - fx
        yy = pix // W - fy
        base = n * H * W * C * K * K + p * C + c
        a = tl.load(
            DY + base + (yy * W + xx) * C * K * K,
            m & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H),
            other=0,
        ).to(tl.float32)
        b = tl.load(
            DY + base + (yy * W + xx - 1) * C * K * K,
            m & (xx - 1 >= 0) & (xx - 1 < W) & (yy >= 0) & (yy < H),
            other=0,
        ).to(tl.float32)
        d = tl.load(
            DY + base + ((yy - 1) * W + xx) * C * K * K,
            m & (xx >= 0) & (xx < W) & (yy - 1 >= 0) & (yy - 1 < H),
            other=0,
        ).to(tl.float32)
        e = tl.load(
            DY + base + ((yy - 1) * W + xx - 1) * C * K * K,
            m & (xx - 1 >= 0) & (xx - 1 < W) & (yy - 1 >= 0) & (yy - 1 < H),
            other=0,
        ).to(tl.float32)
        result += (
            a * (1 - wx) * (1 - wy)
            + b * wx * (1 - wy)
            + d * (1 - wx) * wy
            + e * wx * wy
        )
        g = tl.load(DY + base + pix * C * K * K, m, other=0).to(tl.float32)
        xx = pix % W + fx
        yy = pix // W + fy
        base = n * H * W * C + c
        a = tl.load(
            X + base + (yy * W + xx) * C,
            m & (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H),
            other=0,
        ).to(tl.float32)
        b = tl.load(
            X + base + (yy * W + xx + 1) * C,
            m & (xx + 1 >= 0) & (xx + 1 < W) & (yy >= 0) & (yy < H),
            other=0,
        ).to(tl.float32)
        d = tl.load(
            X + base + ((yy + 1) * W + xx) * C,
            m & (xx >= 0) & (xx < W) & (yy + 1 >= 0) & (yy + 1 < H),
            other=0,
        ).to(tl.float32)
        e = tl.load(
            X + base + ((yy + 1) * W + xx + 1) * C,
            m & (xx + 1 >= 0) & (xx + 1 < W) & (yy + 1 >= 0) & (yy + 1 < H),
            other=0,
        ).to(tl.float32)
        rd += g * (
            ox * ((b - a) * (1 - wy) + (e - d) * wy)
            + oy * ((d - a) * (1 - wx) + (e - b) * wx)
        )
    tl.store(DX + n * H * W * C + i, result, m)
    tl.store(PART + n * TILES + tile, tl.sum(rd, 0))


@torch.library.custom_op("geoco_spatial::sample", mutates_args=())
def sample(x: torch.Tensor, d: torch.Tensor, k: int, low: bool) -> torch.Tensor:
    if x.device.type != "cuda":
        raise ValueError("spatial sampling requires a CUDA tensor")
    n, c, h, w = x.shape
    y = torch.empty(
        (n, c * k * k, h, w),
        device=x.device,
        dtype=torch.bfloat16 if low else x.dtype,
        memory_format=torch.channels_last,
    )
    block = 1024
    swap = True
    tiles = tr.cdiv(c * h * w * k * k, block)
    forward_kernel[(tiles, n) if swap else (n, tiles)](
        x, d, y, c, h, w, k, block, SWAP=swap, enable_fp_fusion=False
    )
    return y


@sample.register_fake
def fake(x, d, k, low):
    n, c, h, w = x.shape
    return torch.empty(
        (n, c * k * k, h, w),
        device=x.device,
        dtype=torch.bfloat16 if low else x.dtype,
        memory_format=torch.channels_last,
    )


@torch.library.custom_op("geoco_spatial::backward", mutates_args=())
def backward_op(
    x: torch.Tensor, d: torch.Tensor, dy: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    n, c, h, w = x.shape
    dy = dy.contiguous(memory_format=torch.channels_last)
    block = 1024
    dx = torch.empty_like(x, memory_format=torch.channels_last)
    tiles = tr.cdiv(c * h * w, block)
    part = torch.empty((n, tiles), device=x.device, dtype=torch.float32)
    swap = True
    backward_kernel[(tiles, n) if swap else (n, tiles)](
        x,
        d,
        dy,
        dx,
        part,
        c,
        h,
        w,
        k,
        block,
        tiles,
        SWAP=swap,
        enable_fp_fusion=False,
        num_warps=4,
    )
    return dx, part.sum(1).reshape_as(d)


@backward_op.register_fake
def fake_bwd(x, d, dy, k):
    return torch.empty_like(x, memory_format=torch.channels_last), torch.empty_like(d)


def setup(ctx, inputs, output):
    x, d, k, low = inputs
    ctx.save_for_backward(x, d)
    ctx.k = k


def backward(ctx, dy):
    return (*backward_op(*ctx.saved_tensors, dy, ctx.k), None, None)


sample.register_autograd(backward, setup_context=setup)


@tr.jit
def gather_backward(
    DY, DX, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr, B: tl.constexpr
):
    n = tl.program_id(1)
    i = tl.program_id(0) * B + tl.arange(0, B)
    m = i < C * H * W
    c = i % C
    ix = (i // C) % W
    iy = i // (C * W)
    OH: tl.constexpr = H * 2
    OW: tl.constexpr = W * 2
    sx: tl.constexpr = (W - 1) / (OW - 1)
    sy: tl.constexpr = (H - 1) / (OH - 1)
    startx = tl.maximum(0, tl.ceil((ix - 1) / sx).to(tl.int32))
    starty = tl.maximum(0, tl.ceil((iy - 1) / sy).to(tl.int32))
    value = tl.full((B,), 0, tl.float32)
    for ky in range(5):
        oy = starty + ky
        ry = oy * sy
        fy = tl.floor(ry).to(tl.int32)
        ly = ry - fy
        wy = tl.where(iy == fy, 1 - ly, tl.where(iy == fy + 1, ly, 0))
        for kx in range(5):
            ox = startx + kx
            rx = ox * sx
            fx = tl.floor(rx).to(tl.int32)
            lx = rx - fx
            wx = tl.where(ix == fx, 1 - lx, tl.where(ix == fx + 1, lx, 0))
            v = tl.load(
                DY + ((n * OH + oy) * OW + ox) * C + c,
                m
                & (oy >= 0)
                & (oy < OH)
                & (ox >= 0)
                & (ox < OW)
                & (wy != 0)
                & (wx != 0),
                other=0,
            ).to(tl.float32)
            value += v * wy * wx
    tl.store(DX + n * C * H * W + i, value, m)


@torch.library.custom_op("geoco_resize::forward", mutates_args=())
def upsample(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cuda" or min(x.shape[-2:]) < 2:
        raise ValueError(
            "bilinear resizing requires a CUDA tensor with spatial dimensions at least two"
        )
    return torch.nn.functional.interpolate(
        x, scale_factor=2.0, mode="bilinear", align_corners=True
    )


@upsample.register_fake
def upsample_fake(x):
    n, c, h, w = x.shape
    return torch.empty(
        (n, c, h * 2, w * 2),
        device=x.device,
        dtype=x.dtype,
        memory_format=torch.channels_last,
    )


@torch.library.custom_op("geoco_resize::backward", mutates_args=())
def upsample_backward_op(dy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    n, c, oh, ow = dy.shape
    dy = dy.contiguous(memory_format=torch.channels_last)
    dx = torch.empty(
        (n, c, h, w),
        device=dy.device,
        dtype=dy.dtype,
        memory_format=torch.channels_last,
    )
    gather_backward[(tr.cdiv(c * h * w, 512), n)](
        dy, dx, c, h, w, 512, enable_fp_fusion=False
    )
    return dx


@upsample_backward_op.register_fake
def fake_back(dy, h, w):
    return torch.empty(
        (dy.shape[0], dy.shape[1], h, w),
        device=dy.device,
        dtype=dy.dtype,
        memory_format=torch.channels_last,
    )


def upsample_setup(ctx, inputs, output):
    ctx.hw = inputs[0].shape[-2:]


def upsample_backward(ctx, dy):
    return upsample_backward_op(dy, *ctx.hw)


upsample.register_autograd(upsample_backward, setup_context=upsample_setup)
