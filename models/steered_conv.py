from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        coordinate_dtype = torch.float32
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=coordinate_dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=coordinate_dtype),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).view(1, 1, height, width, 2)
        pixel_spacing = torch.tensor(
            [
                2.0 / max(width - 1, 1),
                2.0 / max(height - 1, 1),
            ],
            device=x.device,
            dtype=coordinate_dtype,
        )
        offsets = self.kernel_offsets_xy.to(x.device) * pixel_spacing
        sample_grid = base_grid + (
            dilation.view(batch, 1, 1, 1, 1)
            * offsets.view(1, -1, 1, 1, 2)
        )

        taps = self.kernel_size * self.kernel_size
        sample_input = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        sampled = F.grid_sample(
            sample_input[:, None]
            .expand(-1, taps, -1, -1, -1)
            .reshape(batch * taps, self.in_channels, height, width),
            sample_grid.reshape(batch * taps, height, width, 2),
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=True,
        ).reshape(batch, taps, self.in_channels, height, width)

        weight = self.weight.float().reshape(
            self.out_channels, self.in_channels, taps
        )
        output = torch.einsum("bkihw,oik->bohw", sampled, weight)
        if self.bias is not None:
            output = output + self.bias.float().view(1, -1, 1, 1)
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

