from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_grid(height: int, width: int, device, dtype) -> torch.Tensor:


    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1).reshape(1, height * width, 2)


class CNNEncoder(nn.Module):


    def __init__(self, in_channels: int = 3, width: int = 64, out_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, width, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(width, width, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(width, width, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(width, out_dim, 5, padding=2),
            nn.ReLU(),
        )

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 4:
            raise ValueError("frames must have shape (B, C, H, W)")
        features = self.network(frames)
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        grid = normalized_grid(height, width, features.device, features.dtype)
        return tokens, grid.expand(batch, -1, -1)


class FrozenDINOv2Encoder(nn.Module):


    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        *,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        input_size: int = 336,
        native_dim: int = 384,
        out_dim: int = 128,
        checkpoint_path: str | None = None,
        pretrained: bool = True,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.native_dim = int(native_dim)
        if backbone is None:
            try:
                import timm
            except ImportError as error:
                raise RuntimeError("MOVi-C requires the optional timm dependency") from error
            kwargs = {"dynamic_img_size": True}
            if checkpoint_path:
                path = Path(checkpoint_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                kwargs["checkpoint_path"] = str(path)
                pretrained = False
            backbone = timm.create_model(
                model_name,
                pretrained=bool(pretrained),
                **kwargs,
            )
        self.backbone = backbone.requires_grad_(False)
        self.backbone.eval()
        self.projection = nn.Sequential(
            nn.LayerNorm(self.native_dim),
            nn.Linear(self.native_dim, int(out_dim)),
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("frames must have shape (B, 3, H, W)")
        images = F.interpolate(
            frames,
            size=(self.input_size, self.input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        images = (images - self.pixel_mean) / self.pixel_std

        captured: list[torch.Tensor] = []
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) < 1:
            raise RuntimeError("DINOv2 backbone must expose transformer blocks")
        hook = blocks[-1].register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        try:
            with torch.no_grad():
                self.backbone.forward_features(images)
        finally:
            hook.remove()
        tokens = captured[0]
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        tokens = tokens[:, prefix_tokens:]
        if tokens.shape[-1] != self.native_dim:
            raise RuntimeError("unexpected DINOv2 token width")
        tokens = self.projection(tokens)

        side = math.isqrt(tokens.shape[1])
        if side * side != tokens.shape[1]:
            raise RuntimeError("DINOv2 patch tokens must form a square grid")
        grid = normalized_grid(side, side, tokens.device, tokens.dtype)
        return tokens, grid.expand(tokens.shape[0], -1, -1)
