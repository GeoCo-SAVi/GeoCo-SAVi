from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .steered_conv import ScaleSteeredConv2d, ScaleSteeredConvBlock


def _coordinate_grid(size: int, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=-1)


def _gather_slot(field: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:


    if field.ndim == 4:
        return field.gather(1, indices[:, None])[:, 0]
    if field.ndim == 5:
        expanded = indices[:, None, None].expand(-1, 1, field.shape[2], -1, -1)
        return field.gather(1, expanded)[:, 0]
    raise ValueError("field must be (B,N,H,W) or (B,N,C,H,W)")


@dataclass
class DecoderOutput:
    reconstruction: torch.Tensor
    reconstruction_pre_selector: torch.Tensor
    slot_rgb: torch.Tensor
    alpha: torch.Tensor
    alpha_pre_selector: torch.Tensor
    mask_logits: torch.Tensor
    mask_logits_pre_selector: torch.Tensor
    spatial_features: torch.Tensor
    selector: dict[str, torch.Tensor] | None


class SelectiveAlphaSelector(nn.Module):


    def __init__(self, feature_dim: int, embedding_dim: int = 8) -> None:
        super().__init__()
        if embedding_dim != 8:
            raise ValueError("the published selector uses an 8-D projection")
        self.feature_dim = int(feature_dim)
        self.feature_projection = nn.Linear(self.feature_dim, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(8 + 2 * embedding_dim + 3, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        expected = 8 + 2 * self.feature_dim + 3
        if raw_features.shape[-1] != expected:
            raise ValueError(f"expected {expected} raw selector features")
        scalars = raw_features[..., :8].float()
        owner = raw_features[..., 8 : 8 + self.feature_dim].float()
        difference = raw_features[
            ..., 8 + self.feature_dim : 8 + 2 * self.feature_dim
        ].float()
        rgb_difference = raw_features[..., -3:].float()
        owner_projected = self.feature_projection(owner)

        difference_projected = F.linear(
            difference,
            self.feature_projection.weight.float(),
            None,
        )
        features = torch.cat(
            [
                scalars,
                owner_projected,
                difference_projected,
                rgb_difference,
            ],
            dim=-1,
        )
        return self.mlp(features).squeeze(-1)


class ScaleSteeredDecoder(nn.Module):


    def __init__(
        self,
        *,
        appearance_dim: int,
        num_slots: int,
        image_size: int,
        broadcast_size: int,
        hidden_dim: int = 64,
        output_channels: int = 3,
        scales_factor: float = 5.0,
        s_ref: float = 0.2,
        gain_max: float = 2.0,
        learn_gain: bool = True,
        canonical_kernel: int | None = None,
        upsample_kernels: Iterable[int] = (),
        output_kernels: Iterable[int] = (),
        rgb_kernel: int = 3,
        alpha_kernel: int = 1,
        rgb_micro: bool = False,
        rgb_micro_eta_max: float = 0.5,
        selector_enabled: bool = False,
        selector_threshold: float = 0.5,
        selector_max_delta: float = 0.0,
        selector_label_max_delta: float = 2.0,
        selector_takeover_margin: float = 1e-3,
        selector_border_width: int = 4,
        selector_owned_min: int = 20,
        selector_peak_min: float = 0.3,
        selector_bg_score_eta: float = 0.25,
    ) -> None:
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.num_slots = int(num_slots)
        self.image_size = int(image_size)
        self.broadcast_size = int(broadcast_size)
        self.hidden_dim = int(hidden_dim)
        self.output_channels = int(output_channels)
        self.scales_factor = float(scales_factor)
        self.rgb_micro_enabled = bool(rgb_micro)
        self.rgb_micro_eta_max = float(rgb_micro_eta_max)
        self.selector_label_max_delta = float(selector_label_max_delta)
        self.selector_takeover_margin = float(selector_takeover_margin)
        self.selector_border_width = int(selector_border_width)
        self.selector_owned_min = int(selector_owned_min)
        self.selector_peak_min = float(selector_peak_min)
        self.selector_bg_score_eta = float(selector_bg_score_eta)

        upsample_kernels = tuple(int(value) for value in upsample_kernels)
        output_kernels = tuple(int(value) for value in output_kernels)
        expected_size = self.broadcast_size * (2 ** len(upsample_kernels))
        if expected_size != self.image_size:
            raise ValueError(
                "broadcast_size * 2**len(upsample_kernels) must equal image_size"
            )
        if self.selector_label_max_delta <= 0.0:
            raise ValueError("selector_label_max_delta must be positive")

        spatial_kwargs = dict(
            s_ref=s_ref,
            gain_max=gain_max,
            learn_gain=learn_gain,
        )
        self.grid_projection = nn.Linear(2, self.appearance_dim)
        current_channels = self.appearance_dim
        self.canonical_block: ScaleSteeredConvBlock | None = None
        if canonical_kernel is not None:
            self.canonical_block = ScaleSteeredConvBlock(
                current_channels,
                self.hidden_dim,
                int(canonical_kernel),
                **spatial_kwargs,
            )
            current_channels = self.hidden_dim

        self.upsample_blocks = nn.ModuleList()
        for kernel in upsample_kernels:
            block = ScaleSteeredConvBlock(
                current_channels,
                self.hidden_dim,
                kernel,
                **spatial_kwargs,
            )
            self.upsample_blocks.append(block)
            current_channels = self.hidden_dim

        self.output_blocks = nn.ModuleList()
        for kernel in output_kernels:
            block = ScaleSteeredConvBlock(
                current_channels,
                self.hidden_dim,
                kernel,
                **spatial_kwargs,
            )
            self.output_blocks.append(block)
            current_channels = self.hidden_dim


        self.rgb_style = nn.Linear(self.appearance_dim, current_channels)
        self.rgb_fusion = nn.Conv2d(
            2 * current_channels, current_channels, kernel_size=1
        )
        self.rgb_head = ScaleSteeredConv2d(
            current_channels,
            self.output_channels,
            rgb_kernel,
            **spatial_kwargs,
        )
        self.rgb_micro_head = None
        if self.rgb_micro_enabled:
            self.rgb_micro_head = nn.Conv2d(
                current_channels,
                self.output_channels,
                kernel_size=3,
                padding=1,
            )
            nn.init.zeros_(self.rgb_micro_head.weight)
            nn.init.zeros_(self.rgb_micro_head.bias)
        self.alpha_head = ScaleSteeredConv2d(
            current_channels,
            1,
            alpha_kernel,
            **spatial_kwargs,
        )
        self.selector = (
            SelectiveAlphaSelector(current_channels) if selector_enabled else None
        )
        self.register_buffer(
            "_selector_threshold",
            torch.tensor(float(selector_threshold)),
            persistent=False,
        )
        self.register_buffer(
            "_selector_max_delta",
            torch.tensor(float(selector_max_delta)),
            persistent=False,
        )
        self.register_buffer(
            "_selector_apply",
            torch.tensor(float(bool(selector_enabled))),
            persistent=False,
        )

    def steered_layers(self):
        if self.canonical_block is not None:
            yield self.canonical_block.conv
        for block in self.upsample_blocks:
            yield block.conv
        for block in self.output_blocks:
            yield block.conv
        yield self.rgb_head
        yield self.alpha_head

    def set_effective_s_ref(self, value: float | None) -> None:
        for layer in self.steered_layers():
            layer.set_effective_s_ref(value)

    def set_selector_calibration(
        self,
        *,
        apply: bool | None = None,
        threshold: float | None = None,
        max_delta: float | None = None,
    ) -> None:
        if self.selector is None:
            raise RuntimeError("selector is not enabled")
        if apply is not None:
            self._selector_apply.fill_(float(bool(apply)))
        if threshold is not None:
            if not 0.0 < float(threshold) < 1.0:
                raise ValueError("selector threshold must lie in (0,1)")
            self._selector_threshold.fill_(float(threshold))
        if max_delta is not None:
            if float(max_delta) < 0.0:
                raise ValueError("selector max_delta must be non-negative")
            self._selector_max_delta.fill_(float(max_delta))

    def _spatial_trunk(
        self, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if slots.ndim != 3 or slots.shape[-1] != self.appearance_dim + 3:
            raise ValueError("slots must have shape (B,N,appearance_dim+3)")
        batch, num_slots, _ = slots.shape
        if num_slots != self.num_slots:
            raise ValueError("slot count does not match the decoder")
        appearance = slots[..., : self.appearance_dim]
        position = slots[..., self.appearance_dim : self.appearance_dim + 2]
        scale = slots[..., -1:]
        merged_scale = scale.reshape(batch * num_slots, 1)

        size = self.broadcast_size
        broadcast = appearance.reshape(
            batch * num_slots, self.appearance_dim, 1, 1
        ).expand(-1, -1, size, size)
        grid = _coordinate_grid(size, slots.device, slots.dtype)
        relative = (
            grid.view(1, 1, size, size, 2)
            - position[:, :, None, None]
        ) / self.scales_factor / (scale[:, :, None, None] + 1e-8)
        position_features = self.grid_projection(relative).permute(0, 1, 4, 2, 3)
        features = broadcast + position_features.reshape(
            batch * num_slots, self.appearance_dim, size, size
        )

        if self.canonical_block is not None:
            features = self.canonical_block(features, merged_scale)
        for block in self.upsample_blocks:
            features = F.interpolate(
                features,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=True,
            )
            features = block(features, merged_scale)
        for block in self.output_blocks:
            features = block(features, merged_scale)
        if features.shape[-2:] != (self.image_size, self.image_size):
            raise RuntimeError("decoder topology did not reach the output canvas")
        return features, merged_scale, batch, num_slots

    def _selector_fields(
        self,
        *,
        mask_logits: torch.Tensor,
        alpha: torch.Tensor,
        spatial_features: torch.Tensor,
        slot_rgb: torch.Tensor,
        attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.selector is None:
            raise RuntimeError("selector is not enabled")
        batch, num_slots, _, height, width = alpha.shape
        alpha_2d = alpha[:, :, 0].float().detach()
        logits_2d = mask_logits[:, :, 0].float().detach()
        border_width = min(
            self.selector_border_width,
            max(height // 2, 1),
            max(width // 2, 1),
        )
        border = torch.zeros(height, width, dtype=torch.bool, device=alpha.device)
        border[:border_width] = True
        border[-border_width:] = True
        border[:, :border_width] = True
        border[:, -border_width:] = True

        with torch.no_grad():
            bg_score = alpha_2d[..., border].mean(dim=-1)
            bg_score = bg_score + self.selector_bg_score_eta * alpha_2d.mean(
                dim=(-2, -1)
            )
            bg_index = bg_score.argmax(dim=1)
            owner = alpha_2d.argmax(dim=1)
            bg_pixel = bg_index[:, None, None].expand(-1, height, width)
            hard = F.one_hot(owner, num_slots).permute(0, 3, 1, 2).bool()
            active = (
                hard.sum(dim=(-2, -1)) > self.selector_owned_min
            ) & (
                alpha_2d.amax(dim=(-2, -1)) > self.selector_peak_min
            )
            active.scatter_(1, bg_index[:, None], False)
            owner_active = active.gather(
                1, owner.reshape(batch, -1)
            ).reshape(batch, height, width)
            action_eligible = owner_active & (owner != bg_pixel)

            slot_ids = torch.arange(
                num_slots, device=alpha.device
            ).view(1, num_slots, 1, 1)
            other = (slot_ids != owner[:, None]) & (slot_ids != bg_pixel[:, None])
            has_other = other.any(dim=1)
            max_other_logits = logits_2d.masked_fill(
                ~other, float("-inf")
            ).amax(dim=1)
            max_other_logits = torch.where(
                has_other, max_other_logits, torch.zeros_like(max_other_logits)
            )
            max_other_alpha = alpha_2d.masked_fill(~other, 0.0).amax(dim=1)
            owner_logits = _gather_slot(logits_2d, owner)
            bg_logits = _gather_slot(logits_2d, bg_pixel)
            owner_alpha = _gather_slot(alpha_2d, owner)
            bg_alpha = _gather_slot(alpha_2d, bg_pixel)
            other_gap = torch.where(
                has_other,
                max_other_logits - bg_logits,
                torch.zeros_like(bg_logits),
            )
            required_owner = 0.5 * (
                owner_logits - bg_logits + self.selector_takeover_margin
            )
            required_other = (
                max_other_logits - bg_logits + self.selector_takeover_margin
            )
            required_other = torch.where(
                has_other, required_other, torch.zeros_like(required_other)
            )
            delta_required = torch.maximum(
                torch.zeros_like(required_owner),
                torch.maximum(required_owner, required_other),
            )

            token_side = math.isqrt(attention.shape[-1])
            if (
                attention.ndim != 3
                or attention.shape[:2] != (batch, num_slots)
                or token_side * token_side != attention.shape[-1]
            ):
                raise ValueError("selector requires square-grid attention")
            attention_map = F.interpolate(
                attention.detach().float().reshape(
                    batch * num_slots, 1, token_side, token_side
                ),
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            ).reshape(batch, num_slots, height, width)
            attention_density = attention_map / attention_map.sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-8)
            alpha_density = alpha_2d / alpha_2d.sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1e-8)
            owner_alpha_density = _gather_slot(alpha_density, owner)
            owner_attention_density = _gather_slot(attention_density, owner)

        feature_map = spatial_features.detach().float().reshape(
            batch, num_slots, spatial_features.shape[1], height, width
        )
        owner_feature = _gather_slot(feature_map, owner)
        bg_feature = _gather_slot(feature_map, bg_pixel)
        owner_rgb = _gather_slot(slot_rgb.detach().float(), owner)
        bg_rgb = _gather_slot(slot_rgb.detach().float(), bg_pixel)
        raw_features = torch.cat(
            [
                owner_alpha[:, None],
                bg_alpha[:, None],
                max_other_alpha[:, None],
                (owner_logits - bg_logits)[:, None],
                other_gap[:, None],
                owner_alpha_density[:, None],
                owner_attention_density[:, None],
                (
                    owner_alpha_density.clamp_min(1e-8).log()
                    - owner_attention_density.clamp_min(1e-8).log()
                )[:, None],
                owner_feature,
                owner_feature - bg_feature,
                owner_rgb - bg_rgb,
            ],
            dim=1,
        ).permute(0, 2, 3, 1).contiguous()
        selector_logits = self.selector(raw_features.detach())
        return {
            "raw_features": raw_features.detach(),
            "logits": selector_logits[:, None],
            "probability": torch.sigmoid(selector_logits)[:, None],
            "owner": owner[:, None],
            "background_index": bg_index,
            "background_pixel": bg_pixel[:, None],
            "action_eligible": action_eligible[:, None],
            "delta_required": delta_required[:, None],
        }

    def forward(
        self,
        slots: torch.Tensor,
        *,
        attention: torch.Tensor | None = None,
        use_selector: bool = True,
    ) -> DecoderOutput:
        features, scales, batch, num_slots = self._spatial_trunk(slots)
        appearance = slots[..., : self.appearance_dim]
        style = self.rgb_style(appearance).reshape(
            batch * num_slots, features.shape[1], 1, 1
        )
        style = style.expand(-1, -1, self.image_size, self.image_size)
        rgb_features = F.relu(
            self.rgb_fusion(torch.cat([features, style], dim=1))
        )
        rgb_equivariant_logits = self.rgb_head(rgb_features, scales)
        rgb_logits = rgb_equivariant_logits
        if self.rgb_micro_head is not None:
            rgb_logits = rgb_logits + self.rgb_micro_eta_max * torch.tanh(
                self.rgb_micro_head(rgb_features)
            )
        slot_rgb = torch.sigmoid(rgb_logits).reshape(
            batch,
            num_slots,
            self.output_channels,
            self.image_size,
            self.image_size,
        )
        base_logits = self.alpha_head(features, scales).reshape(
            batch, num_slots, 1, self.image_size, self.image_size
        )
        base_alpha = F.softmax(base_logits, dim=1)
        base_reconstruction = (slot_rgb * base_alpha).sum(dim=1)

        selector_fields = None
        mask_logits = base_logits
        if self.selector is not None and use_selector:
            if attention is None:
                raise ValueError("attention is required when selector is enabled")
            selector_fields = self._selector_fields(
                mask_logits=base_logits,
                alpha=base_alpha,
                spatial_features=rgb_features,
                slot_rgb=torch.sigmoid(rgb_equivariant_logits).reshape(
                    batch,
                    num_slots,
                    self.output_channels,
                    self.image_size,
                    self.image_size,
                ),
                attention=attention,
            )
            selected = (
                selector_fields["probability"] >= self._selector_threshold
            ) & selector_fields["action_eligible"] & (
                selector_fields["delta_required"] <= self.selector_label_max_delta
            )
            delta = torch.minimum(
                selector_fields["delta_required"],
                self._selector_max_delta,
            )
            delta = delta * selected.float() * self._selector_apply
            correction = torch.zeros_like(base_logits[:, :, 0])
            correction.scatter_add_(
                1, selector_fields["owner"], -delta
            )
            correction.scatter_add_(
                1, selector_fields["background_pixel"], delta
            )
            mask_logits = base_logits + correction[:, :, None]
            selector_fields["selected"] = selected
            selector_fields["delta"] = delta

        alpha = F.softmax(mask_logits, dim=1)
        reconstruction = (slot_rgb * alpha).sum(dim=1)
        return DecoderOutput(
            reconstruction=reconstruction,
            reconstruction_pre_selector=base_reconstruction,
            slot_rgb=slot_rgb,
            alpha=alpha,
            alpha_pre_selector=base_alpha,
            mask_logits=mask_logits,
            mask_logits_pre_selector=base_logits,
            spatial_features=rgb_features.reshape(
                batch,
                num_slots,
                rgb_features.shape[1],
                self.image_size,
                self.image_size,
            ),
            selector=selector_fields,
        )
