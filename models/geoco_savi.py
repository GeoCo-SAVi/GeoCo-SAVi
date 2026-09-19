from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from .decoder import DecoderOutput, ScaleSteeredDecoder
from .encoder import CNNEncoder, FrozenDINOv2Encoder
from .slot_attention import InvariantSlotAttention
from .temporal import STATMResidualInitializer


@dataclass
class VideoOutput:


    reconstruction: torch.Tensor
    reconstruction_pre_selector: torch.Tensor
    slot_rgb: torch.Tensor
    alpha: torch.Tensor
    alpha_pre_selector: torch.Tensor
    mask_logits: torch.Tensor
    mask_logits_pre_selector: torch.Tensor
    slots: torch.Tensor
    attention: torch.Tensor
    ownership: torch.Tensor
    frames: tuple[DecoderOutput, ...]
    decoder_slots: tuple[torch.Tensor, ...] = ()


class GeoCoSAVi(nn.Module):


    def __init__(
        self,
        *,
        encoder: nn.Module,
        slot_attention: InvariantSlotAttention,
        decoder: ScaleSteeredDecoder,
        temporal_initializer: STATMResidualInitializer | None = None,
        detach_temporal_history: bool = True,
        iterations_first: int = 3,
        iterations_rest: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.slot_attention = slot_attention
        self.decoder = decoder
        self.temporal_initializer = temporal_initializer
        self.detach_temporal_history = bool(detach_temporal_history)
        self.iterations_first = int(iterations_first)
        self.iterations_rest = int(iterations_rest)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GeoCoSAVi":


        model = config["model"]
        encoder_config = model["encoder"]
        encoder_type = encoder_config["type"]
        if encoder_type == "cnn":
            encoder = CNNEncoder(
                in_channels=encoder_config.get("in_channels", 3),
                width=encoder_config.get("width", 64),
                out_dim=encoder_config["feature_dim"],
            )
        elif encoder_type == "dinov2":
            encoder = FrozenDINOv2Encoder(
                model_name=encoder_config["model_name"],
                input_size=encoder_config["input_size"],
                native_dim=encoder_config["native_dim"],
                out_dim=encoder_config["feature_dim"],
                checkpoint_path=encoder_config.get("checkpoint"),
                pretrained=encoder_config.get("pretrained", True),
            )
        else:
            raise ValueError("encoder type must be 'cnn' or 'dinov2'")

        slot_config = model["slot_attention"]
        slot_attention = InvariantSlotAttention(
            num_slots=model["num_slots"],
            appearance_dim=model["appearance_dim"],
            feature_dim=encoder_config["feature_dim"],
            qkv_dim=slot_config["qkv_dim"],
            grid_hidden_dim=slot_config["grid_hidden_dim"],
            update_hidden_dim=slot_config["update_hidden_dim"],
            iterations=slot_config["iterations_first"],
            scales_factor=slot_config["scales_factor"],
            min_scale=slot_config["min_scale"],
            max_scale=slot_config["max_scale"],
            terminal_readout=slot_config.get("terminal_readout", False),
            terminal_gate_max=slot_config.get("terminal_gate_max", 0.5),
            terminal_gate_mode=slot_config.get(
                "terminal_gate_mode", "channel"
            ),
        )

        decoder_config = model["decoder"]
        selector_config = model["teacher_selector"]
        decoder = ScaleSteeredDecoder(
            appearance_dim=model["appearance_dim"],
            num_slots=model["num_slots"],
            image_size=decoder_config["image_size"],
            broadcast_size=decoder_config["broadcast_size"],
            hidden_dim=decoder_config["hidden_dim"],
            output_channels=3,
            scales_factor=slot_config["scales_factor"],
            s_ref=decoder_config["s_ref"],
            gain_max=decoder_config["gain_max"],
            learn_gain=True,
            canonical_kernel=decoder_config.get("canonical_kernel"),
            upsample_kernels=decoder_config["upsample_kernels"],
            output_kernels=decoder_config["output_kernels"],
            rgb_kernel=decoder_config["rgb_kernel"],
            alpha_kernel=decoder_config["alpha_kernel"],
            rgb_micro=decoder_config.get("rgb_micro", False),
            rgb_micro_eta_max=decoder_config.get("rgb_micro_eta_max", 0.5),
            selector_enabled=selector_config.get("enabled", False),
            selector_threshold=selector_config.get("threshold", 0.5),
            selector_max_delta=selector_config.get("max_delta", 0.0),
            selector_label_max_delta=selector_config.get("label_max_delta", 2.0),
            selector_takeover_margin=selector_config.get(
                "takeover_margin", 1e-3
            ),
            selector_border_width=selector_config.get("border_width", 4),
            selector_owned_min=selector_config.get("owned_min", 20),
            selector_peak_min=selector_config.get("peak_min", 0.3),
            selector_bg_score_eta=selector_config.get("bg_score_eta", 0.25),
        )

        temporal_config = model["temporal"]
        temporal_initializer = None
        if temporal_config["enabled"]:
            temporal_initializer = STATMResidualInitializer(
                slot_dim=model["appearance_dim"] + 3,
                appearance_dim=model["appearance_dim"],
                model_dim=temporal_config["model_dim"],
                num_heads=temporal_config["num_heads"],
                mlp_dim=temporal_config["mlp_dim"],
                memory_size=temporal_config["memory_size"],
                min_scale=slot_config["min_scale"],
                max_scale=slot_config["max_scale"],
                dropout=temporal_config["dropout"],
            )
        return cls(
            encoder=encoder,
            slot_attention=slot_attention,
            decoder=decoder,
            temporal_initializer=temporal_initializer,
            detach_temporal_history=temporal_config.get("detach_history", True),
            iterations_first=slot_config["iterations_first"],
            iterations_rest=slot_config["iterations_rest"],
        )

    def forward(
        self,
        video: torch.Tensor,
        *,
        temporal_active: bool = True,
    ) -> VideoOutput:
        if video.ndim != 5:
            raise ValueError("video must have shape (B,T,C,H,W)")
        outputs: list[DecoderOutput] = []
        decoder_slots = []
        slots_by_time: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        ownerships: list[torch.Tensor] = []

        for time_index in range(video.shape[1]):
            features, grid = self.encoder(video[:, time_index])
            initial_slots = None
            if slots_by_time:
                previous = slots_by_time[-1]
                if temporal_active and self.temporal_initializer is not None:
                    history = torch.stack(slots_by_time, dim=1)
                    if self.detach_temporal_history:
                        previous = previous.detach()
                        history = history.detach()
                    initial_slots = self.temporal_initializer(previous, history)
                else:
                    initial_slots = previous
            slot_output = self.slot_attention(
                features,
                grid,
                initial_slots=initial_slots,
                iterations=(
                    self.iterations_first
                    if time_index == 0
                    else self.iterations_rest
                ),
            )
            rendering_slots = slot_output.slots.clone()
            decoded = self.decoder(
                rendering_slots,
                attention=slot_output.attention,
            )
            slots_by_time.append(slot_output.slots)
            attentions.append(slot_output.attention)
            ownerships.append(slot_output.ownership)
            outputs.append(decoded)
            decoder_slots.append(rendering_slots)

        return VideoOutput(
            reconstruction=torch.stack(
                [output.reconstruction for output in outputs], dim=1
            ),
            reconstruction_pre_selector=torch.stack(
                [output.reconstruction_pre_selector for output in outputs], dim=1
            ),
            slot_rgb=torch.stack(
                [output.slot_rgb for output in outputs], dim=1
            ),
            alpha=torch.stack([output.alpha for output in outputs], dim=1),
            alpha_pre_selector=torch.stack(
                [output.alpha_pre_selector for output in outputs], dim=1
            ),
            mask_logits=torch.stack(
                [output.mask_logits for output in outputs], dim=1
            ),
            mask_logits_pre_selector=torch.stack(
                [output.mask_logits_pre_selector for output in outputs], dim=1
            ),
            slots=torch.stack(slots_by_time, dim=1),
            attention=torch.stack(attentions, dim=1),
            ownership=torch.stack(ownerships, dim=1),
            frames=tuple(outputs),
            decoder_slots=tuple(decoder_slots),
        )
