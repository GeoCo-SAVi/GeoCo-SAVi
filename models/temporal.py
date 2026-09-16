from __future__ import annotations

import math

import torch
import torch.nn as nn


class STATMResidualInitializer(nn.Module):


    def __init__(
        self,
        *,
        slot_dim: int,
        appearance_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        mlp_dim: int = 512,
        memory_size: int = 4,
        min_scale: float = 1e-3,
        max_scale: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if slot_dim != appearance_dim + 3:
            raise ValueError("slot_dim must equal appearance_dim + 3")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 < float(min_scale) <= float(max_scale):
            raise ValueError("scale bounds must satisfy 0 < min_scale <= max_scale")
        self.slot_dim = int(slot_dim)
        self.appearance_dim = int(appearance_dim)
        self.model_dim = int(model_dim)
        self.memory_size = int(memory_size)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

        self.current_projection = nn.Linear(slot_dim, model_dim)
        self.memory_projection = nn.Linear(slot_dim, model_dim)
        self.spatial_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_attention = nn.MultiheadAttention(
            model_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.spatial_norm = nn.LayerNorm(model_dim)
        self.temporal_norm = nn.LayerNorm(model_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, model_dim),
        )
        self.fusion_norm = nn.LayerNorm(model_dim)
        self.residual_projection = nn.Linear(model_dim, slot_dim)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(
        self,
        current_slots: torch.Tensor,
        memory_slots: torch.Tensor,
    ) -> torch.Tensor:
        if current_slots.ndim != 3 or memory_slots.ndim != 4:
            raise ValueError("expected current (B,N,D) and memory (B,T,N,D)")
        if current_slots.shape[0] != memory_slots.shape[0]:
            raise ValueError("current and memory batch sizes differ")
        if current_slots.shape[1] != memory_slots.shape[2]:
            raise ValueError("current and memory slot counts differ")
        memory_slots = memory_slots[:, -self.memory_size :]
        batch, num_slots, _ = current_slots.shape

        query = self.current_projection(current_slots)
        spatial_update, _ = self.spatial_attention(
            query, query, query, need_weights=False
        )
        spatial = self.spatial_norm(query + spatial_update)

        memory = self.memory_projection(memory_slots)
        temporal_query = query.reshape(batch * num_slots, 1, self.model_dim)
        temporal_memory = memory.permute(0, 2, 1, 3).reshape(
            batch * num_slots, memory.shape[1], self.model_dim
        )
        temporal_update, _ = self.temporal_attention(
            temporal_query,
            temporal_memory,
            temporal_memory,
            need_weights=False,
        )
        temporal = self.temporal_norm(
            temporal_query + temporal_update
        ).reshape(batch, num_slots, self.model_dim)

        fused = spatial + temporal
        fused = self.fusion_norm(fused + self.fusion_mlp(fused))
        residual = self.residual_projection(fused)
        appearance = (
            current_slots[..., : self.appearance_dim]
            + residual[..., : self.appearance_dim]
        )
        position = (
            current_slots[..., self.appearance_dim : self.appearance_dim + 2]
            + residual[..., self.appearance_dim : self.appearance_dim + 2]
        ).clamp(-1.0, 1.0)
        log_scale = (
            current_slots[..., -1:]
            .clamp(self.min_scale, self.max_scale)
            .log()
            + residual[..., -1:]
        )
        scale = log_scale.clamp(
            math.log(self.min_scale),
            math.log(self.max_scale),
        ).exp()
        return torch.cat([appearance, position, scale], dim=-1)
