from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SlotAttentionOutput:


    slots: torch.Tensor
    attention: torch.Tensor
    ownership: torch.Tensor


class InvariantSlotAttention(nn.Module):


    def __init__(
        self,
        *,
        num_slots: int,
        appearance_dim: int,
        feature_dim: int,
        qkv_dim: int = 64,
        grid_hidden_dim: int = 128,
        update_hidden_dim: int = 256,
        iterations: int = 3,
        scales_factor: float = 5.0,
        min_scale: float = 1e-3,
        max_scale: float = 2.0,
        epsilon: float = 1e-8,
        terminal_readout: bool = False,
        terminal_gate_max: float = 0.5,
        terminal_gate_mode: str = "channel",
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.appearance_dim = int(appearance_dim)
        self.slot_dim = self.appearance_dim + 3
        self.qkv_dim = int(qkv_dim)
        self.iterations = int(iterations)
        self.scales_factor = float(scales_factor)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.epsilon = float(epsilon)
        self.terminal_readout = bool(terminal_readout)
        self.terminal_gate_max = float(terminal_gate_max)
        self.terminal_gate_mode = str(terminal_gate_mode)
        self.terminal_gate: torch.Tensor | None = None
        if self.terminal_gate_max <= 0.0:
            raise ValueError("terminal_gate_max must be positive")
        if self.terminal_gate_mode not in {"scalar", "channel"}:
            raise ValueError("terminal_gate_mode must be scalar or channel")

        self.input_norm = nn.LayerNorm(feature_dim)
        self.slot_norm = nn.LayerNorm(self.appearance_dim)
        self.key = nn.Linear(feature_dim, self.qkv_dim, bias=False)
        self.value = nn.Linear(feature_dim, self.qkv_dim, bias=False)
        self.query = nn.Linear(self.appearance_dim, self.qkv_dim, bias=False)
        self.grid_projection = nn.Linear(2, self.qkv_dim)
        self.grid_encoder = nn.Sequential(
            nn.LayerNorm(self.qkv_dim),
            nn.Linear(self.qkv_dim, grid_hidden_dim),
            nn.ReLU(),
            nn.Linear(grid_hidden_dim, self.qkv_dim),
        )
        self.gru = nn.GRUCell(self.qkv_dim, self.appearance_dim)
        self.update_mlp = nn.Sequential(
            nn.LayerNorm(self.appearance_dim),
            nn.Linear(self.appearance_dim, update_hidden_dim),
            nn.ReLU(),
            nn.Linear(update_hidden_dim, self.appearance_dim),
        )
        self.initial_appearance = nn.Parameter(
            torch.randn(1, self.num_slots, self.appearance_dim)
        )
        self.initial_scale = nn.Parameter(torch.ones(1, self.num_slots, 1))
        if self.terminal_readout:
            self.terminal_gate_norm = nn.LayerNorm(
                2 * self.appearance_dim,
                elementwise_affine=False,
            )
            gate_dim = (
                1
                if self.terminal_gate_mode == "scalar"
                else self.appearance_dim
            )
            self.terminal_gate_projection = nn.Linear(
                2 * self.appearance_dim, gate_dim
            )
            nn.init.zeros_(self.terminal_gate_projection.weight)
            nn.init.zeros_(self.terminal_gate_projection.bias)

    def _update_appearance(
        self,
        appearance: torch.Tensor,
        updates: torch.Tensor,
    ) -> torch.Tensor:
        batch = appearance.shape[0]
        with torch.autocast(
            device_type=appearance.device.type,
            enabled=False,
        ):
            updated = self.gru(
                updates.reshape(-1, self.qkv_dim).float(),
                appearance.reshape(-1, self.appearance_dim).float(),
            )
        updated = updated.reshape(batch, self.num_slots, self.appearance_dim)
        return updated + self.update_mlp(updated)

    def _initial_slots(self, batch: int, device) -> tuple[torch.Tensor, ...]:
        appearance = self.initial_appearance.expand(batch, -1, -1)


        position = torch.empty(batch, self.num_slots, 2, device=device).uniform_(-1.0, 1.0)
        scale = self.initial_scale.expand(batch, -1, -1).clone()
        return appearance, position, scale

    def forward(
        self,
        features: torch.Tensor,
        grid: torch.Tensor,
        *,
        initial_slots: torch.Tensor | None = None,
        iterations: int | None = None,
    ) -> SlotAttentionOutput:
        if features.ndim != 3 or grid.shape != (*features.shape[:2], 2):
            raise ValueError("features/grid must have shapes (B,L,D) and (B,L,2)")
        batch, token_count, _ = features.shape
        if initial_slots is None:
            appearance, position, scale = self._initial_slots(batch, features.device)
        else:
            if initial_slots.shape != (batch, self.num_slots, self.slot_dim):
                raise ValueError("initial_slots has an incompatible shape")
            appearance = initial_slots[..., : self.appearance_dim]
            position = initial_slots[..., self.appearance_dim : self.appearance_dim + 2]
            scale = initial_slots[..., -1:]
        scale = scale.clamp(self.min_scale, self.max_scale)

        inputs = self.input_norm(features)
        key = self.key(inputs)[:, None].expand(-1, self.num_slots, -1, -1)
        value = self.value(inputs)[:, None].expand(-1, self.num_slots, -1, -1)
        expanded_grid = grid[:, None].expand(-1, self.num_slots, -1, -1)
        rounds = self.iterations if iterations is None else int(iterations)

        for round_index in range(rounds + 1):
            relative = (
                (expanded_grid - position[:, :, None])
                / self.scales_factor
                / (scale[:, :, None] + self.epsilon)
            )
            grid_embedding = self.grid_projection(relative)
            relative_key = self.grid_encoder(key + grid_embedding)
            relative_value = self.grid_encoder(value + grid_embedding)
            query = self.query(self.slot_norm(appearance)) / math.sqrt(self.qkv_dim)
            logits = torch.einsum("bnd,bnld->bnl", query, relative_key)


            ownership = F.softmax(logits, dim=1)
            attention = ownership / (
                ownership.sum(dim=-1, keepdim=True) + self.epsilon
            )
            updates = torch.einsum("bnl,bnld->bnd", attention, relative_value)
            position = torch.einsum("bnl,bld->bnd", attention, grid)
            squared_distance = (
                expanded_grid - position[:, :, None]
            ).square().sum(dim=-1)
            scale = torch.sqrt(
                torch.einsum("bnl,bnl->bn", attention + self.epsilon, squared_distance)
            )[..., None].clamp(self.min_scale, self.max_scale)


            if round_index < rounds:
                appearance = self._update_appearance(appearance, updates)

        if self.terminal_readout:
            terminal = self._update_appearance(appearance, updates)
            delta = terminal - appearance
            gate_features = self.terminal_gate_norm(
                torch.cat([appearance, delta], dim=-1)
            )
            gate = self.terminal_gate_max * torch.tanh(
                self.terminal_gate_projection(gate_features)
            )
            appearance = appearance + gate * delta
            self.terminal_gate = gate
        else:
            self.terminal_gate = None

        slots = torch.cat([appearance, position, scale], dim=-1)
        return SlotAttentionOutput(slots, attention, ownership)
