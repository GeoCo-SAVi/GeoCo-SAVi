from __future__ import annotations

import torch
import torch.nn.functional as F

from .geometry import onefg_support, support_moments


def tail_penalty(
    margin: torch.Tensor,
    weights: torch.Tensor,
    *,
    threshold: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    if not 0.0 < threshold < 1.0:
        raise ValueError("tail threshold must lie between zero and one")
    scaled = float(gamma) * margin.float()
    selected = torch.sigmoid(scaled.detach()) < float(threshold)
    per_object = (F.softplus(scaled) * selected).mean(dim=(-2, -1))
    weights = weights.detach().to(per_object)
    if weights.shape != per_object.shape:
        raise ValueError("tail weights must match the object dimensions")
    return (per_object * weights).sum() / weights.sum().clamp_min(1e-8)


def reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:


    return F.mse_loss(reconstruction, target)


def normalized_attention_overlap(attention: torch.Tensor) -> torch.Tensor:


    if attention.ndim != 3:
        raise ValueError("attention must have shape (B,N,L)")
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    pairwise = torch.bmm(attention, attention.transpose(1, 2))
    num_slots = attention.shape[1]
    diagonal = torch.eye(
        num_slots, device=attention.device, dtype=torch.bool
    )[None]
    return pairwise.masked_select(~diagonal).mean()


def position_alignment_loss(
    *,
    slots: torch.Tensor,
    mask_logits: torch.Tensor,
    background_mask: torch.Tensor,
    valid: torch.Tensor,
    appearance_dim: int,
    max_support_coverage: float | None,
    onefg_gamma: float = 2.0,
    huber_delta: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:


    support = onefg_support(
        mask_logits,
        background_mask,
        gamma=float(onefg_gamma),
        allow_missing_background=True,
    )
    moments = support_moments(support)
    selected = valid.detach()
    if max_support_coverage is not None:
        selected = selected & (
            moments["coverage"].detach() <= float(max_support_coverage)
        )
    if not bool(selected.any()):
        zero = support.sum() * 0.0
        return zero, {"valid": zero.detach(), "mean_error": zero.detach()}


    command = slots[
        ..., appearance_dim : appearance_dim + 2
    ].detach()
    error = moments["centroid"] - command
    loss = F.huber_loss(
        error[selected],
        torch.zeros_like(error[selected]),
        delta=float(huber_delta),
        reduction="mean",
    )
    return loss, {
        "valid": selected.sum().detach(),
        "mean_error": error[selected].norm(dim=-1).mean().detach(),
    }
