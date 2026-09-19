from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _mask_field(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 5 and value.shape[2] == 1:
        value = value[:, :, 0]
    if value.ndim != 4:
        raise ValueError(f"{name} must have shape (B,N,H,W) or (B,N,1,H,W)")
    return value


def hard_slot_ownership(alpha: torch.Tensor) -> torch.Tensor:


    alpha = _mask_field(alpha, "alpha")
    owner = alpha.argmax(dim=1)
    return F.one_hot(owner, alpha.shape[1]).permute(0, 3, 1, 2).to(alpha.dtype)


def onefg_margin(
    mask_logits: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    allow_missing_background: bool = False,
    background_source: torch.Tensor | None = None,
) -> torch.Tensor:


    logits = _mask_field(mask_logits, "mask_logits")
    source = logits if background_source is None else _mask_field(
        background_source, "background_source"
    )
    if source.shape[0] != logits.shape[0] or source.shape[-2:] != logits.shape[-2:]:
        raise ValueError("background_source must match batch and spatial dimensions")
    if background_mask.shape != source.shape[:2]:
        raise ValueError("background_mask must have shape (B,N)")
    background_mask = background_mask.to(device=logits.device, dtype=torch.bool)
    has_background = background_mask.any(dim=1)
    if not allow_missing_background and bool((~has_background).any()):
        raise ValueError("every scene requires at least one factual background slot")
    negative = torch.finfo(source.dtype).min
    background_logits = torch.logsumexp(
        source.masked_fill(~background_mask[..., None, None], negative),
        dim=1,
        keepdim=True,
    )
    if allow_missing_background:
        background_logits = torch.where(
            has_background[:, None, None, None],
            background_logits,
            torch.zeros_like(background_logits),
        )
    return logits - background_logits


def onefg_support(
    mask_logits: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    allow_missing_background: bool = False,
    background_source: torch.Tensor | None = None,
) -> torch.Tensor:
    margin = onefg_margin(
        mask_logits,
        background_mask,
        allow_missing_background=allow_missing_background,
        background_source=background_source,
    )
    return torch.sigmoid(float(gamma) * margin)


def support_moments(
    support: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:


    if support.ndim < 2:
        raise ValueError("support must end in (H,W)")
    height, width = support.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=support.device, dtype=support.dtype),
        torch.linspace(-1.0, 1.0, width, device=support.device, dtype=support.dtype),
        indexing="ij",
    )
    mass = support.sum(dim=(-2, -1))
    centroid_x = (support * xx).sum(dim=(-2, -1)) / (mass + epsilon)
    centroid_y = (support * yy).sum(dim=(-2, -1)) / (mass + epsilon)
    centroid = torch.stack([centroid_x, centroid_y], dim=-1)
    distance_squared = (
        xx - centroid_x[..., None, None]
    ).square() + (
        yy - centroid_y[..., None, None]
    ).square()
    radius = torch.sqrt(
        (support * distance_squared).sum(dim=(-2, -1)) / (mass + epsilon)
        + epsilon
    )
    coverage = mass / float(height * width)
    compactness = coverage / (radius.square() + epsilon)
    return {
        "mass": mass,
        "centroid": centroid,
        "radius": radius,
        "coverage": coverage,
        "compactness": compactness,
    }


@torch.no_grad()
def factual_slot_quality(
    slots: torch.Tensor,
    alpha: torch.Tensor,
    attention: torch.Tensor,
    *,
    appearance_dim: int,
    owned_min: int,
    owned_max_fraction: float,
    alpha_peak_min: float,
    scale_max: float,
    effective_pixels_min: float,
    effective_pixels_max_fraction: float,
    max_attention_cosine: float,
    boundary_z: float,
    boundary_margin_pixels: float,
) -> dict[str, torch.Tensor]:


    alpha = _mask_field(alpha, "alpha")
    batch, num_slots, height, width = alpha.shape
    hard = hard_slot_ownership(alpha)
    owned = hard.sum(dim=(-2, -1))
    peak = alpha.amax(dim=(-2, -1))
    scale = slots[..., appearance_dim + 2]
    foreground = (
        (owned > float(owned_min))
        & (owned < float(owned_max_fraction * height * width))
        & (peak > float(alpha_peak_min))
        & (scale < float(scale_max))
    )
    background = ~foreground
    has_background = background.any(dim=1, keepdim=True)

    normalized_attention = attention / attention.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    effective_pixels = normalized_attention.square().sum(
        dim=-1
    ).clamp_min(1e-8).reciprocal()
    effective_ok = (
        (effective_pixels >= float(effective_pixels_min))
        & (
            effective_pixels
            <= float(effective_pixels_max_fraction * attention.shape[-1])
        )
    )

    side = int(round(attention.shape[-1] ** 0.5))
    if side * side != attention.shape[-1]:
        raise ValueError("attention tokens must form a square grid")
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, side, device=attention.device),
        torch.linspace(-1.0, 1.0, side, device=attention.device),
        indexing="ij",
    )
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=-1).to(attention.dtype)
    position = slots[..., appearance_dim : appearance_dim + 2]
    centered = grid.view(1, 1, -1, 2) - position[:, :, None]
    covariance = torch.einsum(
        "bnm,bnmi,bnmj->bnij",
        normalized_attention,
        centered,
        centered,
    )
    margin = float(boundary_margin_pixels) * 2.0 / max(height - 1, 1)
    boundary = (
        position[..., 0].abs()
        + float(boundary_z) * covariance[..., 0, 0].clamp_min(0.0).sqrt()
        < 1.0 - margin
    ) & (
        position[..., 1].abs()
        + float(boundary_z) * covariance[..., 1, 1].clamp_min(0.0).sqrt()
        < 1.0 - margin
    )

    unit_attention = normalized_attention / normalized_attention.square().sum(
        dim=-1, keepdim=True
    ).sqrt().clamp_min(1e-8)
    cosine = torch.bmm(unit_attention, unit_attention.transpose(1, 2))
    diagonal = torch.eye(num_slots, device=alpha.device, dtype=torch.bool)[None]
    max_other_cosine = cosine.masked_fill(diagonal, -1.0).amax(dim=-1)
    isolated = max_other_cosine < float(max_attention_cosine)
    valid = foreground & has_background & effective_ok & boundary & isolated
    confidence = (
        valid.float()
        * (1.0 - max_other_cosine.clamp(0.0, 1.0))
        * (peak / max(float(alpha_peak_min), 1e-8)).clamp(max=1.0)
    )
    return {
        "foreground": foreground.detach(),
        "background": background.detach(),
        "valid": valid.detach(),
        "confidence": confidence.detach(),
        "owned": owned.detach(),
        "peak": peak.detach(),
        "effective_pixels": effective_pixels.detach(),
        "max_attention_cosine": max_other_cosine.detach(),
    }


@torch.no_grad()
def geometry_coherence_filter(
    quality: dict[str, torch.Tensor],
    mask_logits: torch.Tensor,
    alpha: torch.Tensor,
    slots: torch.Tensor,
    *,
    appearance_dim: int,
    gamma: float,
    compactness_min: float | None,
    position_residual_max: float | None,
    effective_pixels_min: float | None,
    owner_iou_min: float | None,
    max_attention_cosine: float | None,
) -> torch.Tensor:


    if all(
        threshold is None
        for threshold in (
            compactness_min,
            position_residual_max,
            effective_pixels_min,
            owner_iou_min,
            max_attention_cosine,
        )
    ):
        return quality["valid"]

    logits = _mask_field(mask_logits.detach(), "mask_logits")
    alpha_field = _mask_field(alpha.detach(), "alpha")
    support = onefg_support(
        logits,
        quality["background"],
        gamma=float(gamma),
        allow_missing_background=True,
    )
    moments = support_moments(support)
    result = quality["valid"].clone()
    if compactness_min is not None:
        result &= moments["compactness"] >= float(compactness_min)
    if position_residual_max is not None:
        position = slots.detach()[
            ..., appearance_dim : appearance_dim + 2
        ]
        residual = (moments["centroid"] - position).norm(dim=-1)
        result &= residual <= float(position_residual_max)
    if effective_pixels_min is not None:
        result &= quality["effective_pixels"] >= float(effective_pixels_min)
    if owner_iou_min is not None:
        owner = hard_slot_ownership(alpha_field).bool()
        onefg_hard = support >= 0.5
        union = (onefg_hard | owner).sum(dim=(-2, -1)).clamp_min(1)
        owner_iou = (onefg_hard & owner).sum(dim=(-2, -1)) / union
        result &= owner_iou >= float(owner_iou_min)
    if max_attention_cosine is not None:
        result &= quality["max_attention_cosine"] <= float(
            max_attention_cosine
        )
    return result.detach()


@dataclass
class TransplantPairs:
    recipient_batch: torch.Tensor
    recipient_slot: torch.Tensor
    donor_batch: torch.Tensor
    donor_slot: torch.Tensor
    weight: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.recipient_batch.numel())

    def to(self, device) -> "TransplantPairs":
        return TransplantPairs(
            self.recipient_batch.to(device),
            self.recipient_slot.to(device),
            self.donor_batch.to(device),
            self.donor_slot.to(device),
            self.weight.to(device),
        )


@torch.no_grad()
def select_transplant_pairs(
    valid: torch.Tensor,
    confidence: torch.Tensor,
    scales: torch.Tensor,
    *,
    source_ids: torch.Tensor | None = None,
    min_scale_ratio: float = 0.6,
    max_scale_ratio: float = 1.67,
) -> TransplantPairs:


    if (
        valid.shape != confidence.shape
        or scales.shape != valid.shape
        or valid.ndim != 2
    ):
        raise ValueError("valid, confidence, and scales must have shape (B,N)")
    if not 0.0 < min_scale_ratio <= max_scale_ratio:
        raise ValueError("invalid donor/recipient scale-ratio interval")
    empty = torch.empty(0, dtype=torch.long, device=valid.device)
    if source_ids is None:
        source_ids = torch.arange(valid.shape[0], device=valid.device)
    source_ids = source_ids.reshape(-1)
    if source_ids.numel() != valid.shape[0]:
        raise ValueError("source_ids must contain one value per scene")
    if torch.unique(source_ids).numel() < 2:
        return TransplantPairs(empty, empty, empty, empty, empty.float())

    recipient_batches = []
    recipient_slots = []
    donor_batches = []
    donor_slots = []
    weights = []
    valid_cpu = valid.detach().cpu()
    scales_cpu = scales.detach().cpu()
    source_cpu = source_ids.detach().cpu()
    slot_lists = [row.nonzero(as_tuple=False).flatten() for row in valid_cpu]
    source_masks = [valid_cpu & (source_cpu[:, None] != source) for source in source_cpu]
    for recipient_batch in range(valid.shape[0]):
        candidates = slot_lists[recipient_batch]
        if candidates.numel() == 0:
            continue
        candidates = candidates[
            torch.randperm(candidates.numel(), device=valid.device).cpu()
        ]
        chosen = None
        for recipient_slot in candidates.tolist():
            recipient_scale = scales_cpu[
                recipient_batch, recipient_slot
            ].detach()
            ratio = scales_cpu / recipient_scale.clamp_min(1e-8)
            donor_candidate = source_masks[recipient_batch] & (
                (ratio >= float(min_scale_ratio))
                & (ratio <= float(max_scale_ratio))
            )
            donors = donor_candidate.nonzero(as_tuple=False)
            if donors.numel() == 0:
                continue
            donor = donors[
                torch.randint(
                    donors.shape[0],
                    (1,),
                    device=valid.device,
                ).item()
            ]
            chosen = (recipient_slot, int(donor[0]), int(donor[1]))
            break
        if chosen is None:
            continue
        recipient_slot, donor_batch, donor_slot = chosen
        recipient_batches.append(recipient_batch)
        recipient_slots.append(recipient_slot)
        donor_batches.append(donor_batch)
        donor_slots.append(donor_slot)
        weights.append(
            confidence[recipient_batch, recipient_slot]
            * confidence[donor_batch, donor_slot]
        )
    if not recipient_batches:
        return TransplantPairs(empty, empty, empty, empty, empty.float())
    return TransplantPairs(
        torch.tensor(recipient_batches, device=valid.device),
        torch.tensor(recipient_slots, device=valid.device),
        torch.tensor(donor_batches, device=valid.device),
        torch.tensor(donor_slots, device=valid.device),
        torch.stack(weights).detach(),
    )


def geometry_loss(
    *,
    factual_support: torch.Tensor,
    counterfactual_support: torch.Tensor,
    factual_slots: torch.Tensor,
    pairs: TransplantPairs,
    appearance_dim: int,
    compactness_weight: float = 0.25,
    huber_delta: float = 0.05,
    center_metric: str = "coordinate_huber",
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:


    if pairs.count == 0:
        zero = counterfactual_support.sum() * 0.0
        return zero, {
            "center": zero.detach(),
            "radius": zero.detach(),
            "compactness": zero.detach(),
            "pairs": zero.detach(),
        }
    pairs = pairs.to(counterfactual_support.device)
    recipient_support = factual_support[
        pairs.recipient_batch, pairs.recipient_slot
    ].detach()
    donor_support = factual_support[
        pairs.donor_batch, pairs.donor_slot
    ].detach()
    edited_support = counterfactual_support
    if edited_support.ndim == 4:
        edited_support = edited_support[
            torch.arange(pairs.count, device=counterfactual_support.device),
            pairs.recipient_slot,
        ]
    recipient_moments = support_moments(recipient_support)
    donor_moments = support_moments(donor_support)
    edited_moments = support_moments(edited_support)
    recipient_position = factual_slots[
        pairs.recipient_batch,
        pairs.recipient_slot,
        appearance_dim : appearance_dim + 2,
    ].detach()
    weights = pairs.weight.detach().float().clamp_min(0.0)

    center_error = edited_moments["centroid"] - recipient_position
    radius_error = (
        (edited_moments["radius"] + epsilon).log()
        - (recipient_moments["radius"] + epsilon).log()
    )
    compactness_error = (
        (edited_moments["compactness"] + epsilon).log()
        - (donor_moments["compactness"] + epsilon).log()
    )

    def weighted_huber(error: torch.Tensor) -> torch.Tensor:
        per_element = F.huber_loss(
            error,
            torch.zeros_like(error),
            delta=float(huber_delta),
            reduction="none",
        )
        if per_element.ndim > 1:
            per_element = per_element.mean(dim=-1)
        return (per_element * weights).sum() / weights.sum().clamp_min(epsilon)

    if center_metric == "scale_normalized_norm_huber":
        recipient_scale = factual_slots[
            pairs.recipient_batch, pairs.recipient_slot, appearance_dim + 2
        ].detach()
        center_error = center_error.norm(dim=-1) / (recipient_scale + epsilon)
    elif center_metric != "coordinate_huber":
        raise ValueError("unknown center metric")
    center_loss = weighted_huber(center_error)
    radius_loss = weighted_huber(radius_error)
    compactness_loss = weighted_huber(compactness_error)
    total = center_loss + radius_loss + float(compactness_weight) * compactness_loss
    return total, {
        "center": center_loss.detach(),
        "radius": radius_loss.detach(),
        "compactness": compactness_loss.detach(),
        "pairs": torch.tensor(
            float(pairs.count), device=total.device, dtype=total.dtype
        ),
    }
