from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SelectorTeacherConfig:


    bg_score_eta: float = 0.25
    border_width: int = 4
    owned_min: int = 20
    peak_min: float = 0.3
    attention_ratio_low: float = 1.2
    attention_ratio_high: float = 1.8
    positive_gain_low: float = 0.0
    positive_gain_high: float = 0.05
    harmful_gain_max: float = -0.02
    individual_harmful_max: float = -0.05
    label_max_delta: float = 2.0
    local_bg_kernel: int = 9
    local_bg_quantile: float = 0.95
    local_bg_error_max: float = 5.0
    local_bg_delta_max: float = 2.0
    negative_weight: float = 2.0
    rank_weight: float = 0.25
    rank_temperature: float = 0.5
    rank_margin: float = 0.5


def _smoothstep(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    if not low < high:
        raise ValueError("smoothstep requires low < high")
    coordinate = ((value - low) / (high - low)).clamp(0.0, 1.0)
    return coordinate.square() * (3.0 - 2.0 * coordinate)


def _gather_slot(field: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if field.ndim == 4:
        return field.gather(1, index[:, None])[:, 0]
    if field.ndim == 5:
        expanded = index[:, None, None].expand(-1, 1, field.shape[2], -1, -1)
        return field.gather(1, expanded)[:, 0]
    raise ValueError("field must have shape (B,N,H,W) or (B,N,C,H,W)")


@torch.no_grad()
def build_selector_teacher(
    *,
    alpha: torch.Tensor,
    mask_logits: torch.Tensor,
    slot_rgb: torch.Tensor,
    reconstruction: torch.Tensor,
    target_rgb: torch.Tensor,
    attention: torch.Tensor,
    delta_required: torch.Tensor,
    config: SelectorTeacherConfig,
) -> dict[str, torch.Tensor]:


    alpha = alpha.detach().float()
    mask_logits = mask_logits.detach().float()
    slot_rgb = slot_rgb.detach().float()
    reconstruction = reconstruction.detach().float()
    target_rgb = target_rgb.detach().float()
    attention = attention.detach().float()
    delta_required = delta_required.detach().float()
    if alpha.ndim != 4 or mask_logits.shape != alpha.shape:
        raise ValueError("alpha and mask_logits must share shape (B,N,H,W)")
    if slot_rgb.ndim != 5 or reconstruction.shape != target_rgb.shape:
        raise ValueError("invalid RGB tensors for selector teacher")
    batch, num_slots, height, width = alpha.shape
    if delta_required.shape != (batch, 1, height, width):
        raise ValueError("delta_required must have shape (B,1,H,W)")
    if num_slots < 2:
        zero = alpha[:, :1] * 0.0
        return {
            "positive": zero,
            "protected": zero.bool(),
            "eligible": zero.bool(),
            "owner": zero.long(),
        }

    edge = min(config.border_width, max(height // 2, 1), max(width // 2, 1))
    border = torch.zeros(height, width, dtype=torch.bool, device=alpha.device)
    border[:edge] = True
    border[-edge:] = True
    border[:, :edge] = True
    border[:, -edge:] = True
    bg_score = alpha[..., border].mean(dim=-1)
    bg_score = bg_score + config.bg_score_eta * alpha.mean(dim=(-2, -1))
    bg_index = bg_score.argmax(dim=1)
    bg_pixel = bg_index[:, None, None].expand(-1, height, width)

    owner = alpha.argmax(dim=1)
    hard = F.one_hot(owner, num_slots).permute(0, 3, 1, 2).bool()
    owned = hard.sum(dim=(-2, -1))
    active = (
        (owned > config.owned_min)
        & (alpha.amax(dim=(-2, -1)) > config.peak_min)
    )
    active.scatter_(1, bg_index[:, None], False)
    owner_active = active.gather(1, owner.reshape(batch, -1)).reshape(
        batch, height, width
    )
    action_eligible = owner_active & (owner != bg_pixel)

    bg_rgb = _gather_slot(slot_rgb, bg_pixel)
    slot_error = (slot_rgb - target_rgb[:, None]).square().mean(dim=2)
    bg_error = (bg_rgb - target_rgb).square().mean(dim=1)
    individual_advantage = (slot_error - bg_error[:, None]) / (
        slot_error + bg_error[:, None] + 1e-6
    )
    owner_advantage = _gather_slot(individual_advantage, owner)

    residual = reconstruction - target_rgb
    current_error = residual.square().mean(dim=1)
    full_transfer = alpha[:, :, None] * (bg_rgb[:, None] - slot_rgb)
    full_error = (residual[:, None] + full_transfer).square().mean(dim=2)
    full_gain = (current_error[:, None] - full_error) / (
        current_error[:, None] + full_error + 1e-6
    )
    active_hard = hard & active[:, :, None, None]
    trusted_seed = (full_gain >= 0.0) & active_hard
    expanded = F.max_pool2d(
        trusted_seed.float(), kernel_size=3, stride=1, padding=1
    ).bool()

    token_side = math.isqrt(attention.shape[-1])
    if (
        attention.ndim != 3
        or attention.shape[:2] != (batch, num_slots)
        or token_side * token_side != attention.shape[-1]
    ):
        raise ValueError("attention must form a square spatial token grid")
    attention_map = F.interpolate(
        attention.reshape(batch * num_slots, 1, token_side, token_side),
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    ).reshape(batch, num_slots, height, width)
    attention_density = attention_map / attention_map.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    alpha_density = alpha / alpha.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    attention_safety = _smoothstep(
        alpha_density / attention_density.clamp_min(1e-8),
        config.attention_ratio_low,
        config.attention_ratio_high,
    )
    seed_confidence = torch.maximum(
        trusted_seed.float(),
        expanded.float() * attention_safety,
    ) * active_hard.float()
    candidate = _gather_slot(seed_confidence, owner) * action_eligible.float()

    valid_delta = delta_required[:, 0].clamp_min(0.0) <= config.label_max_delta
    eligible = action_eligible & valid_delta
    action_delta = delta_required[:, 0] * eligible.float()
    correction = torch.zeros_like(mask_logits)
    correction.scatter_add_(1, owner[:, None], -action_delta[:, None])
    correction.scatter_add_(1, bg_pixel[:, None], action_delta[:, None])
    action_alpha = F.softmax(mask_logits + correction, dim=1)
    action_reconstruction = (slot_rgb * action_alpha[:, :, None]).sum(dim=1)
    action_error = (action_reconstruction - target_rgb).square().mean(dim=1)
    action_gain = (current_error - action_error) / (
        current_error + action_error + 1e-6
    )
    action_confidence = _smoothstep(
        action_gain,
        config.positive_gain_low,
        config.positive_gain_high,
    )
    positive = candidate * action_confidence

    exact_harmful = eligible & (action_gain <= config.harmful_gain_max)
    individual_harmful = (
        eligible & (owner_advantage <= config.individual_harmful_max)
    )
    protected = exact_harmful | individual_harmful


    kernel = config.local_bg_kernel
    if kernel < 3 or kernel % 2 == 0:
        raise ValueError("local_bg_kernel must be an odd integer >= 3")
    padding = kernel // 2
    bg_owned = (owner == bg_pixel).float()[:, None]
    local_support = F.avg_pool2d(bg_owned, kernel, stride=1, padding=padding)
    local_rgb = F.avg_pool2d(
        target_rgb * bg_owned, kernel, stride=1, padding=padding
    ) / local_support.clamp_min(1e-8)
    local_error = (local_rgb - target_rgb).square().mean(dim=1)
    local_valid = local_support[:, 0] * float(kernel * kernel) >= 4.0
    normalized_local_error = torch.full_like(local_error, float("inf"))
    for scene in range(batch):
        reference_mask = (
            (owner[scene] == bg_index[scene]) & local_valid[scene]
        )
        reference = local_error[scene][reference_mask]
        if reference.numel() < 8:
            reference = local_error[scene][local_valid[scene]]
        if reference.numel() == 0:
            continue
        scale = torch.quantile(
            reference, config.local_bg_quantile
        ).clamp_min(1e-8)
        normalized_local_error[scene][local_valid[scene]] = (
            local_error[scene][local_valid[scene]] / scale
        )
    local_positive = (
        eligible
        & (delta_required[:, 0] <= config.local_bg_delta_max)
        & local_valid
        & (normalized_local_error <= config.local_bg_error_max)
        & ~protected
    )
    positive = torch.maximum(positive, local_positive.float())


    positive = positive.clamp(0.0, 1.0) * (~protected).float()
    return {
        "positive": positive[:, None].detach(),
        "protected": protected[:, None].detach(),
        "eligible": eligible[:, None].detach(),
        "owner": owner[:, None].detach(),
        "action_gain": action_gain[:, None].detach(),
        "local_positive": local_positive[:, None].detach(),
    }


def _scene_owner_macro_mean(
    numerator_map: torch.Tensor,
    weight_map: torch.Tensor,
    owner_index: torch.Tensor,
) -> torch.Tensor:
    batch = numerator_map.shape[0]
    owner = owner_index[:, 0].long()
    num_slots = int(owner.max().detach().item()) + 1
    scene = torch.arange(batch, device=owner.device)[:, None, None]
    group = (scene * num_slots + owner).reshape(-1)
    groups = batch * num_slots
    numerator = numerator_map.reshape(-1).new_zeros(groups)
    denominator = weight_map.reshape(-1).new_zeros(groups)
    numerator.scatter_add_(0, group, numerator_map.reshape(-1))
    denominator.scatter_add_(0, group, weight_map.reshape(-1))
    present = denominator > 0.0
    if not bool(present.any()):
        return numerator_map.sum() * 0.0
    return (numerator[present] / denominator[present].clamp_min(1e-8)).mean()


def _tail_ranking(
    logits: torch.Tensor,
    positive_weight: torch.Tensor,
    protected_weight: torch.Tensor,
    owner_index: torch.Tensor,
    *,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("rank_temperature must be positive")
    owner = owner_index[:, 0].long()
    count = int(owner.max().item()) + 1
    mask = owner[:, None] == torch.arange(count, device=owner.device)[None, :, None, None]
    values = logits[:, None].flatten(2)

    def tail(weight, sign):
        weights = weight[:, None].expand_as(mask).flatten(2)
        valid = mask.flatten(2) & (weights > 0)
        present = valid.any(-1)
        logs = torch.where(valid, weights.clamp_min(1e-8).log(), -torch.inf)
        logs = torch.where(present[..., None], logs, torch.zeros_like(logs))
        return (
            torch.logsumexp(logs + sign * values / temperature, -1)
            - torch.logsumexp(logs, -1),
            present,
        )

    positive, has_positive = tail(positive_weight, -1)
    protected, has_protected = tail(protected_weight, 1)
    present = has_positive & has_protected
    losses = F.softplus(margin / temperature + protected + positive) * temperature
    return torch.where(present, losses, 0).sum() / present.sum().clamp_min(1)


def selector_training_loss(
    selector_logits: torch.Tensor,
    labels: dict[str, torch.Tensor],
    config: SelectorTeacherConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:


    if selector_logits.ndim != 4 or selector_logits.shape[1] != 1:
        raise ValueError("selector_logits must have shape (B,1,H,W)")
    logits = selector_logits.float()[:, 0]
    positive = labels["positive"].to(logits.dtype)[:, 0]
    protected = labels["protected"].to(logits.dtype)[:, 0]
    owner = labels["owner"]
    positive_loss = _scene_owner_macro_mean(
        positive * F.softplus(-logits), positive, owner
    )
    protected_loss = _scene_owner_macro_mean(
        protected * F.softplus(logits), protected, owner
    )
    rank_loss = _tail_ranking(
        logits,
        positive,
        protected,
        owner,
        temperature=config.rank_temperature,
        margin=config.rank_margin,
    )
    total = (
        positive_loss
        + config.negative_weight * protected_loss
        + config.rank_weight * rank_loss
        + selector_logits.sum() * 0.0
    )
    with torch.no_grad():
        eligible = labels["eligible"].float()[:, 0]
        probability = torch.sigmoid(logits)
        metrics = {
            "selector_positive_loss": positive_loss.detach(),
            "selector_protected_loss": protected_loss.detach(),
            "selector_rank_loss": rank_loss.detach(),
            "selector_positive_fraction": positive.mean(),
            "selector_protected_fraction": protected.mean(),
            "selector_eligible_fraction": eligible.mean(),
            "selector_probability_eligible": (
                (probability * eligible).sum()
                / eligible.sum().clamp_min(1.0)
            ),
        }
    return total, metrics
