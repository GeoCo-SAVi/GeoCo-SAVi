from __future__ import annotations

import math
from typing import Iterable

import torch


def psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:


    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must share a batched shape")
    mse = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
    value = 10.0 * torch.log10(float(data_range) ** 2 / mse.clamp_min(1e-12))
    return value.mean()


def _batched_labels(labels: torch.Tensor) -> torch.Tensor:
    labels = torch.as_tensor(labels).long()
    if labels.ndim == 2:
        labels = labels[None]
    if labels.ndim < 3:
        raise ValueError("label maps must be (H,W) or (B,...,H,W)")
    return labels.reshape(labels.shape[0], -1)


def _ari_one(
    target: torch.Tensor,
    prediction: torch.Tensor,
    ignore_label: int | None,
) -> torch.Tensor:
    if ignore_label is not None:
        keep = target != int(ignore_label)
        target = target[keep]
        prediction = prediction[keep]
    count = target.numel()
    if count < 2:
        return torch.ones((), device=target.device, dtype=torch.float64)
    _, target_inverse = torch.unique(target, return_inverse=True)
    _, prediction_inverse = torch.unique(prediction, return_inverse=True)
    prediction_count = int(prediction_inverse.max().item()) + 1
    contingency = torch.bincount(
        target_inverse * prediction_count + prediction_inverse
    ).double()
    target_marginal = torch.bincount(target_inverse).double()
    prediction_marginal = torch.bincount(prediction_inverse).double()
    choose2 = lambda value: value * (value - 1.0) * 0.5
    index = choose2(contingency).sum()
    target_index = choose2(target_marginal).sum()
    prediction_index = choose2(prediction_marginal).sum()
    total_pairs = choose2(torch.tensor(float(count), device=target.device))
    expected = target_index * prediction_index / total_pairs.clamp_min(1.0)
    maximum = 0.5 * (target_index + prediction_index)
    denominator = maximum - expected
    if float(denominator.abs()) < 1e-12:
        return torch.ones_like(denominator)
    return (index - expected) / denominator


def adjusted_rand_index(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    ignore_label: int | None = None,
) -> torch.Tensor:


    target = _batched_labels(target_labels)
    prediction = _batched_labels(predicted_labels)
    if target.shape != prediction.shape:
        raise ValueError("target and predicted label maps must match")
    return torch.stack(
        [
            _ari_one(target[index], prediction[index], ignore_label)
            for index in range(target.shape[0])
        ]
    ).mean().float()


def fg_ari(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    background_label: int = 0,
) -> torch.Tensor:


    return adjusted_rand_index(
        target_labels,
        predicted_labels,
        ignore_label=background_label,
    )


def ari(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
) -> torch.Tensor:


    return adjusted_rand_index(target_labels, predicted_labels)


def _instance_iou_matrix(
    target: torch.Tensor,
    prediction: torch.Tensor,
    background_label: int,
) -> torch.Tensor:
    target_ids = torch.unique(target)
    target_ids = target_ids[target_ids != int(background_label)]
    prediction_ids = torch.unique(prediction)
    if target_ids.numel() == 0 or prediction_ids.numel() == 0:
        return torch.empty(
            target_ids.numel(),
            prediction_ids.numel(),
            device=target.device,
            dtype=torch.float32,
        )
    target_masks = target[None] == target_ids[:, None]
    prediction_masks = prediction[None] == prediction_ids[:, None]
    intersection = (
        target_masks[:, None] & prediction_masks[None]
    ).sum(dim=-1).float()
    union = (
        target_masks[:, None] | prediction_masks[None]
    ).sum(dim=-1).float()
    return intersection / union.clamp_min(1.0)


def _valid_mean(values: list[torch.Tensor], device) -> torch.Tensor:
    if not values:
        return torch.full((), float("nan"), device=device)
    return torch.stack(values).mean()


def hungarian_miou(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    background_label: int = 0,
) -> torch.Tensor:


    from scipy.optimize import linear_sum_assignment

    target = _batched_labels(target_labels)
    prediction = _batched_labels(predicted_labels)
    if target.shape != prediction.shape:
        raise ValueError("target and predicted label maps must match")
    scores = []
    for index in range(target.shape[0]):
        matrix = _instance_iou_matrix(
            target[index], prediction[index], background_label
        )
        if matrix.shape[0] == 0:
            continue
        if matrix.shape[1] == 0:
            scores.append(matrix.new_zeros(()))
            continue
        rows, columns = linear_sum_assignment(
            -matrix.detach().cpu().numpy()
        )
        matched = matrix[
            torch.as_tensor(rows, device=matrix.device),
            torch.as_tensor(columns, device=matrix.device),
        ].sum()
        scores.append(matched / matrix.shape[0])
    return _valid_mean(scores, target.device)


def mean_best_overlap(
    target_labels: torch.Tensor,
    predicted_labels: torch.Tensor,
    *,
    background_label: int = 0,
) -> torch.Tensor:


    target = _batched_labels(target_labels)
    prediction = _batched_labels(predicted_labels)
    if target.shape != prediction.shape:
        raise ValueError("target and predicted label maps must match")
    scores = []
    for index in range(target.shape[0]):
        matrix = _instance_iou_matrix(
            target[index], prediction[index], background_label
        )
        if matrix.shape[0] == 0:
            continue
        if matrix.shape[1] == 0:
            scores.append(matrix.new_zeros(()))
        else:
            scores.append(matrix.amax(dim=1).mean())
    return _valid_mean(scores, target.device)


def hard_labels(alpha: torch.Tensor) -> torch.Tensor:


    if alpha.ndim >= 5 and alpha.shape[-3] == 1:
        alpha = alpha.squeeze(-3)
    if alpha.ndim < 4:
        raise ValueError("alpha must have shape (...,N,H,W)")
    return alpha.argmax(dim=-3)


def binary_f_score(
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:


    prediction = torch.as_tensor(predicted_masks)
    target = torch.as_tensor(target_masks, device=prediction.device)
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("predicted and target masks must share (...,H,W)")
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum(dim=(-2, -1)).float()
    denominator = (
        prediction.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    ).float()
    score = 2.0 * intersection / denominator
    return score.mean()


def tube_f_score(
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.as_tensor(predicted_masks)
    target = torch.as_tensor(target_masks, device=prediction.device)
    if prediction.shape != target.shape or prediction.ndim < 3:
        raise ValueError("predicted and target masks must share (...,T,H,W)")
    prediction = prediction.bool()
    target = target.bool()
    intersection = (prediction & target).sum(dim=(-3, -2, -1)).float()
    denominator = (
        prediction.sum(dim=(-3, -2, -1))
        + target.sum(dim=(-3, -2, -1))
    ).float()
    score = 2.0 * intersection / denominator
    return score.mean()


def hard_mask_geometry(
    masks: torch.Tensor,
    *,
    normalized: bool = True,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:


    masks = torch.as_tensor(masks).float()
    if masks.ndim < 2:
        raise ValueError("masks must end in (H,W)")
    height, width = masks.shape[-2:]
    if normalized:
        y = torch.linspace(-1.0, 1.0, height, device=masks.device)
        x = torch.linspace(-1.0, 1.0, width, device=masks.device)
    else:
        y = torch.arange(height, device=masks.device, dtype=masks.dtype)
        x = torch.arange(width, device=masks.device, dtype=masks.dtype)
    yy, xx = torch.meshgrid(
        y.to(masks.dtype), x.to(masks.dtype), indexing="ij"
    )
    mass = masks.sum(dim=(-2, -1))
    valid = mass > 0.0
    centroid_x = (masks * xx).sum(dim=(-2, -1)) / mass.clamp_min(epsilon)
    centroid_y = (masks * yy).sum(dim=(-2, -1)) / mass.clamp_min(epsilon)
    centroid = torch.stack([centroid_x, centroid_y], dim=-1)
    distance = (
        xx - centroid_x[..., None, None]
    ).square() + (
        yy - centroid_y[..., None, None]
    ).square()
    radius = torch.sqrt(
        (masks * distance).sum(dim=(-2, -1)) / mass.clamp_min(epsilon)
    )
    coverage = mass / float(height * width)
    return {
        "centroid": torch.where(valid[..., None], centroid, float("nan")),
        "radius": radius,
        "coverage": coverage,
        "mass": mass,
        "valid": valid,
    }


def position_centroid_error(
    position: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:


    geometry = hard_mask_geometry(masks, normalized=True)
    position = torch.as_tensor(position, device=masks.device).float()
    if position.shape != geometry["centroid"].shape:
        raise ValueError("position and mask leading dimensions do not match")
    error = (position - geometry["centroid"]).norm(dim=-1)
    return error[geometry["valid"]].mean()


def position_reference_error(
    position: torch.Tensor,
    reference_centroid: torch.Tensor,
) -> torch.Tensor:


    position = torch.as_tensor(position).float()
    reference = torch.as_tensor(
        reference_centroid, device=position.device
    ).float()
    if position.shape != reference.shape or position.shape[-1] != 2:
        raise ValueError("position and reference centroid must share (...,2)")
    valid = torch.isfinite(position).all(dim=-1) & torch.isfinite(
        reference
    ).all(dim=-1)
    return (position - reference).norm(dim=-1)[valid].mean()


def decoded_centroid_reference_error(
    masks: torch.Tensor,
    reference_centroid: torch.Tensor,
) -> torch.Tensor:


    geometry = hard_mask_geometry(masks, normalized=True)
    reference = torch.as_tensor(
        reference_centroid, device=geometry["centroid"].device
    ).float()
    if geometry["centroid"].shape != reference.shape:
        raise ValueError("mask leading dimensions and reference must match")
    valid = geometry["valid"] & torch.isfinite(reference).all(dim=-1)
    return (geometry["centroid"] - reference).norm(dim=-1)[valid].mean()


def attention_overlap(
    attention: torch.Tensor,
    *,
    active_slots: torch.Tensor | None = None,
) -> torch.Tensor:


    attention = torch.as_tensor(attention).float()
    if attention.ndim < 3:
        raise ValueError("attention must end in (slots,tokens)")
    slots, tokens = attention.shape[-2:]
    flat = attention.reshape(-1, slots, tokens)
    flat = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    if active_slots is None:
        active = torch.ones(
            flat.shape[0], slots, dtype=torch.bool, device=flat.device
        )
    else:
        active = torch.as_tensor(
            active_slots, device=flat.device, dtype=torch.bool
        ).reshape(-1, slots)
        if active.shape[0] != flat.shape[0]:
            raise ValueError("active_slots prefix dimensions do not match")
    values = []
    pairwise = torch.bmm(flat, flat.transpose(1, 2))
    diagonal = torch.eye(slots, device=flat.device, dtype=torch.bool)
    for index in range(flat.shape[0]):
        pair_mask = (
            active[index, :, None] & active[index, None, :] & ~diagonal
        )
        if bool(pair_mask.any()):
            values.append(pairwise[index][pair_mask].mean())
    return _valid_mean(values, flat.device)


def translation_error(
    factual_masks: torch.Tensor,
    edited_masks: torch.Tensor,
    delta_pixels: torch.Tensor,
    *,
    command_canvas_size: int = 64,
) -> torch.Tensor:


    command_canvas_size = int(command_canvas_size)
    if command_canvas_size < 2:
        raise ValueError("command_canvas_size must be at least 2")
    factual = hard_mask_geometry(factual_masks, normalized=True)
    edited = hard_mask_geometry(edited_masks, normalized=True)
    delta = torch.as_tensor(delta_pixels, device=factual_masks.device).float()
    if delta.shape != factual["centroid"].shape:
        raise ValueError("delta_pixels must match mask leading dimensions")
    observed_pixels = (
        edited["centroid"] - factual["centroid"]
    ) * ((command_canvas_size - 1) / 2.0)
    residual = observed_pixels - delta
    valid = factual["valid"] & edited["valid"]
    diagonal = math.sqrt(2.0 * command_canvas_size * command_canvas_size)
    return residual.norm(dim=-1)[valid].mean() / diagonal


def appearance_transplant_centroid_drift(
    factual_masks: torch.Tensor,
    transplanted_masks: torch.Tensor,
) -> torch.Tensor:


    factual = hard_mask_geometry(factual_masks, normalized=True)
    edited = hard_mask_geometry(transplanted_masks, normalized=True)
    valid = factual["valid"] & edited["valid"]
    return (
        edited["centroid"] - factual["centroid"]
    ).norm(dim=-1)[valid].mean()


def fixed_scale_consistency(
    masks: torch.Tensor,
    *,
    population_dimension: int | None = None,
) -> dict[str, torch.Tensor]:


    masks = torch.as_tensor(masks)
    if masks.ndim < 3:
        raise ValueError("masks require a population axis before (H,W)")
    geometry = hard_mask_geometry(masks, normalized=True)
    if population_dimension is None:
        radii = geometry["radius"].reshape(-1)
        radii = radii[torch.isfinite(radii)]
        coverage = geometry["coverage"].reshape(-1)
        return {
            "radius_std": radii.std(unbiased=False) if radii.numel() else
                torch.full((), float("nan"), device=masks.device),
            "coverage_std": coverage.std(unbiased=False) if coverage.numel() else
                torch.full((), float("nan"), device=masks.device),
        }
    dimension = (
        population_dimension
        if population_dimension >= 0
        else masks.ndim + population_dimension
    )
    if dimension < 0 or dimension >= masks.ndim - 2:
        raise ValueError("population_dimension must select a leading axis")
    radii = geometry["radius"]
    finite = torch.isfinite(radii)
    count = finite.sum(dim=dimension, keepdim=True)
    mean = radii.masked_fill(~finite, 0).sum(
        dim=dimension, keepdim=True
    ) / count.clamp_min(1)
    variance = (radii - mean).masked_fill(~finite, 0).square().sum(
        dim=dimension
    ) / count.squeeze(dimension).clamp_min(1)
    radius_std = variance.sqrt().masked_fill(
        count.squeeze(dimension) == 0, float("nan")
    )
    coverage_std = geometry["coverage"].std(dim=dimension, unbiased=False)
    return {
        "radius_std": torch.nanmean(radius_std),
        "coverage_std": torch.nanmean(coverage_std),
    }


def fixed_s_radius_std(masks: torch.Tensor) -> torch.Tensor:
    return fixed_scale_consistency(masks)["radius_std"]


def fixed_s_coverage_std(masks: torch.Tensor) -> torch.Tensor:
    return fixed_scale_consistency(masks)["coverage_std"]


def scale_sweep_metrics(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> dict[str, torch.Tensor]:


    masks = torch.as_tensor(masks).float()
    if masks.ndim != 4:
        raise ValueError("masks must have shape (groups,scales,H,W)")
    factors = torch.as_tensor(
        list(scale_factors) if not isinstance(scale_factors, torch.Tensor) else scale_factors,
        device=masks.device,
        dtype=masks.dtype,
    )
    if factors.ndim != 1 or factors.numel() != masks.shape[1]:
        raise ValueError("one scale factor is required per sweep mask")
    if factors.numel() < 2:
        raise ValueError("at least two scale factors are required")
    if bool((factors <= 0).any()) or bool((factors[1:] <= factors[:-1]).any()):
        raise ValueError("scale factors must be positive and strictly increasing")
    geometry = hard_mask_geometry(masks, normalized=True)
    radius = geometry["radius"]
    coverage = geometry["coverage"]
    x = factors.log()
    centered_x = x - x.mean()
    denominator = centered_x.square().sum().clamp_min(1e-12)

    def slope(values: torch.Tensor) -> torch.Tensor:
        y = values.clamp_min(1e-12).log()
        return (
            (y - y.mean(dim=1, keepdim=True)) * centered_x
        ).sum(dim=1) / denominator

    radius_valid = (
        geometry["valid"].all(dim=1)
        & torch.isfinite(radius).all(dim=1)
        & (radius > 0).all(dim=1)
    )
    coverage_positive = (
        torch.isfinite(coverage).all(dim=1)
        & (coverage > 0).all(dim=1)
    )
    radius_slopes = slope(radius)
    coverage_slopes = slope(coverage)
    return {
        "radius_slope": radius_slopes[radius_valid].mean(),
        "coverage_slope": coverage_slopes[coverage_positive].mean(),
    }


def radius_slope(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    return scale_sweep_metrics(masks, scale_factors)["radius_slope"]


def coverage_slope(
    masks: torch.Tensor,
    scale_factors: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    return scale_sweep_metrics(masks, scale_factors)["coverage_slope"]


def _scale_radius_inputs(
    scale: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.as_tensor(scale).to(dtype=torch.float64)
    radius = torch.as_tensor(radius, device=scale.device).to(dtype=torch.float64)
    if scale.shape != radius.shape or scale.numel() == 0:
        raise ValueError("scale and radius must share a nonempty shape")
    if not bool(torch.isfinite(scale).all() & torch.isfinite(radius).all()):
        raise ValueError("scale and radius must be finite")
    return scale.reshape(-1), radius.reshape(-1)


def scale_radius_linear_r2(
    scale: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    scale, radius = _scale_radius_inputs(scale, radius)
    x = scale - scale.mean()
    y = radius - radius.mean()
    variance = x.square().sum()
    slope = (x * y).sum() / torch.where(
        variance > 0, variance, torch.ones_like(variance)
    )
    return 1.0 - (y - slope * x).square().sum() / y.square().sum()


def _log_radius_scale_ratio(
    scale: torch.Tensor,
    radius: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale, radius = _scale_radius_inputs(scale, radius)
    if bool((scale <= 0).any() | (radius <= 0).any()):
        raise ValueError("scale and radius must be positive for logarithms")
    log_radius = radius.log()
    return log_radius, log_radius - scale.log()


def scale_radius_fixed_log_r2(
    scale: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    log_radius, log_ratio = _log_radius_scale_ratio(scale, radius)
    residual = log_ratio - log_ratio.mean()
    total = (log_radius - log_radius.mean()).square().sum()
    return 1.0 - residual.square().sum() / total.masked_fill(
        total == 0, float("nan")
    )


def log_radius_scale_std(
    scale: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    _, log_ratio = _log_radius_scale_ratio(scale, radius)
    return log_ratio.std(unbiased=False)
