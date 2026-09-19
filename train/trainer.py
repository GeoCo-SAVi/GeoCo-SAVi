from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Mapping

import torch
import torch.nn as nn

from models.geoco_savi import GeoCoSAVi, VideoOutput

from .curriculum import CurriculumState, GeoCoCurriculum
from .geometry import (
    factual_slot_quality,
    geometry_coherence_filter,
    geometry_loss,
    onefg_margin,
    onefg_support,
    select_transplant_pairs,
)
from .losses import (
    normalized_attention_overlap,
    position_alignment_loss,
    reconstruction_loss,
    tail_penalty,
)
from .selector import (
    SelectorTeacherConfig,
    build_selector_teacher,
    selector_training_loss,
)


def _compile_neutral_key(key: str) -> str:


    return ".".join(
        segment for segment in str(key).split(".") if segment != "_orig_mod"
    )


def _compile_neutral_state_dict(module: nn.Module) -> OrderedDict[str, Any]:


    source = module.state_dict()
    result: OrderedDict[str, Any] = OrderedDict()
    for key, value in source.items():
        neutral = _compile_neutral_key(key)
        if neutral in result:
            raise RuntimeError(
                f"state-dict key collision after compile normalization: {neutral}"
            )
        result[neutral] = value
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        result._metadata = OrderedDict(
            (_compile_neutral_key(key), value)
            for key, value in metadata.items()
        )
    return result


def _state_dict_for_model(
    module: nn.Module,
    checkpoint_state: Mapping[str, Any],
) -> OrderedDict[str, Any]:


    expected = module.state_dict()
    expected_by_neutral: dict[str, str] = {}
    for key in expected:
        neutral = _compile_neutral_key(key)
        if neutral in expected_by_neutral:
            raise RuntimeError(f"current state-dict key collision: {neutral}")
        expected_by_neutral[neutral] = key

    result: OrderedDict[str, Any] = OrderedDict()
    for key, value in checkpoint_state.items():
        neutral = _compile_neutral_key(key)
        target = expected_by_neutral.get(neutral, neutral)
        if target in result:
            raise RuntimeError(f"checkpoint state-dict key collision: {neutral}")
        result[target] = value

    metadata = getattr(checkpoint_state, "_metadata", None)
    if metadata is not None:
        expected_metadata = getattr(expected, "_metadata", {})
        metadata_by_neutral = {
            _compile_neutral_key(key): key for key in expected_metadata
        }
        result._metadata = OrderedDict(
            (
                metadata_by_neutral.get(
                    _compile_neutral_key(key),
                    _compile_neutral_key(key),
                ),
                value,
            )
            for key, value in metadata.items()
        )
    return result


class GeoCoTrainer:


    def __init__(
        self,
        model: GeoCoSAVi,
        config: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.curriculum = GeoCoCurriculum(config)
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type != "cuda":
            raise ValueError("model training requires a CUDA device")
        self.model.to(self.device)
        training = config["training"]
        self.base_lr = float(training["learning_rate"])
        self.max_grad_norm = float(training["max_grad_norm"])
        self.amp_enabled = bool(training.get("mixed_precision", True)) and (
            self.device.type == "cuda"
        )
        amp_name = str(training.get("amp_dtype", "bfloat16"))
        self.amp_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[amp_name]

        selector_parameters = (
            list(self.model.decoder.selector.parameters())
            if self.model.decoder.selector is not None
            else []
        )
        selector_ids = {id(parameter) for parameter in selector_parameters}
        temporal_parameters = (
            list(self.model.temporal_initializer.parameters())
            if self.model.temporal_initializer is not None
            else []
        )
        temporal_ids = {id(parameter) for parameter in temporal_parameters}
        core_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
            and id(parameter) not in selector_ids
            and id(parameter) not in temporal_ids
        ]
        groups: list[dict[str, Any]] = [
            {
                "name": "core",
                "params": core_parameters,
                "lr": self.base_lr,
            }
        ]
        if temporal_parameters:
            groups.append(
                {
                    "name": "temporal",
                    "params": temporal_parameters,
                    "lr": 0.0,
                }
            )
        self.core_optimizer = torch.optim.AdamW(
            groups,
            lr=self.base_lr,
            weight_decay=float(training.get("weight_decay", 0.0)),
        )
        self.selector_optimizer = None
        if selector_parameters:
            selector_config = config["model"]["teacher_selector"]
            self.selector_optimizer = torch.optim.AdamW(
                selector_parameters,
                lr=float(selector_config["learning_rate"]),
                weight_decay=float(selector_config["weight_decay"]),
            )
        if selector_ids.intersection(
            id(parameter)
            for group in self.core_optimizer.param_groups
            for parameter in group["params"]
        ):
            raise RuntimeError("selector and core optimizer parameters overlap")
        self.core_parameters = [
            parameter
            for group in self.core_optimizer.param_groups
            for parameter in group["params"]
        ]
        self.selector_parameters = selector_parameters
        try:
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=self.amp_enabled
            )
        except (AttributeError, TypeError):


            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.amp_enabled
            )
        teacher_fields = config["model"]["teacher_selector"].get("teacher", {})
        self.teacher_config = SelectorTeacherConfig(**teacher_fields)

    def _set_controls(self, state: CurriculumState) -> None:
        self.model.decoder.set_effective_s_ref(state.effective_s_ref)
        if self.model.decoder.selector is not None:
            selector_config = self.config["model"]["teacher_selector"]
            self.model.decoder.set_selector_calibration(
                apply=state.selector_apply,
                threshold=float(selector_config["threshold"]),
                max_delta=state.selector_max_delta,
            )
        for group in self.core_optimizer.param_groups:
            if group["name"] == "temporal":
                group["lr"] = self.base_lr * state.temporal_lr_multiplier
            else:
                group["lr"] = self.base_lr * state.core_lr_multiplier
        if self.selector_optimizer is not None:
            selector = self.config["model"]["teacher_selector"]
            start = int(self.config["curriculum"]["selector_train_start_step"])
            warmup = int(selector["warmup_steps"])
            progress = 0.0
            if state.selector_train:
                progress = (
                    1.0
                    if warmup <= 0
                    else min((state.step - start + 1) / float(warmup), 1.0)
                )
            for group in self.selector_optimizer.param_groups:
                group["lr"] = float(selector["learning_rate"]) * progress

    def _quality(
        self,
        slots: torch.Tensor,
        alpha: torch.Tensor,
        attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        quality = self.config["quality_filter"]
        return factual_slot_quality(
            slots,
            alpha,
            attention,
            appearance_dim=self.model.slot_attention.appearance_dim,
            owned_min=int(quality["owned_min"]),
            owned_max_fraction=float(quality["owned_max_fraction"]),
            alpha_peak_min=float(quality["alpha_peak_min"]),
            scale_max=float(quality["scale_max"]),
            effective_pixels_min=float(quality["effective_pixels_min"]),
            effective_pixels_max_fraction=float(
                quality["effective_pixels_max_fraction"]
            ),
            max_attention_cosine=float(quality["max_attention_cosine"]),
            boundary_z=float(quality["boundary_z"]),
            boundary_margin_pixels=float(quality["boundary_margin_pixels"]),
        )

    @staticmethod
    def _flatten_video(output: VideoOutput) -> tuple[torch.Tensor, ...]:
        batch, frames, slots, slot_dim = output.slots.shape
        return (
            output.slots.reshape(batch * frames, slots, slot_dim),
            output.alpha.reshape(
                batch * frames, slots, *output.alpha.shape[-3:]
            ),
            output.mask_logits.reshape(
                batch * frames, slots, *output.mask_logits.shape[-3:]
            ),
            output.attention.reshape(
                batch * frames, slots, output.attention.shape[-1]
            ),
        )

    def _position_objective(
        self,
        output: VideoOutput,
        *,
        include_tail: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slots, alpha, logits, attention = self._flatten_video(output)
        quality = self._quality(slots, alpha, attention)
        appearance_dim = self.model.slot_attention.appearance_dim


        loss, metrics = position_alignment_loss(
            slots=slots,
            mask_logits=logits,
            background_mask=quality["background"],
            valid=quality["foreground"],
            appearance_dim=appearance_dim,
            max_support_coverage=self.config["loss"].get(
                "position_max_support_coverage"
            ),
            onefg_gamma=float(self.config["loss"]["onefg_gamma"]),
            huber_delta=float(self.config["loss"]["huber_delta"]),
        )
        if include_tail:
            margin = onefg_margin(
                logits.float(),
                quality["background"],
                allow_missing_background=True,
            )
            metrics["tail"] = tail_penalty(
                margin,
                quality["foreground"] & quality["background"].any(dim=1, keepdim=True),
                threshold=float(self.config["loss"].get("tail_threshold", 0.25)),
                gamma=float(self.config["loss"]["onefg_gamma"]),
            )
        return loss, metrics

    def _geometry_objective(
        self,
        output: VideoOutput,
        source_ids: torch.Tensor,
        *,
        center_metric: str = "coordinate_huber",
        include_tail: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, frame_count, num_slots, slot_dim = output.slots.shape
        indices = [
            int(index)
            for index in self.config["curriculum"]["geometry_time_indices"]
            if int(index) < frame_count
        ]
        if not indices:
            indices = [0]
        slots = output.slots[:, indices].reshape(
            batch * len(indices), num_slots, slot_dim
        )
        alpha = output.alpha[:, indices].reshape(
            batch * len(indices), num_slots, *output.alpha.shape[-3:]
        )
        logits = output.mask_logits[:, indices].reshape(
            batch * len(indices), num_slots, *output.mask_logits.shape[-3:]
        )
        attention = output.attention[:, indices].reshape(
            batch * len(indices), num_slots, output.attention.shape[-1]
        )
        quality = self._quality(slots, alpha, attention)
        coherence = self.config["quality_filter"].get(
            "geometry_coherence", {}
        )
        geometry_valid = geometry_coherence_filter(
            quality,
            logits,
            alpha,
            slots,
            appearance_dim=self.model.slot_attention.appearance_dim,
            gamma=float(self.config["loss"]["onefg_gamma"]),
            compactness_min=coherence.get("compactness_min"),
            position_residual_max=coherence.get(
                "position_residual_max"
            ),
            effective_pixels_min=coherence.get("effective_pixels_min"),
            owner_iou_min=coherence.get("owner_iou_min"),
            max_attention_cosine=coherence.get(
                "max_attention_cosine"
            ),
        )
        expanded_sources = source_ids[:, None].expand(
            -1, len(indices)
        ).reshape(-1)
        pairs = select_transplant_pairs(
            geometry_valid,
            quality["confidence"],
            slots[..., -1],
            source_ids=expanded_sources,
            min_scale_ratio=float(
                self.config["loss"]["min_scale_ratio"]
            ),
            max_scale_ratio=float(
                self.config["loss"]["max_scale_ratio"]
            ),
        )
        factual_support = onefg_support(
            logits,
            quality["background"],
            gamma=float(self.config["loss"]["onefg_gamma"]),
            allow_missing_background=True,
        )
        if pairs.count == 0:
            zero = factual_support.sum() * 0.0
            return zero, {
                "geometry_center": zero.detach(),
                "geometry_radius": zero.detach(),
                "geometry_compactness": zero.detach(),
                "geometry_pairs": zero.detach(),
                "tail": zero,
            }
        appearance_dim = self.model.slot_attention.appearance_dim
        target_slots = torch.cat(
            [
                slots[pairs.donor_batch, pairs.donor_slot, :appearance_dim],
                slots[pairs.recipient_batch, pairs.recipient_slot, appearance_dim:].detach(),
            ],
            dim=-1,
        ).unsqueeze(1)
        target_logits = self.model.decoder.forward_alpha_logits(target_slots)
        counterfactual_support = onefg_support(
            target_logits,
            quality["background"][pairs.recipient_batch],
            gamma=float(self.config["loss"]["onefg_gamma"]),
            allow_missing_background=True,
            background_source=logits[pairs.recipient_batch],
        )
        loss, parts = geometry_loss(
            factual_support=factual_support,
            counterfactual_support=counterfactual_support[:, 0],
            factual_slots=slots,
            pairs=pairs,
            appearance_dim=self.model.slot_attention.appearance_dim,
            compactness_weight=float(
                self.config["loss"]["compactness_weight"]
            ),
            huber_delta=float(self.config["loss"]["huber_delta"]),
            center_metric=center_metric,
        )
        tail = loss * 0.0
        if include_tail:
            margin = onefg_margin(
                target_logits.float(),
                quality["background"][pairs.recipient_batch],
                allow_missing_background=True,
                background_source=logits[pairs.recipient_batch].float(),
            )
            target_margin = margin[:, 0]
            tail = tail_penalty(
                target_margin,
                pairs.weight,
                threshold=float(self.config["loss"].get("tail_threshold", 0.25)),
                gamma=float(self.config["loss"]["onefg_gamma"]),
            )
        return loss, {
            "geometry_center": parts["center"],
            "geometry_radius": parts["radius"],
            "geometry_compactness": parts["compactness"],
            "geometry_pairs": parts["pairs"],
            "tail": tail,
        }

    def _selector_objective(
        self,
        output: VideoOutput,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = []
        metrics_by_frame = []
        for time_index, decoded in enumerate(output.frames):
            if decoded.selector is None:
                raise RuntimeError("selector output is unavailable")
            labels = build_selector_teacher(
                alpha=decoded.alpha_pre_selector[:, :, 0],
                mask_logits=decoded.mask_logits_pre_selector[:, :, 0],
                slot_rgb=decoded.slot_rgb,
                reconstruction=decoded.reconstruction_pre_selector,
                target_rgb=target[:, time_index],
                attention=output.attention[:, time_index],
                delta_required=decoded.selector["delta_required"],
                config=self.teacher_config,
            )
            loss, metrics = selector_training_loss(
                decoded.selector["logits"],
                labels,
                self.teacher_config,
            )
            losses.append(loss)
            metrics_by_frame.append(metrics)
        return (
            torch.stack(losses).mean(),
            {
                name: torch.stack([item[name] for item in metrics_by_frame]).mean()
                for name in metrics_by_frame[0]
            },
        )

    def train_step(
        self,
        batch: Mapping[str, Any] | torch.Tensor,
        step: int,
    ) -> dict[str, float]:


        state = self.curriculum.at(step)
        self._set_controls(state)
        if isinstance(batch, torch.Tensor):
            video = batch
            source_ids = None
        else:
            video = batch["video"]
            source_ids = batch.get("source_id")
        if video.ndim != 5 or video.shape[1] < state.frame_count:
            raise ValueError("batch video does not contain the required frames")
        video = video[:, : state.frame_count].to(self.device, non_blocking=True)
        batch_size = video.shape[0]
        if source_ids is None:
            source_ids = torch.arange(batch_size, device=self.device)
        elif not isinstance(source_ids, torch.Tensor):
            source_ids = torch.as_tensor(source_ids)
        source_ids = source_ids.to(self.device).reshape(-1)
        if source_ids.numel() != batch_size:
            raise ValueError("source_id must contain one value per video")

        self.core_optimizer.zero_grad(set_to_none=True)
        if self.selector_optimizer is not None:
            self.selector_optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=True,
            )
            if self.amp_enabled
            else nullcontext()
        )
        with autocast:
            output = self.model(
                video,
                temporal_active=state.temporal_active,
            )
            reconstruction = reconstruction_loss(
                output.reconstruction,
                video,
            )
            attention = output.attention.reshape(
                -1,
                output.attention.shape[-2],
                output.attention.shape[-1],
            )
            overlap = normalized_attention_overlap(attention)
            zero = reconstruction * 0.0
            if state.position_weight > 0.0 or state.tail_weight > 0.0:
                position, position_metrics = self._position_objective(
                    output, include_tail=state.tail_weight > 0.0
                )
            else:
                position = zero
                position_metrics = {"valid": zero.detach()}
            factual_tail = position_metrics.pop("tail", zero)
            if state.geometry_weight > 0.0 or state.tail_weight > 0.0:
                geometry, geometry_metrics = self._geometry_objective(
                    output, source_ids,
                    center_metric=state.center_metric,
                    include_tail=state.tail_weight > 0.0,
                )
            else:
                geometry = zero
                geometry_metrics = {
                    "geometry_center": zero.detach(),
                    "geometry_radius": zero.detach(),
                    "geometry_compactness": zero.detach(),
                    "geometry_pairs": zero.detach(),
                }
            counterfactual_tail = geometry_metrics.pop("tail", zero)
            tail = 0.5 * (factual_tail + counterfactual_tail)
            terminal_gate = self.model.slot_attention.terminal_gate
            gate_regularization = (
                terminal_gate.square().mean()
                if terminal_gate is not None
                else zero
            )
            routed_loss = (
                state.position_weight * position
                + 0.5 * state.tail_weight * factual_tail
            )
            base_loss = (
                state.reconstruction_weight * reconstruction
                + state.overlap_weight * overlap
                + state.geometry_weight * geometry
                + 0.5 * state.tail_weight * counterfactual_tail
                + float(self.config["loss"]["terminal_gate_weight"])
                * gate_regularization
            )
            core_loss = base_loss + routed_loss
            selector_loss = None
            selector_metrics: dict[str, torch.Tensor] = {}
            if state.selector_train:
                selector_loss, selector_metrics = self._selector_objective(
                    output, video
                )

        has_routed_loss = state.position_weight > 0.0 or state.tail_weight > 0.0
        self.scaler.scale(base_loss).backward(retain_graph=has_routed_loss)
        if has_routed_loss:
            appearance_dim = self.model.slot_attention.appearance_dim

            def block_commands(gradient):
                return torch.cat(
                    [gradient[..., :appearance_dim], torch.zeros_like(gradient[..., appearance_dim:])],
                    dim=-1,
                )

            hooks = [slots.register_hook(block_commands) for slots in output.decoder_slots]
            try:
                self.scaler.scale(routed_loss).backward()
            finally:
                for hook in hooks:
                    hook.remove()
        if selector_loss is not None:
            self.scaler.scale(selector_loss).backward()
        self.scaler.unscale_(self.core_optimizer)
        nn.utils.clip_grad_norm_(self.core_parameters, self.max_grad_norm)
        if selector_loss is not None and self.selector_optimizer is not None:
            self.scaler.unscale_(self.selector_optimizer)
            nn.utils.clip_grad_norm_(
                self.selector_parameters,
                float(
                    self.config["model"]["teacher_selector"]["max_grad_norm"]
                ),
            )
        self.scaler.step(self.core_optimizer)
        if selector_loss is not None and self.selector_optimizer is not None:
            self.scaler.step(self.selector_optimizer)
        self.scaler.update()

        values: dict[str, torch.Tensor | float] = {
            "loss": core_loss.detach(),
            "reconstruction": reconstruction.detach(),
            "overlap": overlap.detach(),
            "position": position.detach(),
            "geometry": geometry.detach(),
            "tail": tail.detach(),
            "tail_factual": factual_tail.detach(),
            "tail_counterfactual": counterfactual_tail.detach(),
            "terminal_gate": gate_regularization.detach(),
            "position_valid": position_metrics["valid"],
            **geometry_metrics,
            **selector_metrics,
            "selector_loss": (
                selector_loss.detach()
                if selector_loss is not None
                else torch.zeros((), device=self.device)
            ),
            "effective_s_ref": state.effective_s_ref,
            "core_lr": self.core_optimizer.param_groups[0]["lr"],
        }
        return {
            name: (
                float(value.detach().float().cpu())
                if isinstance(value, torch.Tensor)
                else float(value)
            )
            for name, value in values.items()
        }

    def checkpoint(self, step: int) -> dict[str, Any]:


        return {
            "step": int(step),
            "model": _compile_neutral_state_dict(self.model),
            "core_optimizer": self.core_optimizer.state_dict(),
            "selector_optimizer": (
                None
                if self.selector_optimizer is None
                else self.selector_optimizer.state_dict()
            ),
            "grad_scaler": self.scaler.state_dict(),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> int:


        model_state = checkpoint["model"]
        if not isinstance(model_state, Mapping):
            raise TypeError("checkpoint model state must be a mapping")
        self.model.load_state_dict(
            _state_dict_for_model(self.model, model_state),
            strict=True,
        )
        self.core_optimizer.load_state_dict(checkpoint["core_optimizer"])
        if self.selector_optimizer is not None:
            state = checkpoint.get("selector_optimizer")
            if state is None:
                raise RuntimeError("selector optimizer state is missing")
            self.selector_optimizer.load_state_dict(state)
        if checkpoint.get("grad_scaler") is not None:
            self.scaler.load_state_dict(checkpoint["grad_scaler"])
        state = self.curriculum.at(max(int(checkpoint["step"]) - 1, 0))
        self.model.decoder.set_effective_s_ref(state.effective_s_ref)
        if self.model.decoder.selector is not None:
            self.model.decoder.set_selector_calibration(
                apply=state.selector_apply,
                threshold=float(self.config["model"]["teacher_selector"]["threshold"]),
                max_delta=state.selector_max_delta,
            )
        return int(checkpoint["step"])
