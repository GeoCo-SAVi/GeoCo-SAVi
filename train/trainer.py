from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Mapping

import torch
import torch.nn as nn

from models.geoco_savi import GeoCoSAVi, VideoOutput

from .curriculum import CurriculumState, GeoCoCurriculum
from .geometry import (
    build_transplanted_scenes,
    factual_slot_quality,
    geometry_coherence_filter,
    geometry_loss,
    onefg_support,
    select_transplant_pairs,
)
from .losses import (
    normalized_attention_overlap,
    position_alignment_loss,
    reconstruction_loss,
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
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        slots, alpha, _, attention = self._flatten_video(output)
        quality = self._quality(slots, alpha, attention)
        appearance_dim = self.model.slot_attention.appearance_dim


        routed_slots = torch.cat(
            [
                slots[..., :appearance_dim],
                slots[..., appearance_dim:].detach(),
            ],
            dim=-1,
        )
        decoded = self.model.decoder(
            routed_slots,
            attention=attention,
            use_selector=True,
        )
        loss, metrics = position_alignment_loss(
            slots=routed_slots,
            mask_logits=decoded.mask_logits,
            background_mask=quality["background"],
            valid=quality["foreground"],
            appearance_dim=appearance_dim,
            max_support_coverage=self.config["loss"].get(
                "position_max_support_coverage"
            ),
            onefg_gamma=float(self.config["loss"]["onefg_gamma"]),
            huber_delta=float(self.config["loss"]["huber_delta"]),
        )
        return loss, metrics

    def _geometry_objective(
        self,
        output: VideoOutput,
        source_ids: torch.Tensor,
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
            }
        transplanted = build_transplanted_scenes(
            slots,
            pairs,
            appearance_dim=self.model.slot_attention.appearance_dim,
        )


        counterfactual = self.model.decoder(
            transplanted,
            use_selector=False,
        )
        counterfactual_support = onefg_support(
            counterfactual.mask_logits,
            quality["background"][pairs.recipient_batch],
            gamma=float(self.config["loss"]["onefg_gamma"]),
            allow_missing_background=True,
        )
        loss, parts = geometry_loss(
            factual_support=factual_support,
            counterfactual_support=counterfactual_support,
            factual_slots=slots,
            pairs=pairs,
            appearance_dim=self.model.slot_attention.appearance_dim,
            compactness_weight=float(
                self.config["loss"]["compactness_weight"]
            ),
            huber_delta=float(self.config["loss"]["huber_delta"]),
        )
        return loss, {
            "geometry_center": parts["center"],
            "geometry_radius": parts["radius"],
            "geometry_compactness": parts["compactness"],
            "geometry_pairs": parts["pairs"],
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
            if state.position_weight > 0.0:
                position, position_metrics = self._position_objective(output)
            else:
                position = zero
                position_metrics = {"valid": zero.detach()}
            if state.geometry_weight > 0.0:
                geometry, geometry_metrics = self._geometry_objective(
                    output, source_ids
                )
            else:
                geometry = zero
                geometry_metrics = {
                    "geometry_center": zero.detach(),
                    "geometry_radius": zero.detach(),
                    "geometry_compactness": zero.detach(),
                    "geometry_pairs": zero.detach(),
                }
            terminal_gate = self.model.slot_attention.terminal_gate
            gate_regularization = (
                terminal_gate.square().mean()
                if terminal_gate is not None
                else zero
            )
            core_loss = (
                state.reconstruction_weight * reconstruction
                + state.overlap_weight * overlap
                + state.position_weight * position
                + state.geometry_weight * geometry
                + float(self.config["loss"]["terminal_gate_weight"])
                * gate_regularization
            )
            selector_loss = None
            selector_metrics: dict[str, torch.Tensor] = {}
            if state.selector_train:
                selector_loss, selector_metrics = self._selector_objective(
                    output, video
                )

        self.scaler.scale(core_loss).backward()
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
