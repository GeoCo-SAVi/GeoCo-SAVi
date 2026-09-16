from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


def _ramp(step: int, start: int, duration: int) -> float:
    if step < start:
        return 0.0
    if duration <= 0:
        return 1.0
    return min(max((step - start + 1) / float(duration), 0.0), 1.0)


@dataclass(frozen=True)
class CurriculumState:


    step: int
    frame_count: int
    reconstruction_weight: float
    position_weight: float
    overlap_weight: float
    geometry_weight: float
    effective_s_ref: float
    temporal_active: bool
    core_lr_multiplier: float
    temporal_lr_multiplier: float
    selector_train: bool
    selector_apply: bool
    selector_max_delta: float


class GeoCoCurriculum:


    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.training = config["training"]
        self.loss = config["loss"]
        self.schedule = config["curriculum"]
        self.decoder = config["model"]["decoder"]
        self.selector = config["model"]["teacher_selector"]
        self.temporal = config["model"]["temporal"]
        self.max_steps = int(self.training["max_steps"])

    def _core_lr(self, step: int) -> float:
        warmup = int(self.training["warmup_steps"])
        if warmup > 0 and step < warmup:
            return (step + 1) / float(warmup)
        span = max(self.max_steps - warmup, 1)
        progress = min(max((step - warmup) / float(span), 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _effective_s_ref(self, step: int) -> float:
        final = float(self.decoder["s_ref"])
        cold = float(self.decoder.get("coldstart_s_ref", final))
        hold = int(self.decoder.get("s_ref_hold_steps", 0))
        duration = int(self.decoder.get("s_ref_ramp_steps", 0))
        if step < hold:
            return cold
        if duration <= 0 or step >= hold + duration:
            return final
        progress = _ramp(step, hold, duration)
        return math.exp(
            (1.0 - progress) * math.log(cold)
            + progress * math.log(final)
        )

    def _temporal_lr(self, step: int) -> float:
        start = int(self.schedule["temporal_start_step"])
        warmup = int(self.schedule["temporal_warmup_steps"])
        peak = float(
            self.schedule.get("temporal_learning_rate_multiplier", 0.25)
        )
        if step < start:
            return 0.0
        if step < start + warmup:
            return peak * _ramp(step, start, warmup)
        span = max(self.max_steps - start - warmup, 1)
        progress = min(
            max((step - start - warmup) / float(span), 0.0),
            1.0,
        )
        return peak * 0.5 * (1.0 + math.cos(math.pi * progress))

    def at(self, step: int) -> CurriculumState:
        step = int(step)
        if step < 0 or step > self.max_steps:
            raise ValueError("step lies outside the configured trajectory")
        position_ramp = _ramp(
            step,
            int(self.schedule["position_start_step"]),
            int(self.schedule["position_ramp_steps"]),
        )
        geometry_ramp = _ramp(
            step,
            int(self.schedule["geometry_start_step"]),
            int(self.schedule["geometry_ramp_steps"]),
        )
        temporal_start = int(self.schedule["temporal_start_step"])
        phase3_frames = int(self.schedule["phase3_frames"])
        phase12_frames = int(self.schedule["phase12_frames"])

        selector_enabled = bool(self.selector.get("enabled", False))
        selector_train_start = int(
            self.schedule.get("selector_train_start_step", self.max_steps + 1)
        )
        selector_apply_start = int(
            self.schedule.get("selector_apply_start_step", self.max_steps + 1)
        )
        selector_apply_ramp = int(
            self.schedule.get("selector_apply_ramp_steps", 0)
        )
        selector_apply = selector_enabled and step >= selector_apply_start
        selector_max_delta = 0.0
        if selector_apply:
            selector_max_delta = float(self.selector["max_delta"]) * _ramp(
                step, selector_apply_start, selector_apply_ramp
            )

        return CurriculumState(
            step=step,
            frame_count=(
                phase3_frames if step >= temporal_start else phase12_frames
            ),
            reconstruction_weight=float(self.loss["reconstruction_weight"]),
            position_weight=float(self.loss["position_weight"]) * position_ramp,
            overlap_weight=float(self.loss["overlap_weight"]) * position_ramp,
            geometry_weight=float(self.loss["geometry_weight"]) * geometry_ramp,
            effective_s_ref=self._effective_s_ref(step),
            temporal_active=(
                bool(self.temporal["enabled"]) and step >= temporal_start
            ),
            core_lr_multiplier=self._core_lr(step),
            temporal_lr_multiplier=(
                self._temporal_lr(step)
                if bool(self.temporal["enabled"])
                else 0.0
            ),
            selector_train=(
                selector_enabled and step >= selector_train_start
            ),
            selector_apply=selector_apply,
            selector_max_delta=selector_max_delta,
        )
