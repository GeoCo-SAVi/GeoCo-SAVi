from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch._functorch.config
import torch._inductor.config
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import GeoCoSAVi
from scripts.data_loaders import build_data_loader
from train import GeoCoTrainer


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def load_factory(specification: str):


    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("loader must be written as package.module:function")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError("loader factory is not callable")
    return factory


def parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    return {
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "total": sum(parameter.numel() for parameter in model.parameters()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--loader",
        help="Advanced loader override as package.module:function",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disable-compile", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    seed = int(config["training"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model = GeoCoSAVi.from_config(config)
    print(json.dumps(parameter_summary(model), sort_keys=True))
    if arguments.dry_run:
        return
    if arguments.workers < 0:
        parser.error("--workers must be non-negative")
    if (
        bool(config["training"].get("compile_decoder", False))
        and not arguments.disable_compile
    ):
        if hasattr(torch._functorch.config, "donated_buffer"):
            torch._functorch.config.donated_buffer = False
        torch._inductor.config.triton.cudagraph_trees = False
        model.decoder = torch.compile(model.decoder, mode="reduce-overhead")
        model.decoder.forward_alpha_logits = torch.compile(
            model.decoder.forward_alpha_logits, mode="reduce-overhead"
        )

    trainer = GeoCoTrainer(model, config)
    start_step = 0
    if arguments.resume is not None:
        checkpoint = torch.load(
            arguments.resume,
            map_location=trainer.device,
            weights_only=False,
        )
        start_step = trainer.restore(checkpoint)

    external_factory = (
        load_factory(arguments.loader) if arguments.loader else None
    )

    def make_loader(frame_count: int):
        if external_factory is not None:
            return external_factory(config)
        return build_data_loader(
            config,
            split=str(config["dataset"]["train_split"]),
            frame_count=frame_count,
            num_workers=arguments.workers,
        )

    current_frame_count = trainer.curriculum.at(start_step).frame_count
    data_loader = make_loader(current_frame_count)
    loader = iter(data_loader)
    arguments.output.mkdir(parents=True, exist_ok=True)
    max_steps = int(config["training"]["max_steps"])
    log_every = int(config["training"]["log_every"])
    checkpoint_every = int(config["training"]["checkpoint_every"])
    for step in range(start_step, max_steps):
        required_frames = trainer.curriculum.at(step).frame_count
        if (
            external_factory is None
            and required_frames != current_frame_count
        ):


            del loader
            del data_loader
            current_frame_count = required_frames
            data_loader = make_loader(current_frame_count)
            loader = iter(data_loader)
        try:
            batch = next(loader)
        except StopIteration:
            if external_factory is not None:
                data_loader = make_loader(current_frame_count)
            loader = iter(data_loader)
            batch = next(loader)
        metrics = trainer.train_step(batch, step)
        if (step + 1) % log_every == 0:
            print(json.dumps({"step": step + 1, **metrics}, sort_keys=True))
        if (step + 1) % checkpoint_every == 0 or step + 1 == max_steps:
            torch.save(
                trainer.checkpoint(step + 1),
                arguments.output / f"step_{step + 1:06d}.pt",
            )


if __name__ == "__main__":
    main()
