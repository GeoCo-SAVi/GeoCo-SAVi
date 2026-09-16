from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def natural_key(path: Path) -> list[object]:


    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def convert_video(
    job: tuple[Path, int, int],
) -> tuple[str, torch.Tensor]:


    video_dir, frames_per_video, image_size = job
    frame_paths = sorted(video_dir.glob("*.png"), key=natural_key)
    if len(frame_paths) != frames_per_video:
        raise ValueError(
            f"{video_dir}: expected {frames_per_video} PNG frames, "
            f"found {len(frame_paths)}"
        )

    frames: list[np.ndarray] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            if image.mode != "RGB":
                raise ValueError(
                    f"{frame_path}: expected RGB PNG, found mode {image.mode!r}"
                )
            image = image.resize(
                (image_size, image_size),
                Image.Resampling.LANCZOS,
            )
            array = np.asarray(image, dtype=np.uint8)
        if array.shape != (image_size, image_size, 3):
            raise ValueError(f"{frame_path}: unexpected shape {array.shape}")
        frames.append(array.transpose(2, 0, 1))

    video = np.stack(frames, axis=0)
    return video_dir.name, torch.from_numpy(video).contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Raw split directory containing one subdirectory per video.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/obj3d"),
        help="Prepared Obj3D root.",
    )
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--frames-per-video", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _jobs(
    video_dirs: Iterable[Path],
    frames_per_video: int,
    image_size: int,
) -> Iterable[tuple[Path, int, int]]:
    for video_dir in video_dirs:
        yield video_dir, frames_per_video, image_size


def main() -> None:
    arguments = parse_args()
    if arguments.frames_per_video <= 0 or arguments.image_size <= 0:
        raise ValueError("frame count and image size must be positive")
    if arguments.workers <= 0:
        raise ValueError("--workers must be positive")

    input_dir = arguments.input
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Obj3D input directory not found: {input_dir}")
    video_dirs = sorted(
        (path for path in input_dir.iterdir() if path.is_dir()),
        key=natural_key,
    )
    if not video_dirs:
        raise FileNotFoundError(
            f"no per-video directories found under {input_dir}"
        )
    video_ids = [path.name for path in video_dirs]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("duplicate Obj3D video identifiers")

    split_dir = arguments.output / arguments.split
    if split_dir.exists() and any(split_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty prepared split: {split_dir}"
        )
    video_output = split_dir / "videos"
    video_output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    jobs = _jobs(
        video_dirs,
        arguments.frames_per_video,
        arguments.image_size,
    )
    with ProcessPoolExecutor(max_workers=arguments.workers) as executor:
        converted = executor.map(convert_video, jobs, chunksize=1)
        for source_id, (video_id, video) in enumerate(converted):
            if video.dtype != torch.uint8 or video.shape != (
                arguments.frames_per_video,
                3,
                arguments.image_size,
                arguments.image_size,
            ):
                raise RuntimeError(f"invalid converted tensor for {video_id}")
            relative_path = Path("videos") / f"{source_id:06d}.pt"
            torch.save(video, split_dir / relative_path)
            records.append(
                {
                    "id": video_id,
                    "source_id": source_id,
                    "path": relative_path.as_posix(),
                }
            )

    manifest = {
        "schema_version": 1,
        "dataset": "Obj3D",
        "split": arguments.split,
        "image_size": arguments.image_size,
        "frames_per_video": arguments.frames_per_video,
        "video_count": len(records),
        "videos": records,
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"prepared {len(records)} Obj3D {arguments.split} videos "
        f"under {split_dir}"
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
