from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate


class Obj3DPreparedDataset(Dataset[dict[str, torch.Tensor]]):


    def __init__(
        self,
        root: Path,
        split: str,
        frame_count: int,
        *,
        stride: int = 1,
    ) -> None:
        if frame_count <= 0 or stride <= 0:
            raise ValueError("frame count and stride must be positive")
        self.split_dir = root / split
        manifest_path = self.split_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"prepared Obj3D manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("dataset") != "Obj3D"
            or manifest.get("split") != split
        ):
            raise ValueError(f"invalid Obj3D manifest: {manifest_path}")
        self.image_size = int(manifest["image_size"])
        self.total_frames = int(manifest["frames_per_video"])
        self.frame_count = int(frame_count)
        self.records = list(manifest["videos"])
        if int(manifest["video_count"]) != len(self.records):
            raise ValueError("Obj3D manifest video count is inconsistent")
        source_ids = [int(record["source_id"]) for record in self.records]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Obj3D manifest contains duplicate source ids")
        for record in self.records:
            path = Path(str(record["path"]))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Obj3D manifest contains an unsafe video path")
            if not (self.split_dir / path).is_file():
                raise FileNotFoundError(
                    f"prepared Obj3D video not found: {self.split_dir / path}"
                )

        self.windows: list[tuple[int, int]] = []
        if self.frame_count > self.total_frames:
            raise ValueError(
                f"requested {self.frame_count} frames from "
                f"{self.total_frames}-frame Obj3D videos"
            )
        for record_index in range(len(self.records)):
            for start in range(
                0,
                self.total_frames - self.frame_count + 1,
                stride,
            ):
                self.windows.append((record_index, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, start = self.windows[index]
        record = self.records[record_index]
        path = self.split_dir / str(record["path"])
        video = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(video, torch.Tensor):
            raise ValueError(f"invalid prepared Obj3D video: {path}")
        expected = (
            self.total_frames,
            3,
            self.image_size,
            self.image_size,
        )
        if (
            video.dtype != torch.uint8
            or tuple(video.shape) != expected
        ):
            raise ValueError(f"invalid prepared Obj3D video: {path}")
        clip = video[start : start + self.frame_count].float().div_(255.0)
        return {
            "video": clip,
            "source_id": torch.tensor(
                int(record["source_id"]),
                dtype=torch.long,
            ),
        }


def _stable_source_id(key: object) -> int:
    text = str(key)
    if text.isdecimal():
        return int(text)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class MOViCWebDataset:


    def __init__(
        self,
        root: Path,
        split: str,
        frame_count: int,
        image_size: int,
        *,
        horizontal_flip_probability: float,
    ) -> None:
        split = {"val": "validation", "valid": "validation"}.get(split, split)
        if split not in {"train", "validation"}:
            raise ValueError("MOVi-C split must be train or validation")
        if frame_count <= 0 or image_size <= 0:
            raise ValueError("frame count and image size must be positive")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal flip probability must lie in [0, 1]")
        self.split = split
        self.frame_count = int(frame_count)
        self.image_size = int(image_size)
        self.horizontal_flip_probability = (
            float(horizontal_flip_probability) if split == "train" else 0.0
        )
        split_dir = root / split
        manifest_path = split_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"prepared MOVi-C manifest not found: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("dataset") != "MOVi-C"
            or manifest.get("split") != split
        ):
            raise ValueError(f"invalid MOVi-C manifest: {manifest_path}")
        shard_paths = sorted(
            split_dir.glob(f"movi_c-{split}-*.tar")
        )
        declared_names = {
            str(item["name"]) for item in manifest.get("shards", [])
        }
        actual_names = {path.name for path in shard_paths}
        if declared_names != actual_names:
            raise ValueError(
                f"MOVi-C manifest/shard mismatch under {split_dir}"
            )
        self.shards = [
            str(path) for path in shard_paths
        ]
        if not self.shards or int(manifest["example_count"]) <= 0:
            raise FileNotFoundError(
                f"no prepared MOVi-C {split} shards under {split_dir}"
            )

    def _indices(self, video_length: int) -> np.ndarray:
        if self.frame_count > video_length:
            raise ValueError(
                f"requested {self.frame_count} frames from a "
                f"{video_length}-frame MOVi-C video"
            )
        maximum_start = video_length - self.frame_count


        start = (
            random.randint(0, maximum_start)
            if self.split == "train"
            else maximum_start // 2
        )
        return start + np.arange(self.frame_count)

    def _decode(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        video = np.asarray(sample["video.npy"])
        if video.ndim != 4 or video.shape[-1] != 3:
            raise ValueError(f"expected video [T,H,W,3], got {video.shape}")
        indices = self._indices(video.shape[0])
        selected = video[indices]
        if (
            self.horizontal_flip_probability > 0.0
            and random.random() < self.horizontal_flip_probability
        ):

            selected = np.flip(selected, axis=2).copy()
        tensor = (
            torch.tensor(selected, dtype=torch.float32)
            .permute(0, 3, 1, 2)
            .div_(255.0)
        )
        if tuple(tensor.shape[-2:]) != (
            self.image_size,
            self.image_size,
        ):
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        output: dict[str, Any] = {
            "video": tensor,
            "source_id": torch.tensor(
                _stable_source_id(sample["__key__"]),
                dtype=torch.long,
            ),
        }
        if self.split == "validation":
            segmentation = np.asarray(sample["segmentations.npy"])[indices]
            if segmentation.ndim == 4 and segmentation.shape[-1] == 1:
                segmentation = segmentation[..., 0]
            if segmentation.ndim != 3:
                raise ValueError(
                    "expected segmentation [T,H,W] or [T,H,W,1]"
                )
            flow = np.asarray(sample["forward_flow.npy"])[indices]
            if flow.ndim != 4 or flow.shape[-1] != 2:
                raise ValueError("expected forward flow [T,H,W,2]")
            output["segmentations"] = torch.tensor(
                segmentation,
                dtype=torch.long,
            )
            output["forward_flow"] = torch.tensor(
                flow,
                dtype=torch.float32,
            ).permute(0, 3, 1, 2)
        return output

    def stream(
        self,
        *,
        batch_size: int,
        training: bool,
        num_workers: int,
        generator: torch.Generator,
    ) -> Iterable[dict[str, Any]]:
        try:
            import webdataset as wds
        except ImportError as error:
            raise RuntimeError(
                "MOVi-C loading requires requirements-data.txt"
            ) from error
        dataset = wds.WebDataset(
            self.shards,
            resampled=training,
            shardshuffle=False,
            handler=wds.handlers.reraise_exception,
        ).decode()
        if training:
            dataset = dataset.shuffle(512)
        dataset = dataset.map(self._decode).batched(
            batch_size,
            partial=not training,
            collation_fn=default_collate,
        )
        loader_options: dict[str, Any] = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
            "generator": generator,
        }
        if num_workers > 0:
            loader_options["prefetch_factor"] = 2
        return wds.WebLoader(dataset, **loader_options)


def build_data_loader(
    config: Mapping[str, Any],
    *,
    split: str | None = None,
    frame_count: int | None = None,
    num_workers: int = 4,
) -> Iterable[dict[str, Any]]:


    dataset_config = config["dataset"]
    training_config = config["training"]
    split = str(split or dataset_config["train_split"])
    training = split == str(dataset_config["train_split"])
    frame_count = int(frame_count or dataset_config["frames"])
    batch_size = int(training_config["batch_size"])
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    generator = torch.Generator().manual_seed(
        int(training_config["seed"])
    )
    name = str(dataset_config["name"]).casefold()
    root = Path(dataset_config["root"])

    if name == "obj3d":
        dataset = Obj3DPreparedDataset(root, split, frame_count)
        options: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": training,
            "drop_last": training,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
            "generator": generator,
        }
        if num_workers > 0:
            options["prefetch_factor"] = 2
        return DataLoader(dataset, **options)
    if name == "movi_c":
        dataset = MOViCWebDataset(
            root,
            split,
            frame_count,
            int(dataset_config["image_size"]),
            horizontal_flip_probability=0.5,
        )
        return dataset.stream(
            batch_size=batch_size,
            training=training,
            num_workers=num_workers,
            generator=generator,
        )
    raise ValueError(f"unsupported built-in dataset: {name!r}")
