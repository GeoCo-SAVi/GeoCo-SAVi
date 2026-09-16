from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_EXAMPLES = {"train": 9_737, "validation": 250}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tfds-dir",
        type=Path,
        required=True,
        help="Prepared MOVi-C TFDS builder directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/movi_c"),
        help="Prepared MOVi-C root.",
    )
    parser.add_argument(
        "--split",
        choices=tuple(EXPECTED_EXAMPLES),
        required=True,
    )
    parser.add_argument("--videos-per-shard", type=int, default=32)
    parser.add_argument(
        "--allow-count-mismatch",
        action="store_true",
        help="Allow a non-official example count for a deliberate subset.",
    )
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def array_spec(array: Any) -> dict[str, Any]:
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def main() -> None:
    arguments = parse_args()
    if arguments.videos_per_shard <= 0:
        raise ValueError("--videos-per-shard must be positive")


    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import numpy as np
        import tensorflow_datasets as tfds
        import tqdm
        import webdataset as wds
    except ImportError as error:
        raise RuntimeError(
            "MOVi-C conversion dependencies are missing; install "
            "requirements-data.txt"
        ) from error

    tfds_dir = arguments.tfds_dir
    info_path = tfds_dir / "dataset_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing TFDS metadata: {info_path}")

    split_dir = arguments.output / arguments.split
    if split_dir.exists() and any(split_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty prepared split: {split_dir}"
        )
    split_dir.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(str(tfds_dir))
    declared_count = int(
        builder.info.splits[arguments.split].num_examples
    )
    expected_count = EXPECTED_EXAMPLES[arguments.split]
    if (
        declared_count != expected_count
        and not arguments.allow_count_mismatch
    ):
        raise RuntimeError(
            f"TFDS metadata declares {declared_count} examples; "
            f"the official {arguments.split} split has {expected_count}"
        )
    dataset = builder.as_dataset(
        split=arguments.split,
        shuffle_files=False,
    )
    fields = (
        ("video",)
        if arguments.split == "train"
        else ("video", "segmentations", "forward_flow")
    )
    shard_pattern = str(
        split_dir / f"movi_c-{arguments.split}-%06d.tar"
    )
    first_specs: dict[str, dict[str, Any]] = {}
    example_count = 0
    with wds.ShardWriter(
        shard_pattern,
        maxcount=arguments.videos_per_shard,
    ) as sink:
        for index, record in enumerate(
            tqdm.tqdm(tfds.as_numpy(dataset), desc=arguments.split)
        ):
            sample: dict[str, Any] = {"__key__": f"{index:06d}"}
            video_length: int | None = None
            for field in fields:
                value = np.asarray(record[field])
                if field == "video":
                    if (
                        value.ndim != 4
                        or value.shape[-1] != 3
                        or value.dtype != np.uint8
                    ):
                        raise ValueError(
                            f"invalid MOVi-C RGB video at example {index}: "
                            f"{value.shape}, {value.dtype}"
                        )
                    video_length = int(value.shape[0])
                elif field == "segmentations":
                    if (
                        value.ndim not in (3, 4)
                        or (value.ndim == 4 and value.shape[-1] != 1)
                    ):
                        raise ValueError(
                            f"invalid segmentation at example {index}: "
                            f"{value.shape}"
                        )
                elif (
                    value.ndim != 4
                    or value.shape[-1] != 2
                ):
                    raise ValueError(
                        f"invalid forward flow at example {index}: "
                        f"{value.shape}"
                    )
                if video_length is None or value.shape[0] != video_length:
                    raise ValueError(
                        f"field {field!r} has an inconsistent time axis "
                        f"at example {index}"
                    )
                if field == "forward_flow":
                    encoded = value.dtype == np.uint16
                    value = value.astype(np.float32, copy=False)
                    if encoded:
                        lo, hi = np.asarray(
                            record["metadata"]["forward_flow_range"],
                            dtype=np.float32,
                        )
                        if not np.isfinite([lo, hi]).all() or hi < lo:
                            raise ValueError("invalid forward_flow_range")
                        value = value / np.float32(65535.0) * (hi - lo) + lo
                sample[f"{field}.npy"] = value
                first_specs.setdefault(field, array_spec(value))
            sink.write(sample)
            example_count += 1

    if (
        example_count != expected_count
        and not arguments.allow_count_mismatch
    ):
        raise RuntimeError(
            f"expected {expected_count} examples, converted {example_count}"
        )
    shards = sorted(
        split_dir.glob(f"movi_c-{arguments.split}-*.tar")
    )
    expected_shards = (
        example_count + arguments.videos_per_shard - 1
    ) // arguments.videos_per_shard
    if len(shards) != expected_shards:
        raise RuntimeError(
            f"expected {expected_shards} shards, found {len(shards)}"
        )

    manifest = {
        "schema_version": 1,
        "dataset": "MOVi-C",
        "split": arguments.split,
        "example_count": example_count,
        "videos_per_shard": arguments.videos_per_shard,
        "shard_count": len(shards),
        "fields": list(fields),
        "first_example": first_specs,
        "dataset_info_sha256": file_sha256(info_path),
        "shards": [
            {"name": path.name, "bytes": path.stat().st_size}
            for path in shards
        ],
    }
    (split_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"prepared {example_count} MOVi-C {arguments.split} videos "
        f"in {len(shards)} shards under {split_dir}"
    )


if __name__ == "__main__":
    main()
