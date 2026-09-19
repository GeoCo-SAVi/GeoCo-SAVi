# GeoCo-SAVi

GeoCo-SAVi is a geometry-consistent slot-attention model with an explicit appearance state `a`, position `p`, and scale `s`. Position and scale are read from normalized attention moments, while a geometry-conditioned RGB/alpha decoder makes those variables effective controls of the rendered support.

This repository contains the method-defining core used for the Obj3D and MOVi-C protocols:

- invariant Slot Attention with iid random position initialization;
- scale-steered convolution and the five-block equivariant renderer;
- appearance transplantation with recipient-center/radius and
  donor-compactness geometry supervision (`L_geo`);
- the STATM temporal initializer with a zero-initialized residual projection;
- the Obj3D selective-alpha teacher and RGB-target-free selector;
- factual and counterfactual metric primitives.

## Layout

```text
config/   Obj3D and MOVi-C method configurations
models/   encoder, slot core, equivariant renderer, and temporal initializer
train/    objectives, filters, curricula, and optimizer routing
scripts/  data preparation, training-time data processing, and training entry
eval/     evaluation metric primitives
```

## Data preparation

Model execution requires Linux, Python 3.10 or later, an NVIDIA CUDA GPU, PyTorch 2.5.1 or later, and Triton 3.1 or later. Install the model dependencies with `pip install -r requirements.txt`. Obj3D and MOVi-C use the same CUDA sampling and bilinear-resizing operators.

Install the optional conversion and streaming dependencies with:

```bash
pip install -r requirements-data.txt
```

Obj3D is expected as one directory per video with exactly 100 RGB PNG frames:

```text
data/raw/obj3d/
  train/<video_id>/*.png
  val/<video_id>/*.png
```

Convert each split into per-video uint8 tensors:

```bash
python scripts/prepare_obj3d.py \
  --input data/raw/obj3d/train --output data/obj3d --split train
python scripts/prepare_obj3d.py \
  --input data/raw/obj3d/val --output data/obj3d --split val
```

MOVi-C must first be obtained and prepared through the official TFDS release. Point the converter at the builder directory containing `dataset_info.json`; the command never downloads data:

```bash
python scripts/prepare_movi_c.py \
  --tfds-dir data/raw/movi_c_tfds/movi_c/128x128/1.0.0 \
  --output data/movi_c --split train
python scripts/prepare_movi_c.py \
  --tfds-dir data/raw/movi_c_tfds/movi_c/128x128/1.0.0 \
  --output data/movi_c --split validation
```

Each MOVi-C shard retains a complete 24-frame video. Validation shards also retain instance masks and forward flow. The training loader draws random clips and applies one shared horizontal-flip decision to the complete clip; the validation loader draws deterministic centered clips only for lightweight training monitoring. Framewise evaluation uses all 24 frames; whole-video tube F-score pools the same 24 frames before scoring.

Both built-in loaders return:

```python
{
    "video": torch.Tensor,      # (batch, frames, 3, height, width), [0, 1]
    "source_id": torch.Tensor,  # (batch,), same id for windows of one video
}
```

The source identifier is used only to prevent appearance-transplant pairs from coming from the same video. Place the frozen DINOv2 checkpoint at the relative path shown in `config/movi_c.yaml`.

## Configurations

`config/obj3d.yaml` uses a four-layer CNN that maps 64x64 RGB frames to a 16x16 token grid, six slots, one-frame training, the zero-initialized terminal appearance readout, and the selective-alpha teacher-selector. Its five learned decoder blocks are two scale-steered upsampling blocks followed by three output-resolution refinements. A bounded, zero-initialized RGB-only micro residual contributes to the RGB output without entering the alpha path.

`config/movi_c.yaml` first resizes native RGB bilinearly to 64x64, then bicubically to 336x336 for frozen DINOv2 block-12 features projected from 384 to 128 dimensions. It uses eleven slots, two-frame clips through 70k, and
four-frame clips afterward. Its five learned decoder blocks are one 32x32 refinement, one 32-to-64 upsampling block, and three 64x64 refinements. Selective alpha ownership is disabled for MOVi-C. The terminal appearance readout and RGB micro residual are also disabled in the reported MOVi-C configuration.

The renderer reference-scale continuation is a global numerical conditioning schedule. It neither replaces nor supervises the per-slot scale.

## Training interface

Validate model construction:

```bash
python scripts/train.py --config config/obj3d.yaml --dry-run
```

Train directly with the prepared built-in dataset:

```bash
python scripts/train.py \
  --config config/obj3d.yaml \
  --output outputs/obj3d
```

The reported MOVi-C run uses eight data-loader workers:

```bash
python scripts/train.py \
  --config config/movi_c.yaml \
  --output outputs/movi_c \
  --workers 8
```

Use `--loader package.module:function` to provide a custom data loader; it must return the same batch contract and at least the maximum frame count declared by the curriculum.

Checkpoints retain the training state needed to resume the continuous curriculum.

## Selective alpha ownership

Obj3D uses target RGB only to create detached, tri-state training labels: trusted transfer, protected object evidence, or unknown. The deployed selector never receives target RGB. It predicts whether to apply a bounded symmetric logit correction that subtracts `delta` from the current owner and adds the same `delta` to background; every unselected pixel is exactly unchanged.

MOVi-C uses neither this teacher nor this selector. Multiple factual background slots are aggregated by log-sum-exp in the gamma-2 OneFG geometry proxy. Formal segmentation always uses the full-softmax slot argmax.
