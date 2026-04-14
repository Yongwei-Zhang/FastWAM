---
description: Fastwam Project Introduction
alwaysApply: true
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastWAM (Fast World Action Model) is a robot action prediction system built on Wan2.2-TI2V-5B video diffusion. It uses a Mixture of Transformers (MoT) to jointly process video and action token streams, enabling action prediction from observation image(s) without generating future video frames at test time. Supports configurable `num_anchor_frames` for multi-frame history conditioning (see `thoughts/` for experiment plans and implementation details).

## Environment Setup

```bash
conda create -n fastwam python=3.10 -y && conda activate fastwam
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
```

For LIBERO evaluation: `pip install mujoco==3.3.2`

## Key Commands

### Preprocessing (required once before training)

```bash
# Initialize ActionDiT backbone from Wan2.2 weights
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16

# Precompute T5 text embeddings
python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4
```

### Training

```bash
# DeepSpeed Zero-1 (8 GPUs, LIBERO)
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# DeepSpeed Zero-2 (64 GPUs, RoboTwin)
bash scripts/train_zero2.sh 64 task=robotwin_uncond_3cam_384_1e-4

# Multi-anchor training (override num_anchor_frames)
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 model.num_anchor_frames=2
```

Training entrypoint: `scripts/train.py` -> `runtime.run_training(cfg)` -> `Wan22Trainer`

### Evaluation

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 ckpt=<path> MULTIRUN.num_gpus=8

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 ckpt=<path> MULTIRUN.num_gpus=8
```

### No test suite exists. Validation is done via benchmark evaluation (success rates).

## Architecture

### Source Layout: `src/fastwam/`

- **runtime.py** — Model factories (`create_fastwam`, `create_fastwam_joint`, `create_fastwam_idm`, `create_wan22_model`) + `run_training`/`run_inference` orchestrators. All factories call `from_wan22_pretrained()` to load Wan2.2 components.
- **trainer.py** — `Wan22Trainer`: accelerate-based training loop, freezes everything except `model.dit` (the MoT) and optional `proprio_encoder`. AdamW, cosine schedule, gradient accumulation, periodic eval with PSNR/SSIM/action metrics, checkpoint saving, WandB logging.

### Model Hierarchy: `src/fastwam/models/wan22/`

```
Wan22Core (wan22.py)           — Base video-only Wan2.2 model
FastWAM (fastwam.py)           — Main model: action sees anchor-frame video tokens (configurable num_anchor_frames)
  └─ FastWAMJoint (fastwam_joint.py)  — Variant: action sees ALL video tokens
     └─ FastWAMIDM (fastwam_idm.py)   — Variant: two-stage with teacher-forcing
```

Core components:
- **MoT** (`mot.py`) — Mixture of Transformers. Per layer: independent Q/K/V per expert -> concatenate -> single flash-attention over `[video || action]` -> split -> per-expert cross-attn + MLP. Has KV-cache optimization (`prefill_video_cache()` / `forward_action_with_video_cache()`) for fast action-only inference.
- **ActionDiT** (`action_dit.py`) — Action DiT (30 blocks, 1024 hidden dim). Backbone initialized by linearly interpolating Wan2.2 VideoDiT weights (3072->1024) with alpha-scaling.
- **WanVideoDiT** (`wan_video_dit.py`) — Video transformer with RoPE, flash-attention.
- **WanVideoVAE** (`wan_video_vae.py`) — Video autoencoder (frozen during training).

### Inference Modes

- **`infer_joint()`** — Denoise video + action simultaneously (produces both predicted video and actions)
- **`infer_action()`** — Encode observation via VAE, run video expert once to cache K/V, then denoise only action tokens using cached K/V. No future video generation needed.

### Dataset Pipeline: `src/fastwam/datasets/lerobot/`

- **RobotVideoDataset** — LeRobot-format dataset. Multi-camera frames are spatially concatenated (horizontal for LIBERO 2-cam, grid for RoboTwin 3-cam) before VAE encoding.
- **FastWAMProcessor** — Normalization (min-max or z-score), delta-action handling, action-state merging.

## Configuration System

Hydra + OmegaConf with composition pattern:

```
configs/train.yaml           (base: batch, lr, epochs, precision, wandb)
  + configs/data/*.yaml      (dataset params)
  + configs/model/*.yaml     (model _target_ + architecture params)
  + configs/task/*.yaml      (task-level overrides selecting data+model combo)
```

Task configs are the primary CLI interface: `task=libero_uncond_2cam224_1e-4`

Available tasks: `libero_uncond_2cam224_1e-4`, `libero_idm_2cam224_1e-4`, `libero_joint_2cam224_1e-4`, `robotwin_uncond_3cam_384_1e-4`

Custom OmegaConf resolvers in `config_resolvers.py`: `eval`, `split`, `sum_shapes`, `max_action_dim`, etc.

## Key Design Decisions

- Only `model.dit` (MoT) and `proprio_encoder` are trainable; VAE and text encoder stay frozen
- T5 text embeddings are precomputed and cached as `.pt` files (SHA256-keyed) to skip repeated T5 inference
- First video frame latents (or multiple anchor frame latents when `num_anchor_frames > 1`) are injected clean (no noise) at every diffusion step as conditioning
- Proprioceptive state is projected to text dimension and appended to text context
- Training uses flow matching (continuous diffusion) with separate noise schedules for video and action
- Loss = `lambda_video * loss_video + lambda_action * loss_action`

## Multi-Anchor Frames (`num_anchor_frames`)

Configurable parameter in `configs/model/fastwam*.yaml` that controls how many latent frames are used as clean anchor conditioning. Default is 1 (original single first-frame behavior).

- `num_anchor_frames` operates in **latent space**: VAE encodes `T_latent = 1 + (T_raw - 1) // 4`
- With current `num_frames=9` (3 latent frames), max anchor is 2 (must leave ≥1 frame for denoising)
- At inference, each observation frame is independently VAE-encoded and concatenated (no T%4==1 constraint)
- Evaluation scripts (`eval_libero_single.py`, `deploy_policy.py`) maintain a `deque(maxlen=num_anchor_frames)` frame buffer updated every env step
- Detailed implementation notes: `thoughts/multi-anchor-impl.md`
