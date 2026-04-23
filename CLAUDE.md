---
description: FastWAM 项目介绍
alwaysApply: true
---

## 项目概述

**FastWAM**（Fast World Action Model）是一个基于 Wan2.2-TI2V-5B 视频扩散模型的机器人动作预测系统。它采用 Mixture of Transformers (MoT) 联合处理视频 token 流和动作 token 流，能够仅从观察图像（或多帧）直接预测动作，而无需在测试时生成未来视频帧。支持可配置的 `num_anchor_frames` 进行多帧历史条件输入。

## 架构说明

### 源码结构：`src/fastwam/`

- **runtime.py** —— 模型工厂函数（`create_fastwam`、`create_fastwam_joint`、`create_fastwam_idm`、`create_wan22_model`）以及 `run_training`/`run_inference` 编排器。所有工厂均调用 `from_wan22_pretrained()` 加载 Wan2.2 组件。
- **trainer.py** —— `Wan22Trainer`：基于 accelerate 的训练循环，仅解冻 `model.dit`（即 MoT）和可选的 `proprio_encoder`。使用 AdamW + cosine 学习率调度，支持梯度累积、定期评估（PSNR/SSIM/动作指标）、检查点保存和 WandB 日志记录。

### 模型层级：`src/fastwam/models/wan22/`

```
Wan22Core (wan22.py)                — 纯视频的 Wan2.2 基础模型
FastWAM (fastwam.py)                — 主模型：动作仅关注锚点帧的视频 token（可配置 num_anchor_frames）
  └─ FastWAMJoint (fastwam_joint.py)   — 变体：动作可看到全部视频 token
     └─ FastWAMIDM (fastwam_idm.py)    — 变体：两阶段带 teacher-forcing 的模型
```

**核心组件**：

- **MoT** (`mot.py`) —— Mixture of Transformers。每层独立计算各专家的 Q/K/V → 拼接 → 在 `[video || action]` 上做单次 flash-attention → 拆分 → 各专家 cross-attn + MLP。内置 KV-cache 优化（`prefill_video_cache()` / `forward_action_with_video_cache()`），支持快速的仅动作推理。
- **ActionDiT** (`action_dit.py`) —— 动作 DiT（30 块，1024 隐藏维度）。主干通过线性插值 Wan2.2 VideoDiT 权重（3072→1024）并进行 alpha-scaling 初始化。
- **WanVideoDiT** (`wan_video_dit.py`) —— 带 RoPE 和 flash-attention 的视频 Transformer。
- **WanVideoVAE** (`wan_video_vae.py`) —— 视频自编码器（训练时冻结）。

### 训练入口：`scripts/train.py` → `runtime.run_training(cfg)` → `Wan22Trainer`

### 推理模式

- **`infer_joint()`** —— 同时去噪视频和动作（同时输出预测视频与动作）
- **`infer_action()`** —— 通过 VAE 编码观察图像，运行一次视频专家缓存 K/V，随后仅对动作 token 进行去噪（无需生成未来视频）。

### 数据集流水线：`src/fastwam/datasets/lerobot/`

- **RobotVideoDataset** —— LeRobot 格式数据集。多相机图像在空间上拼接（LIBERO 2-cam 水平拼接，RoboTwin 3-cam 网格拼接）后再进行 VAE 编码。
- **FastWAMProcessor** —— 归一化（min-max 或 z-score）、delta 动作处理、动作-状态融合。

## 配置系统

采用 Hydra + OmegaConf 组合模式：

```
configs/train.yaml           (基础配置：batch、lr、epochs、精度、wandb)
  + configs/data/*.yaml      (数据集参数)
  + configs/model/*.yaml     (模型 _target_ 及架构参数)
  + configs/task/*.yaml      (任务级覆盖，选择数据+模型组合)
```

任务配置是主要的命令行接口：`task=libero_uncond_2cam224_1e-4`

可用任务包括：`libero_uncond_2cam224_1e-4`、`libero_idm_2cam224_1e-4`、`libero_joint_2cam224_1e-4`、`robotwin_uncond_3cam_384_1e-4`

在 `config_resolvers.py` 中定义了自定义 OmegaConf 解析器：`eval`、`split`、`sum_shapes`、`max_action_dim` 等。

## 关键设计决策

- 仅训练 `model.dit`（MoT）和 `proprio_encoder`，VAE 和文本编码器保持冻结
- T5 文本嵌入预先计算并以 `.pt` 文件缓存（使用 SHA256 作为键），避免重复推理
- 首个视频帧（或多个锚点帧）在每个扩散步以干净（无噪声）形式注入作为条件
- 本体感受状态被投影到文本维度并追加到文本上下文
- 训练采用 flow matching（连续扩散），视频和动作使用各自的噪声调度
- 损失函数 = `lambda_video * loss_video + lambda_action * loss_action`

## 多锚点帧（`num_anchor_frames`）

- **语义**：`N=model.num_anchor_frames` 为 anchor latent 数，`M=data.train.num_denoise_latent_frames` 为去噪段 latent 数；`num_frames`、`action_horizon` 由 `config_resolvers.py` 的 `latent_window_to_*` 从 `configs/data/*.yaml` 推导，**勿**在 `configs/model/*.yaml` 写 `num_denoise_latent_frames`（工厂 `create_fastwam*` 不认该字段）。
- **训练/切换**：Hydra 覆盖 `model.num_anchor_frames=N`；数据集右移 action/proprio，`action_horizon` 仅随 M 与 `K_vf`/`D_vae` 变，改 N 一般不改 action 步数。
- **文档**：设计/迁移见 `mawam_dev/plans_and_actions/multi_anchor/plan_opus4.7_v2.4.md`，落地与参数表见 `action_v2.4.md`；机制综述见 `multi_annchor_report.md`。
- **兼容（v2.4.1）**：ckpt 存 `num_anchor_frames`，跨 N 载入会 WARNING；eval/deploy 若 `EVALUATION.action_horizon` 与 `data.train.action_horizon` 不一致会 WARNING；dataset 初始化会打 episode 长度摘要。
