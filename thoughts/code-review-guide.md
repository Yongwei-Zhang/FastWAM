# FastWAM 代码库 Review 指南

## 模块依赖全景

```
configs/task/*.yaml          ← Hydra 入口，组合 model + data
  ├─ configs/model/*.yaml    ← _target_ 指向 runtime.py 工厂函数
  └─ configs/data/*.yaml     ← 数据集参数 (num_frames, freq_ratio, ...)

scripts/train.py
  └─ runtime.run_training(cfg)                    [runtime.py:381]
       ├─ instantiate(cfg.model) → create_fastwam  [runtime.py:82]
       │    └─ FastWAM.from_wan22_pretrained       [fastwam.py:94]
       │         ├─ WanVideoVAE (frozen)
       │         ├─ WanVideoDiT (video expert)
       │         ├─ ActionDiT (action expert)
       │         └─ MoT(video=VideoDiT, action=ActionDiT)
       ├─ build_datasets(cfg.data)
       │    └─ RobotVideoDataset                   [robot_video_dataset.py]
       └─ Wan22Trainer.train()                     [trainer.py:647]
            └─ model.training_loss(sample)          [fastwam.py:482]
```

---

## 按 Review 优先级排序的 6 个核心模块

### 1. 数据采样 — `src/fastwam/datasets/lerobot/robot_video_dataset.py`

关注点：`num_frames` 如何变成 video frames 和 action steps。

- **L59-63**: 两个硬约束 + `video_sample_indices` 计算
  - `num_frames=33`, `ratio=4` → indices `[0,4,8,...,32]` → **9 video frames**
  - action 保留全部 32 步 (`num_frames - 1`)
- **L145-149**: video 子采样 `video[:, self.video_sample_indices]`
- **L153-190**: 多相机拼接（horizontal/vertical/robotwin grid）
- **L202-203**: action 和 proprio 对齐，proprio 取 `[:-1]` 与 action 同长

输出 dict:
```
video:  [C, 9, H, W]     # 归一化到 [-1, 1]
action: [32, action_dim]
proprio: [32, state_dim]   # 注意：32 步，非 9 步
context: [L, D]            # 预计算 T5 embedding
```

### 2. 输入构建 + VAE 编码 — `src/fastwam/models/wan22/fastwam.py` → `build_inputs`

关注点：raw video → latent frames → anchor 切片。

- **L367**: `_encode_video_latents()` — VAE 编码 9 frames → 3 latent frames
  - VAE 公式 `wan_video_vae.py:983`: `T_lat = 1 + (T_raw - 1) // 4`
- **L369-378**: anchor 切片 `input_latents[:, :, 0:num_anchor_frames]`
- **L395-400**: proprio 编码 — 取 anchor 最后一帧的 proprio `proprio[:, num_anchor_frames - 1]`，投影到 text dim，追加到 context

### 3. Token 组织 + pre_dit

关注点：latent → token 序列，per-token timestep，RoPE。

#### Video expert — `src/fastwam/models/wan22/wan_video_dit.py` → `pre_dit` (L510)

- **L539-548**: `seperated_timestep` 模式 — 逐 token 分配 timestep
  - 全局 timestep `t` 赋给所有 token
  - **anchor 帧 token 覆盖为 0**: `token_timesteps[:, 0:num_anchor_frames, :] = 0`
- **L550**: 每个 token 独立做 sinusoidal embedding → `[B, S_v, 6, D]`
- **L562-593**: `action_conditioned=True` 时，action 编码追加到 context，构建 group causal context mask
- 3D RoPE 用于 video token 的空间-时间位置编码

#### Action expert — `src/fastwam/models/wan22/action_dit.py` → `pre_dit` (L226)

- **L283**: `action_encoder` 线性投影 `action_dim → 1024`
- **L286**: 1D RoPE（非 3D），所有 token 共享同一 timestep `t_mod`
- 独立的 context（可含 proprio）

### 4. Attention Mask — `fastwam.py` → `_build_mot_attention_mask` (L419)

关注点：joint mask 的四象限结构。

构建 `[S_v + S_a, S_v + S_a]` 布尔 mask：

```
              │  Video tokens    │  Action tokens
─────────────┼──────────────────┼────────────────
Video tokens │  video-to-video  │  False (video 不看 action)
─────────────┼──────────────────┼────────────────
Action tokens│  action-to-video │  action-to-action (全 True)
```

- **video-to-video** — 委托 `wan_video_dit.py` → `build_video_to_video_mask` (L473)
  - `"first_frame_causal"` (L502-506): anchor tokens 互相可见 + 可被后续帧看到，但 anchor 不看未来帧
  - anchor token 数 = `num_anchor_frames * tokens_per_frame`
- **action-to-video** (L441-442): action tokens **仅看 anchor 帧的 video tokens**
  - `mask[S_v:, :anchor_tokens] = True`
- **action-to-action**: 全 True

> `FastWAMJoint` 的 override 区别：action-to-video 设为全 True（action 看所有 video tokens）

### 5. MoT 前向 — `src/fastwam/models/wan22/mot.py`

关注点：video + action 如何 joint attention。

#### 训练 `forward` (L454)

- 每层：per-expert 计算 Q/K/V → concat `[S_v+S_a]` → **单次 flash attention** (L545) → split → per-expert post-block (cross-attn + MLP)
- attention mask 即上面构建的 joint mask

#### 推理 KV Cache — `infer_action` 的核心加速

- `prefill_video_cache` (L263): 仅 video expert forward，缓存每层 `{k, v}`
- `forward_action_with_video_cache` (L349): action Q/K/V + cached video K/V → concat → attention
  - 即 action queries attend to 所有 cached video K/V + 自身 K/V

### 6. Loss 计算 — `fastwam.py` → `training_loss` (L482)

关注点：anchor 排除、双 loss 加权。

- **L505-516**: video diffusion — 采样 timestep，加噪，**anchor 帧替换为 clean latents**
  ```python
  noised_latents[:, :, 0:n_anchor] = anchor_latents
  ```
- **L520-525**: action diffusion — 独立采样 timestep，加噪
- **L561**: `mot.forward(...)` 联合前向
- **L596-607**: video loss 排除 anchor 帧
  ```python
  pred_video = pred_video[:, :, n_anchor:]
  target_video = target_video[:, :, n_anchor:]
  ```
- **L627**: `loss = λ_video * loss_video + λ_action * loss_action`（默认各 1.0）

---

## 配置组合路径

```
task yaml (e.g. libero_uncond_2cam224_1e-4.yaml)
  defaults:
    - /model: fastwam          → configs/model/fastwam.yaml
    - /data: libero_2cam       → configs/data/libero_2cam.yaml
  overrides:
    lr, batch_size, epochs, ...

model yaml:
  _target_: fastwam.runtime.create_fastwam
  num_anchor_frames: 1         ← 命令行可 override: model.num_anchor_frames=2
  video_dit_config: {...}
  action_dit_config: {...}

data yaml:
  num_frames: 33               ← 控制窗口长度
  action_video_freq_ratio: 4   ← 控制 video 子采样率
```

---

## 推荐 Review 顺序

1. **配置** — `configs/task/` → `configs/model/fastwam.yaml` → `configs/data/libero_2cam.yaml`，理解参数如何组合
2. **数据** — `robot_video_dataset.py:_get()` → 理解 sample dict 结构
3. **输入构建** — `fastwam.py:build_inputs()` → VAE 编码 + anchor 切片
4. **Token + Mask** — `wan_video_dit.py:pre_dit()` + `fastwam.py:_build_mot_attention_mask()`
5. **联合前向** — `mot.py:forward()` → 理解 joint attention 机制
6. **Loss** — `fastwam.py:training_loss()` → anchor 排除 + 双 loss
7. **推理** — `fastwam.py:infer_action()` → KV cache 路径 / `infer_joint()` → 全量去噪路径

---

## 模型变体对比

| 变体 | 类 | action-to-video 可见性 | 特点 |
|---|---|---|---|
| FastWAM | `fastwam.py` | 仅 anchor tokens | 默认模式，action 只看条件帧 |
| FastWAMJoint | `fastwam_joint.py` | 全部 video tokens | action 看所有视频帧（含去噪帧） |
| FastWAMIDM | `fastwam_idm.py` | 全部 video tokens + teacher forcing | 两阶段：先 teacher-forcing video，再去噪 action |

---

## 关键数值速查

| 参数 | 值 | 来源 |
|---|---|---|
| `num_frames` | 33 | `configs/data/*.yaml` |
| `action_video_freq_ratio` | 4 | `configs/data/*.yaml` |
| video frames | 9 | `range(0, 33, 4)` |
| latent frames | 3 | `1 + (9-1)//4` |
| action steps | 32 | `num_frames - 1` |
| video hidden_dim | 3072 | `configs/model/*.yaml` |
| action hidden_dim | 1024 | `configs/model/*.yaml` |
| num_layers (both) | 30 | `configs/model/*.yaml` |
| num_heads (both) | 24 | `configs/model/*.yaml` |
| `num_anchor_frames` | 1 (default) | `configs/model/*.yaml` |
