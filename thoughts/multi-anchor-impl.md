# Multi-Frame Anchor 实现记录

## 概述

将 FastWAM 的单首帧 anchor conditioning 扩展为可配置的 `num_anchor_frames`，支持多帧历史作为条件输入。`num_anchor_frames=1` 时行为与原代码完全一致。

对应 `ex-plan.md` 阶段一。

## VAE 时间压缩约束

编码规则: `T_latent = 1 + (T_raw - 1) // 4`。当前 9 raw frames → 3 latent frames。

- `num_anchor_frames` 指 **latent frames** 数量
- `num_anchor_frames=1`: 1 latent anchor（= raw frame 0），2 latent 去噪 → 原始行为
- `num_anchor_frames=2`: 2 latent anchors（= raw frames 0-4），1 latent 去噪
- 推理时: 每帧独立 VAE 编码（T=1 → 1 latent），N 帧 concat → 无 T%4==1 限制

---

## 修改文件清单

### 1. 底层 — `src/fastwam/models/wan22/wan_video_dit.py`
- `build_video_to_video_mask()`: 新增 `num_anchor_frames` 参数，anchor tokens = `num_anchor_frames * video_tokens_per_frame`
- `pre_dit()`: anchor 帧的 timestep 设为 0，temporal group 数量排除 anchor 帧，action context mask 排除 anchor tokens
- `forward()`: 透传 `num_anchor_frames`

### 2. 核心 — `src/fastwam/models/wan22/fastwam.py`
- `__init__` / `from_wan22_pretrained`: 新增 `num_anchor_frames` 参数
- `build_inputs`: 多帧 anchor 切片 `input_latents[:, :, 0:num_anchor_frames]`，proprio 取最后一个 anchor 帧，返回 key 从 `first_frame_latents` → `anchor_latents`
- `_build_mot_attention_mask`: anchor tokens 数量适配多帧
- `_compute_video_loss_per_sample`: `include_initial_video_step: bool` → `num_excluded_anchor_steps: int`
- `training_loss`: 多帧 anchor 替换 + loss 排除
- `_predict_action_noise` / `_predict_action_noise_with_cache`: 变量名 `first_frame_latents` → `anchor_latents`，透传 `num_anchor_frames`
- 新增 `_encode_multi_image_latents_tensor()`: 对 N 张图独立 VAE 编码后 concat
- `infer_action`: `input_image` 支持 `Union[Tensor, list[Tensor]]`
- `infer_joint`: 多帧 anchor pin + list input 支持

### 3. 子类 — `src/fastwam/models/wan22/fastwam_joint.py`
- `_build_mot_attention_mask`: 透传 `num_anchor_frames`
- `infer_action`: 同 base class 的 list input + anchor 逻辑

### 4. 子类 — `src/fastwam/models/wan22/fastwam_idm.py`
- `_build_teacher_forcing_attention_mask`: 透传 `num_anchor_frames`
- `training_loss`: `first_frame_latents` → `anchor_latents`，多帧 anchor 替换
- `infer_action` / `infer_joint`: 同步多帧支持

### 5. 工厂 — `src/fastwam/runtime.py`
- `create_fastwam` / `create_fastwam_joint` / `create_fastwam_idm`: 新增 `num_anchor_frames` 参数，透传至 `from_wan22_pretrained`

### 6. 配置 — `configs/model/`
- `fastwam.yaml`、`fastwam_joint.yaml`、`fastwam_idm.yaml`: 均添加 `num_anchor_frames: 1`

### 7. 评测 — `experiments/libero/eval_libero_single.py`
- `_predict_action_chunk`: 接受 `frame_history` 参数
- `run_single_episode`: 维护 `deque(maxlen=num_anchor_frames)` 帧缓冲，每步更新

### 8. 评测 — `experiments/robotwin/fastwam_policy/deploy_policy.py`
- `__init__`: 初始化帧 deque
- `_infer_action_chunk`: 更新帧历史，传 list 给 `infer_action`
- `step()`: 每步更新帧历史（即使未 replan）
- `reset()`: 清空帧历史

---

## 使用说明

### 训练

修改 task yaml 或命令行 override 设置 anchor 帧数:

```bash
# 单帧 anchor（默认，等同原始行为）
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4

# 双帧 anchor
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 model.num_anchor_frames=2
```

或直接修改 `configs/model/fastwam.yaml`:
```yaml
num_anchor_frames: 2
```

### 推理 / 评测

配置自动从 model config 读取 `num_anchor_frames`，评测脚本会自动维护帧历史缓冲。

```bash
# LIBERO
python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 ckpt=<path> MULTIRUN.num_gpus=8

# RoboTwin
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_uncond_3cam_384_1e-4 ckpt=<path> MULTIRUN.num_gpus=8
```

### 约束

- 当前数据设置 `num_frames=9`（3 latent frames），`num_anchor_frames` 最大为 2（需保留至少 1 帧去噪）
- 若需 `num_anchor_frames≥3`，须增大 `data.train.num_frames`

---

## 验证清单

- [ ] `num_anchor_frames=1` 训练/推理结果与改动前完全一致
- [ ] `num_anchor_frames=2` 训练能正常启动，loss 正常下降
- [ ] attention mask shape 正确: action tokens 可见 `num_anchor_frames * tokens_per_frame` 个 video tokens
- [ ] 评测 rollout 帧缓冲正确积累和传递
