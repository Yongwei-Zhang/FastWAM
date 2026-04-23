# 多 Anchor 且保持原去噪长度的重构计划

## Summary

- 目标基线按当前主线语义处理：当前训练链路是 `33 raw obs -> 9 sampled video -> 3 latent`，即 `1 anchor latent + 2 denoise latent`。
- 新目标是支持 `2 anchor latent + 2 denoise latent`，同时保持动作 horizon 不变为当前的 `32 raw steps`。
- 为此需要把“观测窗口长度”和“动作预测长度”从当前同一个 `data.train.num_frames` 中拆开；否则把 anchor 从 1 提到 2 会把动作 horizon 被动拉长到 48。

## Key Changes

- 配置层新增并固定两类公开参数：
  - `model.num_anchor_frames`
  - `model.num_denoise_latent_frames`
- 数据配置不再让 `action_horizon` 隐式等于 `num_frames - 1`：
  - 保留 `data.train.num_frames` 作为原始观测窗口长度
  - 新增 `data.train.action_horizon`
  - 用 resolver 自动推导二者，避免手工算帧数
- 推导规则统一为：
  - `data.train.num_frames = 1 + action_video_freq_ratio * vae_temporal_downsample_factor * (num_anchor_frames + num_denoise_latent_frames - 1)`
  - `data.train.action_horizon = action_video_freq_ratio * vae_temporal_downsample_factor * num_denoise_latent_frames`
- 在当前主线默认值下：
  - `num_anchor_frames=1, num_denoise_latent_frames=2 -> num_frames=33, action_horizon=32`
  - `num_anchor_frames=2, num_denoise_latent_frames=2 -> num_frames=49, action_horizon=32`

## Implementation Changes

- 配置与 resolver
  - 在 `src/fastwam/utils/config_resolvers.py` 增加一个专用 resolver，负责从 `anchor latent`、`denoise latent`、`action_video_freq_ratio`、`vae temporal factor` 推导 raw 帧数和动作 horizon。
  - 更新 `configs/model/fastwam*.yaml`，新增 `num_denoise_latent_frames`，默认值设为 `2`。
  - 更新 `configs/data/libero_2cam.yaml` 和 `configs/data/robotwin.yaml`，让 `num_frames` 与 `action_horizon` 由上述参数自动推导，不再手填固定 `33`。
- 数据集
  - 修改 `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`，移除 `action_size == obs_size - 1` 的硬编码约束，改为允许 `obs_size >= action_size + 1`。
  - `RobotVideoDataset` 改为分别接收 `num_frames` 和 `action_horizon`，并传给 `BaseLerobotDataset(obs_size=num_frames, action_size=action_horizon)`。
  - 数据切片语义改为：
    - 图像序列仍覆盖完整观测窗口，用于构造更长的 anchor 历史。
    - 动作序列从“最后一个 anchor 对应的 raw step”开始取，长度为 `action_horizon`。
    - `proprio` 也同步右移，和动作序列对齐，而不是继续从 raw step 0 开始。
    - `image_is_pad` 保持完整观测窗口长度；`action_is_pad` 和动作对齐后的 `proprio/state` pad 也同步右移。
  - 对 `2 anchor + 2 denoise`，动作与本体的起点固定为 raw step `16`。
- 模型与训练
  - 在 `src/fastwam/models/wan22/fastwam.py`、`fastwam_joint.py`、`fastwam_idm.py` 与 `src/fastwam/runtime.py` 中新增 `num_denoise_latent_frames` 并贯通工厂、构造函数和推理接口。
  - `build_inputs()` 不再假设 `action_horizon % (num_video_frames - 1) == 0`；改为校验：
    - `latent_t == num_anchor_frames + num_denoise_latent_frames`
    - 动作长度等于配置推导出的 `action_horizon`
  - `build_inputs()` 中用于追加 proprio 的时刻改为“最后一个 anchor raw step”，不再复用旧的单 anchor 推导。
  - 视频 loss 仍只对 denoise latents 计算，排除全部 anchor latents；这部分逻辑保留，但显式由 `num_anchor_frames` 和 `num_denoise_latent_frames` 驱动。
  - attention mask 语义不变：
    - `FastWAM` 动作只看 anchor video tokens
    - `FastWAMJoint` 动作看全部 video tokens
    - `FastWAMIDM` teacher-forcing 分支继续复用同一 anchor/denoise 划分
- 评测与部署
  - `experiments/libero/eval_libero_single.py`、`experiments/robotwin/fastwam_policy/deploy_policy.py`、`src/fastwam/trainer.py` 中所有 `action_horizon = num_frames - 1` 的默认逻辑改为读取 `cfg.data.train.action_horizon`。
  - 多 anchor 的帧历史缓存逻辑保持现有设计，只继续由 `num_anchor_frames` 控制。
  - 对 `2 anchor`，评测/部署端 raw frame history 长度固定为 `17`，采样后送 VAE 的视频帧数固定为 `5`。

## Test Plan

- 配置回归：
  - `1 anchor + 2 denoise` 时，解析结果必须仍是 `num_frames=33`、`action_horizon=32`，现有行为不变。
  - `2 anchor + 2 denoise` 时，解析结果必须是 `num_frames=49`、`action_horizon=32`。
- 数据集对齐：
  - 单测检查 `RobotVideoDataset` 输出中，视频长度、动作长度、proprio 长度和 pad mask 长度全部符合新定义。
  - 单测检查 `2 anchor + 2 denoise` 时，动作起点与 proprio 起点都从 raw step `16` 开始。
- 模型形状与语义：
  - 训练前向时 `input_latents.shape[2] == 4`。
  - `anchor_latents.shape[2] == 2`，视频 loss 只覆盖后 2 个 latents。
  - base/joint/idm 三个变体的 attention mask shape 和可见性符合各自语义。
- 评测与部署：
  - `trainer.evaluate()` 能在新配置下完成一次前向。
  - LIBERO 与 RobotWin 的 `infer_action` smoke test 能跑通，且多 anchor 历史缓存长度为 `17 raw / 5 sampled`。
- 兼容性：
  - 旧 checkpoint 在 `num_anchor_frames=1, num_denoise_latent_frames=2` 下可继续加载。
  - 旧 task 名称不变，默认配置行为不变。

## Assumptions

- 目标是通用重构，不做只服务于 `2 anchor` 的一次性分支逻辑。
- 动作 horizon 明确保持当前主线的 `32 raw steps`，不随 anchor 增长而变为 `48`。
- 评测和部署接口继续以“给定 `num_anchor_frames` 自动维护历史帧缓存”为准，不引入新的调用方式。
