# 2 Anchor + 2 Denoise 优化版计划

## Summary

- 当前修正版方向是对的，但还差 4 个必须补齐的决策，否则实现时仍会在数据对齐、训练内评估和接口边界上反复返工。
- 目标固定为：LIBERO 保持 `32-step action horizon`，观测窗口变为 `49 raw obs`，采样后 `13 sampled frames`，VAE 后 `4 latent = 2 anchor + 2 denoise`。
- 默认基线保持不变：`1 anchor + 2 denoise -> num_frames=33, action_horizon=32`。

## Key Changes

- 配置层只把 `num_denoise_latent_frames` 作为配置期语义参数，不把它扩散成新的模型推理接口。
  - `model.num_anchor_frames` 继续作为运行时模型参数。
  - `model.num_denoise_latent_frames` 仅用于 resolver 推导 `data.train.num_frames` 和 `data.train.action_horizon`。
  - `EVALUATION.action_horizon` 仅作为显式 override；默认读取 `data.train.action_horizon`，不再回退到 `num_frames - 1`。

- 数据集与 processor 明确分离“观测长度”和“动作长度”。
  - 数据集输出 `images/state` 长度为 `num_obs_steps=49`，`action` 长度为 `num_action_steps=32`。
  - 动作和 proprio 序列都从“最后一个 anchor raw step”开始截取；`2 anchor` 时起点固定为 raw step `16`。
  - processor 新增 `num_action_steps`，文档、断言和 `postprocess()` 全部改为显式使用它，去掉 `num_obs_steps - 1` 这类旧假设。

- 模型侧统一对齐规则，删除重复且不一致的整除约束。
  - 训练校验不再要求 `action_horizon % (num_video_frames - 1) == 0`。
  - 统一改为按“去噪段 sampled transitions”校验：
    - `anchor_sampled_transitions = vae_temporal_downsample_factor * (num_anchor_frames - 1)`
    - `denoise_sampled_transitions = (num_video_frames - 1) - anchor_sampled_transitions`
    - 要求 `action_horizon % denoise_sampled_transitions == 0`
  - `wan_video_dit` 的外层校验与 action-group mask 共用同一套 `num_temporal_groups = num_latent_frames - num_anchor_frames` 逻辑，避免再次出现 `-1` 和 `-num_anchor_frames` 混用。
  - 若数据集已经把 proprio 右移到 action 起点，训练时追加到 context 的 proprio 直接取序列第 0 个时间步，不再重新从完整视频长度反推索引。

- 训练内评估与正式评测语义对齐。
  - `trainer.evaluate()` 在 `num_anchor_frames > 1` 时，使用样本视频的前 `4*(num_anchor_frames-1)+1` 个 sampled frames 作为多帧 `input_image`，不再只传首帧。
  - 训练内评估的 proprio 取右移后序列的第 0 个时间步，与部署/在线评测一致。
  - `replan_steps` 保持现有配置值，但继续像现在一样裁到 `action_horizon` 上界，不引入新默认值。

## Test Plan

- 配置回归：
  - `1 anchor + 2 denoise -> num_frames=33, action_horizon=32`
  - `2 anchor + 2 denoise -> num_frames=49, action_horizon=32`

- 数据对齐：
  - 数据样本中 `video=13 sampled frames`，`action=32`，`proprio=32`
  - `2 anchor` 时 action/proprio 起点都是 raw step `16`
  - processor `postprocess()` 不再按 `num_obs_steps - 1` 裁 action

- 模型与评估：
  - `input_latents.shape[2] == 4`，`anchor_latents.shape[2] == 2`
  - video loss 只覆盖后 2 个 latent
  - base/joint/idm 都能通过新的 shape 校验
  - `trainer.evaluate()`、LIBERO eval、RobotWin deploy 在 `2 anchor` 下都使用多帧 anchor 输入，而不是单帧退化路径

## Assumptions

- 不新增任务名；现有 task 名继续使用，行为由 resolver 后的 `num_frames/action_horizon` 决定。
- `num_denoise_latent_frames` 是配置期参数，不作为新的公开推理 API。
- 推理与部署阶段继续显式传入 `action_horizon`；模型不从 `num_video_frames` 反推动作长度。
- `2 anchor` 的多帧输入仍采用现有语义：需要 `5` 个 sampled frames，对应 `17` 个 raw env/history steps。

## 已经执行的修改（在仓库里已经 undo）
src/fastwam/utils/config_resolvers.py

```bash
# 添加了 2 个函数
def latent_window_to_num_frames(
    num_anchor_frames: int,
    num_denoise_latent_frames: int,
    action_video_freq_ratio: int,
    vae_temporal_downsample_factor: int,
):
    return 1 + int(action_video_freq_ratio) * int(vae_temporal_downsample_factor) * (
        int(num_anchor_frames) + int(num_denoise_latent_frames) - 1
    )

def latent_window_to_action_horizon(
    num_denoise_latent_frames: int,
    action_video_freq_ratio: int,
    vae_temporal_downsample_factor: int,
):
    return int(action_video_freq_ratio) * int(vae_temporal_downsample_factor) * int(num_denoise_latent_frames)

# 这个函数的最后 2 行添加上面定义的新函数
def register_default_resolvers() -> None:
    """
    Register all resolvers commonly used across entrypoints.
    Safe to call multiple times.
    """
    _register("oc.load", _oc_load)
    _register("eval", eval) # allows arbitrary python code execution in configs using the ${eval:''} resolver
    _register("split", lambda s, idx: s.split('/')[int(idx)]) # split string
    _register("max", lambda x: max(x))
    _register("round_up", math.ceil)
    _register("round_down", math.floor)
    _register("sum_shapes", sum_shapes)
    _register("max_action_dim", max_action_dim)
    _register("max_state_dim", max_state_dim)
    # 新添加
    _register("latent_window_to_num_frames", latent_window_to_num_frames)
    _register("latent_window_to_action_horizon", latent_window_to_action_horizon)
```
