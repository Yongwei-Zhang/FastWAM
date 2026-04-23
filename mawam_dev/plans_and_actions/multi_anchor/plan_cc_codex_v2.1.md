# LIBERO: 2 anchor + 2 denoise 修正版计划

## 结论

原计划只有一半是对的。

- 对的部分：`49 raw obs -> 13 sampled video -> 4 latent` 这个帧数推导是对的。
- 错的部分：它把这件事写成了“只改配置、无需改代码”。如果目标是 **2 anchor + 2 denoise，同时保持原来 32-step action horizon**，当前代码并不支持，必须改数据集、模型校验和评测默认逻辑。

如果只是想快速跑一个 **49 obs + 48-step action horizon** 的新实验，那么原计划可以作为一次性试验配置；但那已经不是和当前 LIBERO 基线等价的对比。

---

## 原计划的关键错误

### 1. `num_frames=49` 会把训练目标从 32-step 动作直接改成 48-step

当前训练链路里，`action_horizon` 实际上和 `num_frames` 绑定：

- `BaseLerobotDataset` 强制 `action_size == obs_size - 1`
- `RobotVideoDataset` 直接写死 `action_size=num_frames-1`
- `experiments/libero/eval_libero_single.py` 默认 `action_horizon = cfg.data.train.num_frames - 1`
- `experiments/robotwin/fastwam_policy/deploy_policy.py` 也是同样逻辑

所以原计划里“`num_frames: 33 -> 49`，其它代码不动”并不只是多看历史，而是把动作监督长度从 `32` 改成了 `48`。

### 2. 当前模型校验不允许 `49 raw / 32 action`

现在模型里仍然假设动作长度要和整段视频窗口对齐，而不是只和去噪段对齐：

- `src/fastwam/models/wan22/fastwam.py`
  - `build_inputs()` 要求 `action_horizon % (sampled_video_frames - 1) == 0`
  - 对 `49 raw -> 13 sampled`，这里会要求动作长度能被 `12` 整除
- `src/fastwam/models/wan22/wan22.py`
  - 同样保留了 `action.shape[1] % (num_frames - 1) == 0`
- `src/fastwam/models/wan22/wan_video_dit.py`
  - 外层校验仍按 `num_latent_frames - 1`
  - 但真正构造 action group mask 时，用的是 `num_temporal_groups = num_latent_frames - num_anchor_frames`

也就是说，当前代码对多 anchor 的动作对齐约束本身就没有完全改干净。

### 3. “评测脚本无需修改”是错的

`eval_libero_single.py` 和 `deploy_policy.py` 默认都从 `num_frames - 1` 推导 `action_horizon`。  
只改配置而不改这两处，LIBERO/RobotWin 评测会自动切到 `48-step` rollout。

---

## 正确目标

目标应该明确成下面这个版本：

- LIBERO 保持原有动作预测长度：`action_horizon = 32`
- 观测窗口从 `33 raw` 增加到 `49 raw`
- 采样后视频从 `9` 帧变为 `13` 帧
- VAE latent 从 `3` 帧变为 `4` 帧
- 其中 `2 latent anchor + 2 latent denoise`

对应推导：

- `num_anchor_frames = 2`
- `num_denoise_latent_frames = 2`
- `action_video_freq_ratio = 4`
- `vae_temporal_downsample_factor = 4`
- `num_frames = 1 + 4 * 4 * (2 + 2 - 1) = 49`
- `action_horizon = 4 * 4 * 2 = 32`

---

## 修正版改动范围

### 1. 配置层：把“观测窗口长度”和“动作 horizon”拆开

需要新增两个显式语义参数：

- `model.num_anchor_frames`
- `model.num_denoise_latent_frames`

并把数据侧改成同时公开：

- `data.train.num_frames`
- `data.train.action_horizon`

建议用 resolver 自动推导，避免手工算帧数：

- `num_frames = 1 + action_video_freq_ratio * vae_temporal_downsample_factor * (num_anchor_frames + num_denoise_latent_frames - 1)`
- `action_horizon = action_video_freq_ratio * vae_temporal_downsample_factor * num_denoise_latent_frames`

这样默认基线仍然是：

- `1 anchor + 2 denoise -> num_frames=33, action_horizon=32`

而新实验变成：

- `2 anchor + 2 denoise -> num_frames=49, action_horizon=32`

### 2. 数据集：动作和 proprio 要从最后一个 anchor raw step 开始

当前数据集假设：

- 图像长度 = `num_frames`
- 动作长度 = `num_frames - 1`
- proprio 长度和图像对齐，再简单裁成 `[:-1]`

这套假设必须改。

需要改：

- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
  - 去掉 `action_size == obs_size - 1` 的硬编码
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
  - 分别接收 `num_frames` 和 `action_horizon`
  - 图像窗口继续覆盖完整 `49 raw` 观测
  - 动作窗口从“最后一个 anchor 对应的 raw step”开始截取
  - proprio 也同步右移，和动作对齐

对 `2 anchor + 2 denoise`：

- anchor 覆盖到 raw step `16`
- 动作与 proprio 都应从 raw step `16` 开始取
- 长度固定为 `32`

### 3. 模型：动作对齐约束必须改成“只约束去噪段”

下面几处都不能再继续用“整段视频窗口”做整除校验：

- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/wan22.py`
- `src/fastwam/models/wan22/wan_video_dit.py`

需要统一成两层语义：

- sampled video 级别：
  - 动作长度应对齐 **去噪段 sampled transitions**
  - 也就是对 `2 anchor + 2 denoise`，`32` 只需要对齐后半段 `8` 个 sampled transitions，而不是整段 `12`
- latent 级别：
  - action group mask 应对齐 `num_denoise_latent_frames`
  - 也就是 `num_temporal_groups = num_latent_frames - num_anchor_frames`

`build_inputs()` 里追加 proprio 的位置也要按“最后一个 anchor raw step”确定，而不是继续隐含依赖旧的单窗口假设。

### 4. 评测/部署：默认 `action_horizon` 不能再从 `num_frames-1` 推

需要修改：

- `experiments/libero/eval_libero_single.py`
- `experiments/robotwin/fastwam_policy/deploy_policy.py`

逻辑改为：

- 优先读取 `cfg.data.train.action_horizon`
- 只在显式 override 时再用评测侧参数覆盖

多 anchor 帧历史缓存逻辑本身可以保留。

对 `2 anchor`：

- raw frame history 长度 = `17`
- sampled history 长度 = `5`
- VAE 编码后得到 `2 latent anchors`

---

## 具体文件计划

### 配置

- `configs/model/fastwam.yaml`
- `configs/model/fastwam_joint.yaml`
- `configs/model/fastwam_idm.yaml`
- `configs/data/libero_2cam.yaml`
- `configs/data/robotwin.yaml`
- `src/fastwam/utils/config_resolvers.py`

### 数据集

- `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
- `src/fastwam/datasets/lerobot/robot_video_dataset.py`

### 模型与运行时

- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`
- `src/fastwam/models/wan22/wan22.py`
- `src/fastwam/models/wan22/wan_video_dit.py`
- `src/fastwam/runtime.py`

### 评测/部署

- `experiments/libero/eval_libero_single.py`
- `experiments/robotwin/fastwam_policy/deploy_policy.py`

---

## 验证

### 1. 配置回归

- `1 anchor + 2 denoise` 解析后仍然是：
  - `num_frames=33`
  - `action_horizon=32`
- `2 anchor + 2 denoise` 解析后应是：
  - `num_frames=49`
  - `action_horizon=32`

### 2. 数据集对齐

- 样本中：
  - `video` 对应 `13 sampled` 帧
  - `action` 长度为 `32`
  - `proprio` 长度为 `32`
- `2 anchor` 时，动作/proprio 起点都是 raw step `16`

### 3. 模型前向

- `input_latents.shape[2] == 4`
- `anchor_latents.shape[2] == 2`
- video loss 只覆盖后 `2` 个 latent
- base/joint/idm 三个变体都能通过 shape 校验

### 4. 评测

- LIBERO 默认 rollout 长度仍为 `32`
- `2 anchor` 时 frame history 逻辑正确：
  - raw `17`
  - sampled `5`

---

## 最终判断

原计划不能直接用。

它适合作为“快速试一个 `49/48` 新任务”的配置变更，不适合作为“在原 LIBERO 任务上只增加 1 个 anchor，同时保持 2 denoise”的正式方案。  
正式方案必须按上面的修正版做一次通用重构。
