# Action v2.4: Multi-Anchor 执行记录

基于 [plan_opus4.7_v2.4.md](plan_opus4.7_v2.4.md) 的落地记录。

## 目标

- `num_anchor_frames=2, num_denoise_latent_frames=2` → `num_frames=49, action_horizon=32, sampled_video=13, latent=4`
- baseline `num_anchor_frames=1, num_denoise_latent_frames=2` → `num_frames=33, action_horizon=32`（严格等价旧行为）

## 改动清单

### 1. 配置解析：新增 resolver

[src/fastwam/utils/config_resolvers.py](../../../src/fastwam/utils/config_resolvers.py)

- `latent_window_to_num_frames(num_anchor_frames, num_denoise_latent_frames, action_video_freq_ratio, vae_temporal_downsample_factor)`
- `latent_window_to_action_horizon(num_denoise_latent_frames, action_video_freq_ratio, vae_temporal_downsample_factor)`
- 在 `register_default_resolvers()` 中注册

### 2. 数据配置

[configs/data/libero_2cam.yaml](../../../configs/data/libero_2cam.yaml)

`data.train` 段新增 / 改造（resolver 入参统一走 `${data.train.*}` 以保持 data 段自洽，只在源头 `num_anchor_frames` 一处跨层跳到 `${model.num_anchor_frames}`）：

- `num_anchor_frames: ${model.num_anchor_frames}`（唯一跨层引用；其他派生项全部从这里 fan-out）
- `num_denoise_latent_frames: 2`
- `num_frames: ${latent_window_to_num_frames:${data.train.num_anchor_frames},${data.train.num_denoise_latent_frames},${data.train.action_video_freq_ratio},4}`
- `action_horizon: ${latent_window_to_action_horizon:${data.train.num_denoise_latent_frames},${data.train.action_video_freq_ratio},4}`
- `processor.num_obs_steps: ${data.train.num_frames}`（已有，保留）
- `processor.num_action_steps: ${data.train.action_horizon}`

[configs/data/robotwin.yaml](../../../configs/data/robotwin.yaml)

- `train` 段同上
- `val` 段所有派生项（`num_anchor_frames` / `num_denoise_latent_frames` / `num_frames` / `action_horizon` / `action_video_freq_ratio`）统一 `${data.train.xxx}` 避免双份独立源
- `val.processor.num_obs_steps` / `num_action_steps` 分别指向 `${data.val.num_frames}` / `${data.val.action_horizon}`

### 3. BaseLerobotDataset

[src/fastwam/datasets/lerobot/base_lerobot_dataset.py](../../../src/fastwam/datasets/lerobot/base_lerobot_dataset.py)

- 删除 `assert action_size == obs_size - 1`
- 新增 `action_start_offset: int = 0` 参数
- action 的 `delta_timestamps` 改为从 `action_start_offset` 开始构造
- state 的 `delta_timestamps` 保持 `obs_size` 全长（与 image 对齐），proprio 的右移切片放到 `RobotVideoDataset._get()`

### 4. RobotVideoDataset

[src/fastwam/datasets/lerobot/robot_video_dataset.py](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)

构造签名新增：

- `action_horizon: Optional[int]`
- `num_anchor_frames: int = 1`
- `num_denoise_latent_frames: Optional[int]`

构造内部：

- 一致性校验：`(num_frames-1) % action_video_freq_ratio == 0`、`(sampled_T-1) % 4 == 0`、`num_latent_frames > num_anchor_frames`、`action_horizon == action_video_freq_ratio * 4 * (num_latent_frames - num_anchor_frames)`
- `action_start_offset = 4 * action_video_freq_ratio * (num_anchor_frames - 1)` 传给 `BaseLerobotDataset`

`_get()`：

- `action` 直接透传 BaseLerobot 已右移的 `[action_horizon, D]`
- `proprio = sample["proprio"][action_start_offset:action_start_offset+action_horizon, :]`
- `proprio_is_pad` 同步切片
- `skip_padding_as_possible` 判定改为实际窗口：`proprio_is_pad[offset:offset+H]`、`image_is_pad[video_sample_indices]`、`action_is_pad` 透传
- 最终 `action.shape[0] == proprio.shape[0] == action_horizon` 断言

### 5. Processor

[src/fastwam/datasets/lerobot/processors/base_processor.py](../../../src/fastwam/datasets/lerobot/processors/base_processor.py)
[src/fastwam/datasets/lerobot/processors/fastwam_processor.py](../../../src/fastwam/datasets/lerobot/processors/fastwam_processor.py)

- 新增 `num_action_steps: Optional[int] = None`（放在参数表末尾以兼容旧签名）
- `num_action_steps is None` 时回退为 `num_obs_steps - 1`
- 在 `preprocess()` 断言 `sample["action"].shape[0] == self.num_action_steps`：`BaseProcessor.preprocess`（行 232）与 `FastWAMProcessor.preprocess`（v2.4.1 补强后，行 ~275）都挂了这条 assert，作为 dataset 层兜底之外的第二/第三保险
- `postprocess()` 原 `start_obs_step = num_obs_steps - 1` 原样保留并在 `BaseProcessor` / `FastWAMProcessor` 两处都加死代码注释（v2.4.1 统一；调用前需重审起点）

### 6. 模型对齐

[src/fastwam/models/wan22/fastwam.py](../../../src/fastwam/models/wan22/fastwam.py)

- `build_inputs` 整除检查从 `action_horizon % (num_frames-1) == 0` 改为 `action_horizon % denoise_sampled_transitions == 0`，其中 `denoise_sampled_transitions = (num_frames_sampled - 1) - vae_t * (num_anchor_frames - 1)`
- 删除 `freq_ratio = action_horizon // (num_video_frames-1)` 的 proprio 反推逻辑，改为 `proprio = proprio[:, 0, :]`（dataset 已右移）
- `infer()` 签名 `input_image: torch.Tensor` → `Union[torch.Tensor, list[torch.Tensor]]`，与 `infer_joint()` / `infer_action()` 对齐
- **v2.4.1 补强**：`save_checkpoint` 写入 `payload["num_anchor_frames"]`；`load_checkpoint` 检测到 ckpt 与 runtime 的 N 不一致时打 WARNING（见 §v2.4.1 代码补强清单 #6）

[src/fastwam/models/wan22/wan_video_dit.py](../../../src/fastwam/models/wan22/wan_video_dit.py)

- `_validate_forward_inputs` 移除 `action.shape[1] % (num_latent_frames-1) != 0` 外层校验
- 内部 `pre_dit` 已有 `num_temporal_groups = f - num_anchor_frames` 与 `action_emb.shape[1] % num_temporal_groups == 0` 校验，作为唯一真源

[wan22.py](../../../src/fastwam/models/wan22/wan22.py) 未改（`Wan22Core` 不在 `create_fastwam*` 链路）

[fastwam_joint.py](../../../src/fastwam/models/wan22/fastwam_joint.py) / [fastwam_idm.py](../../../src/fastwam/models/wan22/fastwam_idm.py) 继承 FastWAM 的 `build_inputs` 与 `save_checkpoint` / `load_checkpoint`，无需单独改

### 7. Trainer

[src/fastwam/trainer.py](../../../src/fastwam/trainer.py)

- `_to_batched_eval_sample` 松掉 `action.shape[1] % (num_video_frames-1) != 0` 检查（下游 `build_inputs` 会用正确口径校验）
- `evaluate()` 在 `num_anchor_frames > 1` 时：`input_image = [video0[:, i].unsqueeze(0) for i in range(4*(N-1)+1)]`，走 `_encode_multi_image_latents_tensor`；`proprio` 继续取 `sample["proprio"][0, 0]`（dataset 已右移）

### 8. Eval / Deploy

[experiments/libero/eval_libero_single.py](../../../experiments/libero/eval_libero_single.py)
[experiments/robotwin/fastwam_policy/deploy_policy.py](../../../experiments/robotwin/fastwam_policy/deploy_policy.py)

`action_horizon` 优先级：

1. 显式 CLI 参数（`deploy_policy.py` 支持；`eval_libero_single.py` 复用 Hydra override 到 `cfg.EVALUATION.action_horizon` 路径）
2. `cfg.EVALUATION.action_horizon`
3. `cfg.data.train.action_horizon`（新增）
4. 兼容旧配置：回退 `cfg.data.train.num_frames - 1`

**v2.4.1 补强**：任一分支走到覆盖或回退都会打 WARNING——若 `EVALUATION.action_horizon != data.train.action_horizon` 打"overrides data.train.action_horizon"，若回退到 `num_frames - 1` 打"legacy single-anchor only"。见 §v2.4.1 代码补强清单 #3/#4。

`frame_history` / `_build_multi_anchor_frame_history` / `_get_num_video_frames` 逻辑保留不变（LIBERO 侧 deque 长度 `1 + (4·(N-1)) · K_vf`，按 `K_vf` 子采样后恰好得到 `4·(N-1)+1` 帧喂给 `_encode_multi_image_latents_tensor`）。

## 执行过程

1. 读关键源文件（resolvers / configs / BaseLerobot / RobotVideoDataset / processors / fastwam / wan_video_dit / trainer / eval）明确现状与接口
2. 注册 resolver → 改数据配置 → 改 dataset 层 → 改 processor 层 → 改模型层 → 改 trainer.evaluate → 改 eval/deploy 优先级
3. 三次 linter 扫描，无报错
4. 配置解析 smoke test（`libero_uncond/idm/joint` × `robotwin_uncond/idm/joint` × `N∈{1,2}`）全部通过
5. 数据样本形状验证（见下）

## 验证结果

### 配置解析

| 场景 | num_frames | action_horizon | num_anchor_frames | num_denoise_latent_frames |
| --- | --- | --- | --- | --- |
| LIBERO baseline | 33 | 32 | 1 | 2 |
| LIBERO N=2 | 49 | 32 | 2 | 2 |
| RoboTwin baseline | 33 | 32 | 1 | 2 |
| RoboTwin N=2 | 49 | 32 | 2 | 2 |

### 数据样本形状（LIBERO，真实 lerobot 样本 + FakeProcessor）

| | baseline N=1 | multi-anchor N=2 |
| --- | --- | --- |
| `video` | `(3, 9, 224, 448)` | `(3, 13, 224, 448)` |
| `action` | `(32, 7)` | `(32, 7)` |
| `proprio` | `(32, 8)` | `(32, 8)` |
| `action_is_pad` | `(32,)` | `(32,)` |
| `proprio_is_pad` | `(32,)` | `(32,)` |
| `image_is_pad` | `(9,)` | `(13,)` |
| `action_start_offset` | 0 | 16 |
| 首步 action 来自 raw step | 0 | 16 ✓ |
| `state[offset]` 对齐 action[0] | ✓ | ✓ |

### 整除性一致性

| sampled_T | num_anchor | denoise_sampled_transitions | num_temporal_groups | 32 % 均为 0 |
| --- | --- | --- | --- | --- |
| 9 | 1 | 8 | 2 | ✓ |
| 13 | 2 | 8 | 2 | ✓ |
| 17 | 3 | 8 | 2 | ✓ |

外层 `build_inputs` 与内层 `wan_video_dit.pre_dit` 的约束口径统一。

## 副作用 / 迁移注意

- **LeRobot 尾部 pad 比例上升**：`action_start_offset=16` 让 `action delta_timestamps=[16..47]/fps`，episode 末尾 47 步 rolling sample 都可能触发 action pad。`skip_padding_as_possible=false` 会吃更多 pad 样本。v2.4.1 起 `RobotVideoDataset.__init__` 主进程启动时会打印 episode 长度 `min/mean/max`，并对"长度 < num_frames + action_start_offset"的 episode 发 INFO / WARN（见 §v2.4.1 代码补强清单 #5）
- **归一化统计**：`_get_episode_data` 的 `sliding_window_with_replication(a, action_size)` **不带 `action_start_offset` 偏移**，统计窗口是 `[t, t+action_size)` 而训练/推理实际窗口是 `[t+16(N-1), t+16(N-1)+action_size)`
  - 改 `N` 时 `action_size=K_vf · D_vae · M = 32` 不变，baseline 的 `pretrained_norm_stats` 可复用——但严格来说这是"分布近似一致"（episode-level 动作分布 stationary 的假设）而不是"数值完全一致"
  - 改 `M` 或 `K_vf` 导致 `action_size` 变化时必须重算 normalizer
- **Baseline 重采样等价性**：切窗 pad 检查从「全 obs 窗口判定」改为「实际使用窗口判定」，有效样本数可能轻微上升（< 1%）；样本 id 逐一致 → 分布一致
- **旧 `EVALUATION.action_horizon=48` 需清理**：若老 task 显式设成 `num_frames-1`，会覆盖正确值 32。v2.4.1 后 eval/deploy 启动时会主动打 WARNING，但为了兼容性仍按用户显式值执行
- **`postprocess()` 死代码**：`start_obs_step = num_obs_steps - 1` 保留旧逻辑并加注释（`BaseProcessor` / `FastWAMProcessor` 两处均已加注释，v2.4.1 对齐）；若将来接入，应设为 `0`（dataset 已右移）

## 未覆盖

- 纯视频 `Wan22Core` / `wan22.py`（`create_wan22_model` 链路）不承载 multi-anchor 语义，保留旧检查

## v2.4.1 补强（基于 plan_opus4.7_v2.4.md §11 / §12）

首轮落地后对 plan 做了 Review，在"防错三层 / 运行期诊断 / 迁移 runbook"上补了 5 处轻量守门与 2 段文档。目标是：参数、数据、权重三方任一不匹配时，都能在 **dataset 初始化、eval 启动、ckpt 载入** 任一时机立刻被日志 / assert 拦住，而不是静默跑错。

### 代码补强清单

| # | 文件 | 改动 | 失效语义 | 触发信号 |
| --- | --- | --- | --- | --- |
| 1 | [fastwam_processor.py](../../../src/fastwam/datasets/lerobot/processors/fastwam_processor.py) 第 275 行附近 | `preprocess()` 新增 `assert sample["action"].shape[0] == self.num_action_steps` | 有人绕过 dataset 直接喂样本进 processor | AssertionError |
| 2 | [fastwam_processor.py](../../../src/fastwam/datasets/lerobot/processors/fastwam_processor.py) 第 304 行附近 | `postprocess()` 加死代码注释（与 BaseProcessor 对齐） | 未来有人把该死代码接回 eval 路径 | 注释 + 行内 NOTE |
| 3 | [eval_libero_single.py](../../../experiments/libero/eval_libero_single.py) 第 819 行附近 | `EVALUATION.action_horizon` 覆盖 `data.train.action_horizon` 时打 WARNING；回退 `num_frames-1` 时也打 WARNING | 老 task 的 `EVALUATION.action_horizon=48` 残留覆盖新 32 | logging.warning |
| 4 | [deploy_policy.py](../../../experiments/robotwin/fastwam_policy/deploy_policy.py) 第 382 行附近 | 同 #3 | 同 #3 | logger.warning |
| 5 | [robot_video_dataset.py](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py) `__init__` 末尾 | 主进程打印 episode 长度 `min/mean/max`，与 `num_frames + action_start_offset` 比较；过短 episode 直接 WARN | 数据集 episode 太短，导致滑窗全 pad | logger.warning（`num_too_short>0`）或 logger.info（`num_action_may_pad>0`） |
| 6 | [fastwam.py](../../../src/fastwam/models/wan22/fastwam.py) `save_checkpoint` / `load_checkpoint` | save 时写入 `payload["num_anchor_frames"]`；load 时检测与 runtime N 不一致则 WARN | 跨 N 热插拔旧 ckpt | logger.warning |

### 行为对照表

| 场景 | v2.4（首轮） | v2.4.1（补强后） |
| --- | --- | --- |
| 旧 task 残留 `EVALUATION.action_horizon=48`，跑新 N=2 eval | 静默用 48，推理步 overshoot | WARNING："overrides data.train.action_horizon"，使用者能在第 1 秒看到 |
| LIBERO 任一子集存在 <33 步 episode | 每次 getitem 返回全 pad，loss NaN 风险 | 启动日志 WARN："N/M episodes are shorter than num_frames" |
| N=1 ckpt 被意外加载进 N=2 runtime | 权重 `strict=False` 静默吃下，推理结果语义错 | WARNING："Checkpoint has no num_anchor_frames metadata ... NOT supported" |
| N=a ckpt 加载进 N=b runtime（a≠b） | 同上 | WARNING："Checkpoint num_anchor_frames=a does NOT match runtime=b" |
| 有人绕开 RobotVideoDataset 直接喂 processor | AssertionError 在 collate 后的模型层才爆（信息量低） | processor.preprocess 立刻 AssertionError |

### 文档补强清单

| # | 文档 | 新增内容 |
| --- | --- | --- |
| 1 | [plan_opus4.7_v2.4.md](plan_opus4.7_v2.4.md) §11 | "运行期诊断与兼容性保护"小节，总结 5 处守门 |
| 2 | [plan_opus4.7_v2.4.md](plan_opus4.7_v2.4.md) §迁移 runbook | 配置解析 smoke test 命令、老任务迁移清单、跨 N ckpt 规则、数据规模评估建议 |
| 3 | [plan_opus4.7_v2.4.md](plan_opus4.7_v2.4.md) §验证 | 拆成 A/B/C/D/E/F 六段，每段加"触发信号"列 |

### 回归影响

- **零新构造参数**：所有改动都走"已有字段 + 日志 / assert"，不改构造签名、不影响 Hydra instantiate
- **旧配置可直接跑**：processor 回退分支、`EVALUATION.action_horizon` 兼容分支、ckpt `num_anchor_frames` 缺失分支都仅打 WARN 不 raise
- **训练开销**：`RobotVideoDataset.__init__` 的 episode 长度诊断只在主进程跑一次、O(num_episodes) 复杂度，LIBERO / RoboTwin 级别 <1ms
- **ckpt 格式**：只是往 payload 多写一个 int；旧 ckpt 读取保持 `strict=False` 兼容

### 仍未覆盖（Review 时诚实列出）

- **性能量化未做**：N=1 vs N=2 的训练步时、显存、推理延迟没实测。[multi_annchor_report.md](multi_annchor_report.md) §4.3 有"推理延迟 +10-20%"估算但未落地 benchmark 脚本
- **N≥3 未做矩阵测试**：`action_v2.4.md` §整除性一致性有 N=3 的理论表格，但未在真实 LeRobot 上跑端到端训练/评估
- **跨 embodiment 一致性未测**：只在 LIBERO / RoboTwin 两个数据集上做了配置解析 smoke test，若接入新数据集需重新跑 §迁移 runbook 的 smoke test

## 训练 / 评估参数与命令

> v2.4.1 注：所有命令与当前仓库一致，但相对 v2.4 新增了三条启动日志/WARNING：
>
> 1. `[RobotVideoDataset/train] episodes=... length min=...` — 每次训练/评估启动时由主进程打印
> 2. `[fastwam_processor] action shape[0]=... mismatch ...` — 只在样本形状与 `num_action_steps` 冲突时 raise（正常训练不会看到）
> 3. `Checkpoint num_anchor_frames=... does NOT match runtime ...` — 只在跨 N 加载 ckpt 时出现（详见 §v2.4.1 代码补强清单 #6）

### 关键参数

- `model.num_anchor_frames`（模型端锚点帧数，在 `configs/model/fastwam.yaml:33` / `fastwam_joint.yaml:11` / `fastwam_idm.yaml:11` 默认 `1`，通过 `${model.num_anchor_frames}` 插值进 `data.train.num_anchor_frames`）
  - `1`：baseline，`num_frames=33, action_horizon=32, sampled_T=9, latent=3`
  - `2`：multi-anchor v2.4，`num_frames=49, action_horizon=32, sampled_T=13, latent=4`
- `data.train.num_denoise_latent_frames`（去噪段 latent 帧数 M，默认 `2`，改动需重算 `pretrained_norm_stats`）
- 评估侧 `EVALUATION.action_horizon`：在 [configs/sim_libero.yaml:32](../../../configs/sim_libero.yaml) 与 [configs/sim_robotwin.yaml:27](../../../configs/sim_robotwin.yaml) 默认为 `null`，由 `data.train.action_horizon` 推导；若显式设成 `num_frames-1=48` 会覆盖并触发 `WARNING: overrides data.train.action_horizon`（见 §v2.4.1）

### LIBERO

#### 训练

baseline（`num_anchor_frames=1`）：

```bash
# 单机 8 卡
bash scripts/train_zero2.sh 8 task=libero_uncond_2cam224_1e-4
```

multi-anchor（`num_anchor_frames=2`）：

```bash
bash scripts/train_zero2.sh 8 \
  task=libero_uncond_2cam224_1e-4 \
  model.num_anchor_frames=2
```

可选变体任务：`task=libero_joint_2cam224_1e-4` / `task=libero_idm_2cam224_1e-4`。

检查点将落在 `./runs/libero_uncond_2cam224_1e-4/<run_id>/`。

#### 评估

用 [experiments/libero/run_libero_eval.sh](../../../experiments/libero/run_libero_eval.sh) 的多 GPU 多 suite 并行方案，先改脚本里的 `TASK`/`CKPT`/`DATASET_STATS_PATH`，再：

```bash
# baseline
bash experiments/libero/run_libero_eval.sh

# multi-anchor：命令行追加 Hydra override
bash experiments/libero/run_libero_eval.sh \
  model.num_anchor_frames=2
```

也可直接调 `run_libero_manager.py`（等价）：

```bash
python experiments/libero/run_libero_manager.py \
  --config-name=sim_libero \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  model.num_anchor_frames=2 \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2 \
  MULTIRUN.task_suite_names=[libero_spatial,libero_object,libero_goal,libero_10]
```

单任务 debug：

```bash
python experiments/libero/eval_libero_single.py \
  --config-name=sim_libero \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=<ckpt.pt> \
  EVALUATION.dataset_stats_path=<stats.json> \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=50 \
  model.num_anchor_frames=2 \
  gpu_id=0
```

### RoboTwin

#### 训练

baseline：

```bash
bash scripts/train_zero2.sh 8 task=robotwin_uncond_3cam_384_1e-4
```

multi-anchor：

```bash
bash scripts/train_zero2.sh 8 \
  task=robotwin_uncond_3cam_384_1e-4 \
  model.num_anchor_frames=2
```

可选变体：`task=robotwin_joint_3cam_384_1e-4` / `task=robotwin_idm_3cam_384_1e-4`。

#### 评估

```bash
python experiments/robotwin/run_robotwin_manager.py \
  --config-name=sim_robotwin \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=<ckpt.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  model.num_anchor_frames=2 \
  MULTIRUN.enabled=true \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

单任务 debug：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  --config-name=sim_robotwin \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=<ckpt.pt> \
  EVALUATION.dataset_stats_path=./data/robotwin2.0/dataset_stats.json \
  EVALUATION.task_name=<task_name> \
  EVALUATION.task_config=demo_randomized \
  model.num_anchor_frames=2 \
  gpu_id=0
```

### 配置解析验证（sanity check）

v2.4.1 推荐的完整形式（含 `num_action_steps` 字段，便于验证 processor 联动）：

```bash
python -c "
from hydra import compose, initialize
from fastwam.utils.config_resolvers import register_default_resolvers
register_default_resolvers()
with initialize(config_path='configs', version_base=None):
    for task in ['libero_uncond_2cam224_1e-4','robotwin_uncond_3cam_384_1e-4']:
        for n in (1, 2):
            cfg = compose(config_name='train', overrides=[f'task={task}', f'model.num_anchor_frames={n}'])
            print(task, 'N=', n,
                  'num_frames=', cfg.data.train.num_frames,
                  'action_horizon=', cfg.data.train.action_horizon,
                  'num_action_steps=', cfg.data.train.processor.num_action_steps)
"
```

预期：

```
libero_uncond_2cam224_1e-4 N= 1 num_frames= 33 action_horizon= 32 num_action_steps= 32
libero_uncond_2cam224_1e-4 N= 2 num_frames= 49 action_horizon= 32 num_action_steps= 32
robotwin_uncond_3cam_384_1e-4 N= 1 num_frames= 33 action_horizon= 32 num_action_steps= 32
robotwin_uncond_3cam_384_1e-4 N= 2 num_frames= 49 action_horizon= 32 num_action_steps= 32
```

> 这段验证跟 [plan_opus4.7_v2.4.md §迁移 / 上线 runbook §1](plan_opus4.7_v2.4.md) 给出的命令等价；plan 侧另外展示了"只打印 num_frames / action_horizon"的简版。

## Multi-Anchor 可调参数总览

符号约定：
- `N = model.num_anchor_frames`：锚点 latent 帧数
- `M = data.train.num_denoise_latent_frames`：去噪段 latent 帧数
- `K_vf = data.train.action_video_freq_ratio`：raw → sampled 的帧步长
- `D_vae = 4`（Wan2.2 VAE 的 `temporal_downsample_factor`，resolver 里硬编码）

### 主调参数（用户显式设置）

| 参数 | 默认值 | 位置 | 作用 | 调节规则 |
| --- | --- | --- | --- | --- |
| `model.num_anchor_frames` (N) | `1` | [configs/model/fastwam.yaml:33](../../../configs/model/fastwam.yaml)、`fastwam_joint.yaml:11`、`fastwam_idm.yaml:11` | 锚点 latent 帧数；决定 anchor / action 起点 | 任意整数 `≥ 1`；=1 退化 baseline；增大 N 覆盖更长历史但 `num_frames` 线性增长、episode 尾部 pad 比例上升；**跨 N 的 ckpt 不可热插拔**（v2.4.1 载入时 WARN） |
| `data.train.num_denoise_latent_frames` (M) | `2` | [configs/data/libero_2cam.yaml:35](../../../configs/data/libero_2cam.yaml)、`robotwin.yaml:25` | 去噪段 latent 帧数；**唯一决定 action_horizon** | 任意整数 `≥ 1`；改 M 后 `pretrained_norm_stats` 必须作废重算 |
| `data.train.action_video_freq_ratio` (K_vf) | `4` | `configs/data/*.yaml` | raw → sampled 帧步长 | 必须让 `(num_frames-1) % K_vf == 0` 且 `((num_frames-1)//K_vf) % 4 == 0`；当前所有任务均为 `4` |
| `data.train.global_sample_stride` | `1` | `configs/data/*.yaml` | LeRobot `delta_timestamps` 的 raw step 倍乘 | 放大后等于"抽帧训练"；与 N / M 正交，不影响 resolver 口径 |
| `data.train.skip_padding_as_possible` | LIBERO `false`, RoboTwin `false` | `configs/data/*.yaml` | 出现 pad 时是否重采样 | N 增大 → action 窗口起点更靠后 → pad 样本比例升高时可考虑开启。v2.4.1 起 dataset 初始化会打印 episode 长度统计，当"num_action_may_pad > 0 且 N>1"时打 INFO 建议开启 |
| `data.train.max_padding_retry` | `3` | `configs/data/*.yaml` | `skip_padding_as_possible=true` 时的最大重采样次数 | 只在 skip 开启时生效 |

### 派生量（由 resolver 自动计算，不建议手动覆盖）

| 参数 | 公式 | 示例（N=2, M=2, K_vf=4） |
| --- | --- | --- |
| `data.train.num_frames` | `1 + K_vf · D_vae · (N + M - 1)` | 49 |
| `data.train.action_horizon` | `K_vf · D_vae · M` | 32 |
| `data.train.num_anchor_frames` | `${model.num_anchor_frames}` | 2 |
| `processor.num_obs_steps` | `${data.train.num_frames}` | 49 |
| `processor.num_action_steps` | `${data.train.action_horizon}` | 32 |
| `action_start_offset`（内部） | `K_vf · D_vae · (N - 1)` | 16 |
| `sampled_T`（视频帧数，内部） | `(num_frames - 1) // K_vf + 1` | 13 |
| `num_latent_frames`（内部） | `N + M` | 4 |
| `denoise_sampled_transitions`（内部校验） | `K_vf · M` | 8 |
| `num_temporal_groups`（MoT mask，内部） | `M` | 2 |

### 评估侧参数

| 参数 | 默认值 | 位置 | 作用 | 规则 |
| --- | --- | --- | --- | --- |
| `EVALUATION.action_horizon` | `null` | [configs/sim_libero.yaml:32](../../../configs/sim_libero.yaml)、[configs/sim_robotwin.yaml:27](../../../configs/sim_robotwin.yaml) | 覆盖一次执行的动作步数 | **保持 `null`** 让其按 CLI → `data.train.action_horizon` 推导；若显式设成非 null 值且 ≠ `data.train.action_horizon`，v2.4.1 起打 `WARNING: overrides data.train.action_horizon`；**绝不**设成 `num_frames-1` |
| `EVALUATION.replan_steps` | LIBERO `10`, RoboTwin `24` | `configs/sim_*.yaml` | 每次推理后实际执行的 env 步数 | 必须 `≤ action_horizon`；增大 replan_steps 可减少推理次数但控制频率下降 |
| `EVALUATION.num_inference_steps` | `${eval_num_inference_steps}=10`（[configs/train.yaml:25](../../../configs/train.yaml)） | `configs/sim_*.yaml` | flow-matching 去噪步数 | 与 multi-anchor 无耦合 |
| `EVALUATION.visualize_future_video` | `false` | `configs/sim_libero.yaml` | 走 `infer_joint`（输出视频）还是 `infer_action`（仅动作） | `true` 要求 `model.video_dit_config.action_conditioned=false` |

### 约束

**硬约束（`RobotVideoDataset.__init__` 直接 raise）：**

1. `N ≥ 1`, `M ≥ 1`
2. `(num_frames - 1) % K_vf == 0`
3. `((num_frames - 1) // K_vf) % 4 == 0`（等价 `(sampled_T - 1) % D_vae == 0`）
4. `num_latent_frames > N`（等价 `M ≥ 1`）
5. `action_horizon == K_vf · D_vae · (num_latent_frames - N)`
6. 可选 `num_denoise_latent_frames` 字段存在时必须等于 `num_latent_frames - N`

**运行期硬约束（`RobotVideoDataset._get()` / `BaseProcessor.preprocess` / `FastWAMProcessor.preprocess` / `build_inputs` / `pre_dit`）：**

7. 样本 `action.shape[0] == action_horizon == num_action_steps`
8. 样本 `proprio.shape[0] == action_horizon`
9. `action_horizon % denoise_sampled_transitions == 0`（`denoise_sampled_transitions = sampled_T-1 − D_vae·(N-1)`）
10. `action_emb.shape[1] % num_temporal_groups == 0`（`num_temporal_groups = num_latent_frames − N`）

**软约束（v2.4.1 诊断：打 log 不 raise，兼容性优先）：**

- episode 长度 ≥ `num_frames`：否则产出全 pad 样本，`RobotVideoDataset.__init__` 打 WARNING
- episode 长度 ≥ `num_frames + action_start_offset`：否则末尾样本会有 action pad，`RobotVideoDataset.__init__` 打 INFO（仅 N>1）
- ckpt `num_anchor_frames` 与 runtime 一致：否则 `FastWAM.load_checkpoint` 打 WARNING
- `EVALUATION.action_horizon == data.train.action_horizon`：否则 eval/deploy 启动时打 WARNING

### 何时需要重算 `pretrained_norm_stats`

- **改 M**：`action_size = action_horizon = K_vf · D_vae · M` 变化 → normalizer 的 action 滑窗统计作废
- **改 K_vf**：同上
- **改 N**：`action_size` **不变**，normalizer 可直接复用；**但严格来说是"分布近似一致"**（统计窗口 `[t, t+32)` vs 实际窗口 `[t+16(N-1), t+16(N-1)+32)`），若 episode 内动作分布随时间显著漂移应重算并比较
- `BaseLerobotDataset._get_episode_data` 的 `sliding_window_with_replication(a, action_size)` 决定了这一点（不带 `action_start_offset` 偏移）

### 快速决策

- 想"看更长历史"：加大 N（M 保持 2），不用重算 norm；但会触发"episode 尾部 pad 比例上升"WARN，考虑开 `skip_padding_as_possible=true`
- 想"一次动作规划更长"：加大 M，重算 norm；注意评估 `replan_steps` 默认值需同步调大（否则大部分 action chunk 被浪费）
- 既想更长历史又想更长动作：两者都加大，需重算 norm
- 同样的模型，换数据频率：调 K_vf（同时重新生成 `text_embeds_cache` 与 norm stats）
- 用旧 N ckpt 想跑新 N：**不支持**，`load_checkpoint` 会 WARN；请改回原 N 或从 pretrained 重训
