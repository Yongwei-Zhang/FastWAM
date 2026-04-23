# Multi-Anchor 机制分析报告

对比对象：原始 Fast-WAM（[arxiv.org/abs/2603.16666v2](https://arxiv.org/abs/2603.16666v2)，Yuan et al. 2026）与本仓库扩展的 multi-anchor 训练 / 推理模式。

> **版本对照**：本报告对应仓库 `v2.4.1` 状态（`plan_opus4.7_v2.4.md` 首轮落地 + `action_v2.4.md` v2.4.1 补强）。
> 实现细节与设计决策见 [plan_opus4.7_v2.4.md](plan_opus4.7_v2.4.md)，落地清单与参数索引见 [action_v2.4.md](action_v2.4.md)。

---

## 1. 原始 Fast-WAM 回顾

Fast-WAM 的核心论点：

> World Action Models (WAMs) 在训练时做 video co-training 是必要的，但在 **测试时显式生成未来视频** 并非必要。

对应到代码实现（`src/fastwam/models/wan22/fastwam.py`）：

- **训练**：joint flow-matching，video 分支与 action 分支同时去噪；第一个 latent 帧（raw step 0 对应的 VAE 编码）作为 **clean anchor** 注入，每个扩散步都不加噪。
- **推理**：`infer_action()` 走"video 专家 prefill KV-cache → 仅 action 去噪"的路径，**完全跳过未来视频生成**。
- **约束**：`num_anchor_frames=1`，即仅当前观察帧作为条件。

设 `K_vf = action_video_freq_ratio = 4`，`D_vae = vae.temporal_downsample_factor = 4`：

- `num_frames = 33` 原始观察步（raw step 0..32）
- `action_horizon = 32` 动作步（raw step 0..31，从"当前"起向未来）
- VAE 编码后 `num_latent_frames = 3`，其中 latent 0 为 anchor，latent 1/2 为去噪目标

原始 Fast-WAM 是一个 **Markov 策略**：给定当前单帧观察，输出未来 32 步动作。

---

## 2. Multi-Anchor 机制

### 2.1 什么被改动

引入两个参数：

- `N = model.num_anchor_frames`：clean anchor latent 帧数
- `M = data.train.num_denoise_latent_frames`：去噪段 latent 帧数（默认 2，与 baseline 相同）

派生量（见 `[config_resolvers.py:latent_window_to_num_frames](../../../src/fastwam/utils/config_resolvers.py)`）：

```
num_frames = 1 + K_vf · D_vae · (N + M − 1) = 1 + 16·(N+M−1)
action_horizon = K_vf · D_vae · M = 16·M        ← 只与 M 有关
action_start_offset = K_vf · D_vae · (N − 1) = 16·(N−1)
num_latent_frames = N + M
```

典型配置 `N=2, M=2`：`num_frames=49, action_horizon=32, num_latent_frames=4`。

配置侧完全由 OmegaConf resolver 推导，dataset 侧做唯一真源校验：

- resolver 只算：`data.train.num_frames` = `[latent_window_to_num_frames](../../../src/fastwam/utils/config_resolvers.py)`(N,M,K_vf,4)、`data.train.action_horizon` = `[latent_window_to_action_horizon](../../../src/fastwam/utils/config_resolvers.py)`(M,K_vf,4)
- dataset 拒非法组合：`[RobotVideoDataset.__init_](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)_` L50-72 把 `(num_frames-1) % K_vf`、`(sampled_T-1) % D_vae`、`num_latent_frames > N`、`action_horizon == K_vf·D_vae·(num_latent_frames-N)`、可选 `num_denoise_latent_frames == num_latent_frames - N` 一次性 assert

### 2.2 时间轴语义

以 `N=2, M=2, K_vf=4, D_vae=4` 为例，LeRobot 滑窗拿到 raw step 0..48 共 49 步观察：


| Latent | 编码的 sampled 帧 | 覆盖的 raw steps      | 角色                         |
| ------ | ------------- | ------------------ | -------------------------- |
| 0      | 第 0 帧         | `{0}`              | clean anchor（过去的历史锚点）      |
| 1      | 第 1..4 帧      | `{4, 8, 12, 16}`   | clean anchor（"当前时刻"所在的时间块） |
| 2      | 第 5..8 帧      | `{20, 24, 28, 32}` | denoise target（未来第一块）      |
| 3      | 第 9..12 帧     | `{36, 40, 44, 48}` | denoise target（未来第二块）      |


动作窗口 `action[0..31]` 对应 raw step `[16..47]`，即 **"当前时刻 16 及其未来 31 步"**。

关键语义：**anchor 的最后一帧代表"现在"，其余 anchor 代表"过去"**。

### 2.3 因果 VAE 的角色

Wan2.2 VAE 是 causal 编码：

- Latent 0 = f(frame 0)
- Latent i (i≥1) = f(frame 0..4i) 但只接受从 frame `4(i−1)+1` 起的新信息

换言之，**每个非首 latent 在空间特征之外还压缩了 4 个 sampled 帧的动态**。配合 `K_vf=4`，每个 latent 覆盖 16 个 raw step 的时间窗，是一种"学习到的时间压缩"。

### 2.4 注意力侧如何区分 anchor 与 denoise

三处关键机制（`[wan_video_dit.py](../../../src/fastwam/models/wan22/wan_video_dit.py)` / `[fastwam.py](../../../src/fastwam/models/wan22/fastwam.py)`）：

- `[build_video_to_video_mask](../../../src/fastwam/models/wan22/wan_video_dit.py)`（`first_frame_causal` 模式，L472-507）：anchor tokens 不能看到 denoise tokens（`mask[:N·TPF, N·TPF:] = False`），保证 anchor 作为"已知历史"不被未来污染
- `[pre_dit](../../../src/fastwam/models/wan22/wan_video_dit.py)` 的 action cross-attn mask（L570-583）：`num_temporal_groups = f − N = M`（其中 `f` 是 latent frame 数）；只有 denoise latent 位置能 attend 到 action token，anchor 位置不参与 action 条件反馈。配套运行期 assert `action_emb.shape[1] % num_temporal_groups == 0` 作为唯一真源
- `[FastWAM.build_inputs](../../../src/fastwam/models/wan22/fastwam.py)` 中 `anchor_latents = input_latents[:, :, 0:N]`（L446）：每个扩散步都把 anchor 重置为干净值，保证 anchor 在所有时间步都表示 observed history 而非噪声

> v2.4 之前 `build_inputs` 用 `action_horizon % (num_frames-1) == 0` 检查训练输入；v2.4 改为 `action_horizon % denoise_sampled_transitions == 0`（其中 `denoise_sampled_transitions = sampled_T-1 − D_vae·(N-1)`），N=2 / M=2 下 `32 % 8 == 0` 通过，旧口径 `32 % 12` 非整除会被误拒。
> `_validate_forward_inputs` 的外层整除检查也被移除（L442-444 仅留注释），让"整除性"只由 `pre_dit` 与 `build_inputs` 两处负责，避免双份真源漂移。

---

## 3. Multi-Anchor 体现"记忆"吗？

### 3.1 "记忆"的不同定义下的答案


| 记忆类型                                   | 是否具备 | 说明                                                     |
| -------------------------------------- | ---- | ------------------------------------------------------ |
| **观察历史感知**（history-aware conditioning） | ✅    | 给定 N>1，策略的条件包含过去 `16(N−1)+1` 个 raw step 的观察            |
| **显式有限时间窗**（bounded sliding window）    | ✅    | 覆盖 raw step `[0, 16(N−1)]`，固定窗口                        |
| **Transformer 上下文式记忆**                 | ✅    | 每次推理重新编码 N 个 latent → KV-cache，作为 action 去噪的 context   |
| **跨推理调用的持久状态**                         | ❌    | 每个 `infer_action()` 调用独立重编码 anchor；模型内无持续 hidden state |
| **循环状态 / RNN 式记忆**                     | ❌    | 没有 recurrent connection                                |
| **无限历史 / episodic memory**             | ❌    | 窗口大小固定为 `4(N−1)+1` sampled 帧                           |


### 3.2 定性结论

**Multi-anchor 是 Transformer 式的 "有限滑窗上下文记忆"**：

- 策略从 Markovian（N=1）变成 **N 阶时间依赖**（N≥2）
- 记忆由 **外部 deque (`frame_history`) + VAE encode** 提供，不是模型内部的状态
- 记忆粒度受 causal VAE 约束：每个非首 anchor latent 压缩 4 个 sampled 帧，不能任意粒度回溯

### 3.3 能感知什么？

有了 N≥2 后，策略可从 anchor 中估计：

- **速度**：相邻 latent 之间的特征差（对二阶动力学系统尤为关键）
- **方向性**：区分"手臂正靠近目标"与"手臂正远离目标"这类单帧无法判别的状态
- **周期 / 节拍**：抓取前的预接触运动、夹爪开合过程

这些是 N=1 Markov 策略从单张 RGB 图像中原则上 **无法解出** 的信息（除非依赖 proprio，但 proprio 不含视觉场景动态）。

---

## 4. 与原始 Fast-WAM 的对比

### 4.1 表格总览


| 维度                         | 原始 Fast-WAM（N=1） | Multi-Anchor（N≥2）                              |
| -------------------------- | ---------------- | ---------------------------------------------- |
| 条件观察                       | 当前单帧             | N 个 anchor latent，覆盖 `16(N−1)+1` 个 raw step    |
| 策略类型                       | Markov（无记忆）      | N 阶依赖（滑窗记忆）                                    |
| `num_frames`               | 33               | `1 + 16(N+M−1)`（N=2 时 49）                      |
| `action_horizon`           | 32               | 32（M 不变时不变）                                    |
| `action` 起始 raw step       | 0（当前即起点）         | `16(N−1)`（历史尾 = 当前即起点）                         |
| VAE 输入帧数（训练）               | `K_vf·(M)+1 = 9` | `K_vf·(N+M−1)+1`（N=2 时 13）                     |
| Anchor 覆盖 raw 步数           | 1                | `16(N−1)+1`                                    |
| 推理 KV-cache prefill latent | 1                | N                                              |
| Eval 端帧缓冲                  | 单帧               | `deque(maxlen=1+16(N−1))` 的 raw 帧，按 `K_vf` 子采样 |
| `pretrained_norm_stats`    | baseline         | 改 N 时可复用；改 M 时必须重算                             |
| 训练样本尾部 pad 比例              | 低                | 随 N 增大显著升高                                     |


### 4.2 训练时的差异

- **观察视野**：baseline 每个 batch 样本仅供 1 帧图像（经 VAE）作为 clean conditioning；multi-anchor 供 N 帧，VAE 会在训练损失内看到"过去→未来"完整 `N+M` latent 序列
- **Video loss 覆盖面**：`_compute_video_loss_per_sample` 的 `num_excluded_anchor_steps=N`，两种模式都不对 anchor 做 MSE；multi-anchor 下 video loss 仍然只监督 denoise 段的 M 个 latent（与 baseline 结构同型）
- **Action cross-attn 分组数**：`num_temporal_groups = M`，两种模式恒为 2（当 M=2 时）；即 action 侧的注意力拓扑 **不** 因 N 而变，只是看到的 video KV 多了 N−1 个 anchor 组
- **训练样本量**：由于 `num_frames` 随 N 增大而增长，滑窗起始可用区间 `[0, episode_length − num_frames]` 缩小，有效样本数下降

### 4.3 推理时的差异

- **条件构造**：baseline `input_image` 是单张 tensor；multi-anchor 是 `list[Tensor]`，长度 `4(N−1)+1`，由 `[FastWAM._encode_multi_image_latents_tensor](../../../src/fastwam/models/wan22/fastwam.py)`（L304-346）合成一段"伪视频"供 causal VAE 编码出 N 个 latent。`FastWAM.infer()` 签名（L1213-1216）声明 `input_image: Union[torch.Tensor, list[torch.Tensor]]`，trainer.evaluate 与 eval/deploy 都走这条路径
- **延迟**（**未实测，以下为结构级估算**）：
  - VAE 编码：N=1 时 1 帧，N=2 时 5 帧；VAE 开销小（相对 DiT），可忽略
  - MoT KV-cache prefill：N 个 latent → `tokens_per_frame · N` 个 video token；token 数增加但 prefill 只跑一次
  - Action 去噪步数：完全不变
  - 总体：推理延迟估计增加约 10–20%（取决于 VAE 实现），相对原始 Fast-WAM 相对 imagine-then-execute "4× 提速" 的优势不受影响
  - **TODO**：在 `scripts/` 下补一个 `bench_multi_anchor.py` 基准脚本，实测 N∈{1,2,3} 的 step time / peak mem / infer latency，替换上述估算
- **评估帧缓冲**：
  - `[eval_libero_single.py](../../../experiments/libero/eval_libero_single.py)` L522-526：`frame_history_len = 1 + (4·(N-1)) · K_vf`（N=2 K_vf=4 时 17），`deque(maxlen=frame_history_len)` 维护原始帧流；episode 起点若帧数不足则 `appendleft(frame_history[0])` replicate 首帧填充
  - L292 `list(frame_history)[::K_vf]` 子采样得到 `4(N-1)+1` 帧（N=2 时 5 帧）
  - `[deploy_policy.py](../../../experiments/robotwin/fastwam_policy/deploy_policy.py)` L191-194 / L245-255 同理
- **行为语义**：原始 Fast-WAM 是"看到什么做什么"；multi-anchor 是"看到最近这段做什么"

### 4.4 不变的部分

- 整体训练范式依然是 Fast-WAM 论文的核心思想："video co-training at training, skip future generation at test"
- `infer_action()` 路径结构不变，依然是 video KV-cache prefill + action-only denoising
- 归一化、proprio、context 等输入不变（只是 proprio 被 dataset 右移以对齐 action 窗口）
- Flow-matching schedule、loss 权重、MoT 架构不变

### 4.5 v2.4.1 新增的运行期守门

为保证参数/数据/权重三方一致性，v2.4.1 在代码里加了 5 处轻量守门（详见 [action_v2.4.md §v2.4.1 补强](action_v2.4.md)）：


| 位置                                                                                                          | 检查对象                                                       | 信号                                                             |
| ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- |
| `[RobotVideoDataset.__init_](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)_` L98-141        | episode 长度 vs `num_frames + action_start_offset`           | 过短 episode 打 WARN，N>1 下末尾可能 pad 打 INFO                         |
| `[FastWAMProcessor.preprocess](../../../src/fastwam/datasets/lerobot/processors/fastwam_processor.py)` L274 | `sample["action"].shape[0] == num_action_steps`            | AssertionError                                                 |
| `[eval_libero_single.py](../../../experiments/libero/eval_libero_single.py)` L819-840                       | `EVALUATION.action_horizon` vs `data.train.action_horizon` | "overrides data.train.action_horizon" WARN                     |
| `[deploy_policy.py](../../../experiments/robotwin/fastwam_policy/deploy_policy.py)` L380-412                | 同上                                                         | 同上                                                             |
| `[FastWAM.load_checkpoint](../../../src/fastwam/models/wan22/fastwam.py)` L1285-1306                        | ckpt `num_anchor_frames` 字段 vs runtime N                   | "Checkpoint num_anchor_frames=... does NOT match runtime" WARN |


特别说明：这些都是日志 / assert，**不改变训练-推理数学语义**，只是把"错配"的失败时机从"训练 NaN / eval 全 0"前移到"dataset 初始化日志"，让迁移时能在几秒内看到问题。

---

## 5. 设计含义

### 5.1 这是什么角度的扩展

Multi-anchor **不改变 Fast-WAM 的主张**（video imagine 只在训练时有用），只是把 **"输入观察"从单帧扩展为短时序历史**。这与是否显式生成未来视频正交。

类比：

- **原始 Fast-WAM vs imagine-then-execute**：关于"是否需要显式生成未来视频"
- **Multi-anchor vs N=1**：关于"是否需要显式的历史上下文"

两者回答不同问题。完全可以把 multi-anchor 看作是给 Fast-WAM 的 **输入表示** 层做了扩展，与其视频建模哲学并不冲突。

### 5.2 何时应采用 multi-anchor

- 任务存在显著的 **部分可观测性**（遮挡、相机抖动）
- 动作依赖 **速度 / 加速度**（投掷、抓取接近期）
- 环境存在 **周期性** 或需要区分"趋近 vs 远离"的模式
- Proprio 缺失或不可靠

### 5.3 何时 N=1 已经足够

- 任务静态场景，当前帧信息充分（典型抓取 / 摆放）
- 重视推理延迟与训练数据利用率
- proprio 信号完整（已经隐式编码了短时历史）

---

## 6. 局限与 trade-off

1. **训练数据尾部浪费**：`num_frames` 随 N 线性增长，episode 尾部不足一个窗口的样本被 pad 替代，N=4 时 LIBERO 可能丢失 >10% 的有效滑窗起点
  v2.4.1 起 `RobotVideoDataset.__init_`_ 会主动打印 `min/mean/max` episode 长度，并对"<num_frames"的 episode 发 WARN，让使用者在训练开始前就能看到这个 trade-off
2. **时间粒度受 VAE 限制**：不能在 raw step 粒度回溯，只能在 `{0, 4, 8, ...}` 这类离散位置插入锚点；若要更细粒度历史，需要换非 causal VAE 或降低 `K_vf`
3. **固定窗口 = 固定记忆深度**：无法自适应地"忘记久远、保留新近"；与 RWKV / Mamba 类 linear-attention 的记忆不同
4. **推理端每步重复编码历史**：没有 KV-cache 在 env step 之间的复用；如果环境步执行频率 > 推理频率，会重复 VAE encode 部分帧（优化空间存在但尚未实现）
5. **跨 N 不能热插拔 ckpt**：`anchor/denoise mask` 与 `num_temporal_groups` 均依赖 N，不同 N 的权重虽能以 `strict=False` 加载但物理语义错位；v2.4.1 起 `load_checkpoint` 会 WARN。迁移到新 N 必须重训
6. **归一化统计近似不变**：action normalizer 的统计窗口不包含 `action_start_offset`，N 变化时在 episode-level stationary 假设下可复用，严格分布重合需重算（见 [action_v2.4.md §何时需要重算 pretrained_norm_stats](action_v2.4.md)）

---

## 7. 一句话总结

> Multi-anchor 通过在 Fast-WAM 的干净锚点机制上扩展到 N 帧，把 Markov 策略升级为 **N 阶滑窗上下文策略**，本质是给 WAM 加了一个"Transformer 式的有限时间窗记忆"，而不触动 Fast-WAM 关于"训练时需要 video co-training、测试时不需要未来想象"的核心主张。

---

## 8. 代码索引（与当前仓库对照）

### 配置 / 数据侧


| 能力                                                                            | 文件 / 行                                                                                                         |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| resolver：`num_frames = 1 + K_vf·D_vae·(N+M-1)`                                | `[config_resolvers.py:latent_window_to_num_frames](../../../src/fastwam/utils/config_resolvers.py)` L57-65     |
| resolver：`action_horizon = K_vf·D_vae·M`                                      | `[config_resolvers.py:latent_window_to_action_horizon](../../../src/fastwam/utils/config_resolvers.py)` L67-72 |
| LIBERO `data.train` resolver 接入                                               | `[configs/data/libero_2cam.yaml](../../../configs/data/libero_2cam.yaml)` L34-37                               |
| RoboTwin `data.train / data.val` 引用                                           | `[configs/data/robotwin.yaml](../../../configs/data/robotwin.yaml)` L24-27, L91-94                             |
| `BaseLerobotDataset` 的 `action_start_offset` 参数与 action `delta_timestamps` 右移 | `[base_lerobot_dataset.py](../../../src/fastwam/datasets/lerobot/base_lerobot_dataset.py)` L28, L88-97         |
| `RobotVideoDataset` 多锚点签名 + 一致性 assert                                        | `[robot_video_dataset.py](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)` L45-94                |
| `RobotVideoDataset._get()` proprio 右移切片 + pad 判定                              | `[robot_video_dataset.py](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)` L193-329              |
| Episode 长度诊断（v2.4.1）                                                          | `[robot_video_dataset.py](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)` L98-141               |


### 模型 / 训练侧


| 能力                                                       | 文件 / 行                                                                                                      |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| anchor 重置：`latents[:, :, 0:N] = anchor_latents`          | `[fastwam.py:build_inputs](../../../src/fastwam/models/wan22/fastwam.py)` L446                              |
| 整除检查：`action_horizon % denoise_sampled_transitions == 0` | `[fastwam.py:build_inputs](../../../src/fastwam/models/wan22/fastwam.py)` L392-409                          |
| 多帧 VAE 编码（推理）：`_encode_multi_image_latents_tensor`       | `[fastwam.py](../../../src/fastwam/models/wan22/fastwam.py)` L304-346                                       |
| `infer()` 签名 `Union[Tensor, list[Tensor]]`               | `[fastwam.py](../../../src/fastwam/models/wan22/fastwam.py)` L1213-1216                                     |
| video token mask（`first_frame_causal`）：anchor 屏蔽 denoise | `[wan_video_dit.py:build_video_to_video_mask](../../../src/fastwam/models/wan22/wan_video_dit.py)` L472-507 |
| action cross-attn：`num_temporal_groups = f - N`          | `[wan_video_dit.py:pre_dit](../../../src/fastwam/models/wan22/wan_video_dit.py)` L570-583                   |
| ckpt `num_anchor_frames` 持久化 / 兼容性检查（v2.4.1）             | `[fastwam.py:save_checkpoint / load_checkpoint](../../../src/fastwam/models/wan22/fastwam.py)` L1260-1309   |
| Trainer evaluate 多锚点路径                                   | `[trainer.py](../../../src/fastwam/trainer.py)` L416-430                                                    |


### 评估 / 部署侧


| 能力                                        | 文件 / 行                                                                                                                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `action_horizon` 优先级 + 覆盖/回退 WARN（v2.4.1） | `[eval_libero_single.py](../../../experiments/libero/eval_libero_single.py)` L815-840、`[deploy_policy.py](../../../experiments/robotwin/fastwam_policy/deploy_policy.py)` L380-412                     |
| frame history deque + 子采样                 | `[eval_libero_single.py](../../../experiments/libero/eval_libero_single.py)` L284-298, L522-526、`[deploy_policy.py](../../../experiments/robotwin/fastwam_policy/deploy_policy.py)` L191-194, L245-255 |
| `_get_num_video_frames` 推导 sampled_T      | `[eval_libero_single.py](../../../experiments/libero/eval_libero_single.py)` L280-281                                                                                                                  |


### Processor


| 能力                                               | 文件 / 行                                                                                                                                                                                                |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `num_action_steps` 参数 + 回退                       | `[base_processor.py](../../../src/fastwam/datasets/lerobot/processors/base_processor.py)` L39, L43                                                                                                    |
| preprocess 长度 assert（BaseProcessor 首层）           | `[base_processor.py](../../../src/fastwam/datasets/lerobot/processors/base_processor.py)` L232-234                                                                                                    |
| preprocess 长度 assert（FastWAMProcessor v2.4.1 补强） | `[fastwam_processor.py](../../../src/fastwam/datasets/lerobot/processors/fastwam_processor.py)` L274                                                                                                  |
| postprocess 死代码注释                                | `[base_processor.py](../../../src/fastwam/datasets/lerobot/processors/base_processor.py)` L266 / `[fastwam_processor.py](../../../src/fastwam/datasets/lerobot/processors/fastwam_processor.py)` L311 |


---

## 引用

- Yuan, T., Dong, Z., Liu, Y., Zhao, H. *Fast-WAM: Do World Action Models Need Test-time Future Imagination?* arXiv [2603.16666v2](https://arxiv.org/abs/2603.16666v2), 2026.
- 本仓库（multi-anchor 改动文档群）：
  - [plan_opus4.7_v2.4.md](plan_opus4.7_v2.4.md)：设计 + 实施蓝图 + 迁移 runbook
  - [action_v2.4.md](action_v2.4.md)：落地记录 + v2.4.1 补强清单 + 可调参数总览
- 核心源码（快速跳转，完整索引见 §8）：
  - [fastwam.py](../../../src/fastwam/models/wan22/fastwam.py)：`build_inputs` / `training_loss` / `infer_action` / `infer_joint` / `save_checkpoint` / `load_checkpoint`
  - [wan_video_dit.py](../../../src/fastwam/models/wan22/wan_video_dit.py)：`build_video_to_video_mask` / `pre_dit` 的 `num_temporal_groups`
  - [robot_video_dataset.py](../../../src/fastwam/datasets/lerobot/robot_video_dataset.py)：`action_start_offset` 与 proprio 右移切片 + episode 长度诊断

