# FastWAM 全库 Review 链条

## 起点

先从训练入口往下顺，不要先扎进 `mot.py`。

推荐起手顺序：

1. 配置装配
2. 训练入口
3. 数据集与采样
4. 输入构建 + VAE 编码
5. Token 组织
6. 注意力掩码
7. MoT 联合前向
8. Loss 与训练循环
9. 推理路径
10. 模型变体差异

---

## Review 主链

### 1. 配置装配

先看任务配置怎样把 model/data/train 串起来。

- `configs/task/libero_uncond_2cam224_1e-4.yaml`
- `configs/model/fastwam.yaml`
- `configs/data/libero_2cam.yaml`

这一层先回答 6 个问题：

- 当前实验实际用的是哪个 model factory
- 当前数据窗口长度是多少
- action 和 video 的采样倍率是多少
- `num_anchor_frames` 是多少
- 视频/动作分支 hidden dim、layers、heads 各是多少
- 训练调度、loss 权重、eval/save 频率各是多少

重点核对：

- `num_frames=33`
- `action_video_freq_ratio=4`
- `num_anchor_frames=1`
- video branch 和 action branch 是否都是 30 层
- 当前任务是否走 `FastWAM` 而不是 `FastWAMJoint` / `FastWAMIDM`

### 2. 训练入口

顺着入口确认对象实际怎么实例化。

- `scripts/train.py`
- `src/fastwam/runtime.py`

这里先看两件事：

- Hydra 最终调用的是哪个 `create_*`
- `build_datasets` 和 `Wan22Trainer` 的接线方式

Review 问题：

- `instantiate(cfg.model)` 最终落到哪个类
- `instantiate(cfg.data.train)` 返回的 dataset 是什么
- `val` 没单配时是否直接复用 train dataset
- mixed precision / device 在哪里归一化

### 3. 数据处理与采样

这是第一优先级模块，先把 sample 结构看透。

- `src/fastwam/datasets/lerobot/robot_video_dataset.py`
- `src/fastwam/utils/samplers.py`

按这个顺序看：

1. `RobotVideoDataset.__init__`
2. `RobotVideoDataset._get`
3. `RobotVideoDataset._get_cached_text_context`
4. `RobotVideoDataset.__getitem__`
5. `ResumableEpochSampler.__iter__`

这一层要回答：

- 原始窗口 `num_frames=33` 如何变成 video frames 和 action horizon
- 多相机如何拼接
- video / action / proprio / context 的最终 shape 是什么
- padding 标记是否跟主张量对齐
- dataloader resume 时 sampler 是否会跳过已消费 batch

必须核的几个不变量：

- `(num_frames - 1) % action_video_freq_ratio == 0`
- `((num_frames - 1) // action_video_freq_ratio) % 4 == 0`
- video 实际子采样索引来自 `video_sample_indices`
- `proprio = sample["proprio"][:-1]`，它和 action 对齐，不和 video 对齐
- `context_mask` 从缓存读出后被强制改成全 1，这里要确认是不是有意设计
- `__getitem__` 失败时会随机回退样本，这会不会掩盖脏数据

建议你在这一步记一张 shape 表：

- `video`
- `action`
- `proprio`
- `context`
- `context_mask`
- `image_is_pad`
- `action_is_pad`
- `proprio_is_pad`

### 4. 输入构建 + VAE 编码

数据之后不要直接看 loss，先看模型如何把 sample 变成训练输入。

- `src/fastwam/models/wan22/fastwam.py`

按这个顺序看：

1. `_encode_video_latents`
2. `build_inputs`

这一层要回答：

- raw video 何时搬到 device
- VAE 时间下采样后 latent frame 数是多少
- anchor latent 是怎么切出来的
- proprio 取的是哪一个时刻
- `context/context_mask` 何时转 dtype

必须核的几个不变量：

- `video.shape == [B, 3, T, H, W]`
- `T % 4 == 1`
- `H` 和 `W` 都必须能被 16 整除
- `action.shape[1] % (T - 1) == 0`
- 开了 `fuse_vae_embedding_in_latents` 时，`num_anchor_frames < num_latent_frames`

这里建议重点盯住时间对齐：

- video frame index
- latent frame index
- raw action step index
- proprio index

这四个索引如果没对齐，后面所有 joint modeling 都会偏。

### 5. Token 组织

这一步拆成 video expert 和 action expert 两边看。

- `src/fastwam/models/wan22/wan_video_dit.py`
- `src/fastwam/models/wan22/action_dit.py`

先看：

1. `WanVideoDiT.pre_dit`
2. `ActionDiT.pre_dit`

这一层要回答：

- latent/video token 是怎么 patchify 展平的
- action token 是怎么从 action dim 投到 hidden dim 的
- timestep 是按样本共享还是按 token 分配
- video 用的是 3D RoPE 还是 1D RoPE
- context mask 在 video/action 两支里分别是什么形状

必须核的几个不变量：

- video `tokens_per_frame` 是否等于 patchify 后每帧 token 数
- `seperated_timestep + fuse_vae_embedding_in_latents` 路径里，anchor token 的 timestep 是否被置 0
- `ActionDiT` 的 `t_mod` 是否整条 action 序列共享
- `action_conditioned=false` 时，video context 里不应混入 action

### 6. 注意力掩码

mask 要单独审，不要夹在前向里顺手看。

- `src/fastwam/models/wan22/wan_video_dit.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`

先看：

1. `WanVideoDiT.build_video_to_video_mask`
2. `FastWAM._build_mot_attention_mask`
3. `FastWAMJoint._build_mot_attention_mask`

这一层要回答：

- video-to-video 是双向、逐帧因果，还是 first-frame causal
- action 是否只能看 anchor video tokens
- joint 版本为什么允许 action 看所有 video tokens

必须核的几个不变量：

- `video_seq_len = num_latent_frames * tokens_per_frame`
- `anchor_tokens = num_anchor_frames * video_tokens_per_frame`
- base `FastWAM` 里 video 不能看 action
- base `FastWAM` 里 action 只能看 anchor video
- `FastWAMJoint` 的 mask 语义与 `FastWAM` 明确不同

建议这里手动画一次四象限矩阵：

- video -> video
- video -> action
- action -> video
- action -> action

### 7. MoT 前向

等 mask 看清楚，再进 `mot.py`。

- `src/fastwam/models/wan22/mot.py`

按这个顺序看：

1. `forward`
2. `prefill_video_cache`
3. `forward_action_with_video_cache`

这一层要回答：

- 两个 expert 的 Q/K/V 在哪里拼接
- joint attention 是每层一次，还是多次
- split 回各 expert 后又做了什么
- 推理时 video cache 如何复用到 action 分支

必须核的几个不变量：

- `attention_mask` 是 `[Sv+Sa, Sv+Sa]`
- `prefill_video_cache` 只缓存每层 video 的 `k/v`
- action 推理阶段仍然每步重算 action 的 q/k/v
- 训练路径和推理 cache 路径在注意力语义上要一致

### 8. Loss 与训练流程

MoT 看完再回到训练，不然你会不知道 loss 的输入是怎么来的。

- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/trainer.py`

按这个顺序看：

1. `FastWAM.training_loss`
2. `Wan22Trainer._build_loader`
3. `Wan22Trainer._estimate_total_train_steps`
4. `Wan22Trainer.train`

这一层要回答：

- video / action 两个 diffusion timestep 是否独立采样
- anchor latent 在 video noising 前后是怎么处理的
- video loss 是否排除了 anchor 帧
- action/video pad mask 是否只影响 loss，不影响输入本身
- optimizer/scheduler/grad accumulation 的步进点在哪里

必须核的几个不变量：

- video 和 action scheduler 分开采样 `t`
- `anchor_latents` 会覆盖 noised video latents 的前几帧
- video loss 对 anchor 部分不监督
- `global_step` 只在 `sync_gradients` 时增长
- save/eval/log 频率都以 optimizer step 为单位，不是 dataloader step

### 9. 推理路径

训练看完以后再补推理，否则容易把两条路径混淆。

- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/fastwam_joint.py`
- `src/fastwam/models/wan22/fastwam_idm.py`

优先顺序：

1. `FastWAM.infer_action`
2. `FastWAM.infer_joint`
3. `FastWAMJoint.infer_action`
4. `FastWAMJoint.infer_joint`
5. `FastWAMIDM.training_loss`
6. `FastWAMIDM.infer_action`
7. `FastWAMIDM.infer_joint`

这一层要回答：

- action-only 推理是否复用 video cache
- joint 推理时 video/action 是否同步去噪
- IDM 变体相比 base / joint 多了哪一段 teacher-forcing 逻辑

### 10. 最后补外围模块

主链看完再补，不要反过来。

外围建议顺序：

1. `src/fastwam/datasets/lerobot/processors/fastwam_processor.py`
2. `src/fastwam/datasets/lerobot/base_lerobot_dataset.py`
3. `scripts/precompute_text_embeds.py`
4. `experiments/libero/*`
5. `experiments/robotwin/*`

这一层主要补两件事：

- 数据源如何被 processor 规整成当前 sample schema
- 训练好的模型在评测脚本里如何被调用

---

## 推荐 Review 节奏

### Day 1

- 配置装配
- 训练入口
- 数据处理与采样

目标：

- 画出 sample schema
- 画出时间轴对应关系

### Day 2

- 输入构建 + VAE 编码
- Token 组织
- 注意力掩码

目标：

- 画出 `video/action/context` 三路 token 图
- 画出 joint attention mask 图

### Day 3

- MoT 前向
- Loss 与训练流程
- 推理路径与模型变体

目标：

- 画出训练前向图
- 画出 action-only 推理 cache 图

---

## 最小 Review 清单

- 配置层参数有没有和代码假设冲突
- 数据采样后的时间轴是否自洽
- action / proprio / video 是否严格对齐
- VAE 时间下采样后 anchor 定义是否还成立
- video token / action token 长度是否符合 mask 构造
- joint mask 是否严格符合设计语义
- `FastWAM` / `FastWAMJoint` / `FastWAMIDM` 差异是否只体现在预期位置
- 训练时 loss、pad mask、scheduler、global step 是否一致
- 推理 cache 路径是否和训练语义一致

---

## 起手文件顺序

如果你现在立刻开始，我建议直接按这个文件顺序读：

1. `configs/task/libero_uncond_2cam224_1e-4.yaml`
2. `configs/model/fastwam.yaml`
3. `configs/data/libero_2cam.yaml`
4. `scripts/train.py`
5. `src/fastwam/runtime.py`
6. `src/fastwam/datasets/lerobot/robot_video_dataset.py`
7. `src/fastwam/utils/samplers.py`
8. `src/fastwam/models/wan22/fastwam.py`
9. `src/fastwam/models/wan22/wan_video_dit.py`
10. `src/fastwam/models/wan22/action_dit.py`
11. `src/fastwam/models/wan22/mot.py`
12. `src/fastwam/trainer.py`
13. `src/fastwam/models/wan22/fastwam_joint.py`
14. `src/fastwam/models/wan22/fastwam_idm.py`
