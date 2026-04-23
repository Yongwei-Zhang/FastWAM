# multi_anchor-plan-v1.1

## 核心想法
### 目标
基于 Fast-WAM 代码，先从最小改动出发，验证“短历史上下文 / memory 机制”是否能够提升长程操作表现，并逐步回答 memory 应该作用于表征还是动作这一核心问题。

具体地，将 Fast-WAM 原本使用的“首帧 anchor”扩展为“最近几帧 history anchor”，在训练和推理时都将最近一段历史帧作为上下文输入，而不是只使用首帧。

### 实验设置
- Baseline：首帧 anchor
- Variant-1：最近 1 帧 anchor
- Variant-2：最近 2 帧 anchor

### 主要考察
短历史上下文是否优于首帧单锚点
历史长度对性能与计算开销的影响
是否存在一个较优的 history window，用于平衡 cost 和 performance

## 概述

将 FastWAM 的单首帧 anchor conditioning 扩展为可配置的 `num_anchor_frames`，支持多帧历史作为条件输入。在 num_frames  等相关参数不变的情况下，`num_anchor_frames=1` 时行为与原代码完全一致。
