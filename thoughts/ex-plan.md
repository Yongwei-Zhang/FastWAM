# ex-plan

## 2026-04-14

### 目标
基于 Fast-WAM 代码，先从最小改动出发，验证“短历史上下文 / memory 机制”是否能够提升长程操作表现，并逐步回答 memory 应该作用于表征还是动作这一核心问题。

### 当前实验原则
- 先做最小改动，再做显式 memory token
- 先验证“短历史 anchor 是否有效”，再讨论更复杂的 memory 表示
- 优先保留 Fast-WAM 高效推理的核心优势，不直接照搬 LingBot-VA 的完整 AR + FDM 推理框架

### 阶段一：短历史 anchor 实验
#### 核心想法
将 Fast-WAM 原本使用的“首帧 anchor”扩展为“最近几帧 history anchor”，在训练和推理时都将最近一段历史帧作为上下文输入，而不是只使用首帧。

#### 实验设置
- Baseline：首帧 anchor
- Variant-1：最近 1 帧 anchor
- Variant-2：最近 2 帧 anchor
- Variant-3：最近 4 帧 anchor

#### 主要考察
- 短历史上下文是否优于首帧单锚点
- 历史长度对性能与计算开销的影响
- 是否存在一个较优的 history window，用于平衡 cost 和 performance

### 阶段二：memory 可见性 / mask 消融
#### 前提
在阶段一选出效果最好的 history 长度后，再引入 memory 可见性设计。

#### 核心问题
memory 应该参与 world representation 的塑造，还是只服务 action generation？

#### 消融设置
- All-Attend：video 和 action 都可见 memory
- Action-Only：仅 action 可见 memory
- Video-Only：仅 video 可见 memory

#### 主要考察
- memory 影响 video branch 是否有帮助
- memory 只作用于 action 是否已经足够
- 哪一种 mask 设计最适合 MA-Fast-WAM

### memory 表示的实验顺序
#### Version 1
- memory = recent visual history
- 直接使用最近几帧 visual latent 作为上下文
- 不额外引入新的 memory token 类型

#### Version 2
- memory = recent visual history + recent action summary
- 在视觉历史基础上，增加近期 action chunk 信息

#### Version 3
- memory = compressed video-action history tokens
- 将交错的 video-action 历史压缩为显式 memory token

### 当前建议
- 第一版不要直接上“显式 memory token + 专门 memory encoder”
- 第一版优先做 recent visual history，降低实现复杂度
- 若阶段一无明显收益，则暂缓更复杂 memory 设计
- 若阶段一有效，再继续推进 mask 消融与显式 memory token

### 当前结论
- 你的两个方向都是合理的：
  1. 增加 memory / 设计 mask
  2. 用短历史帧替代首帧 anchor
- 但更稳的起点是：
  - 先做短历史 anchor
  - 再做 memory 可见性消融
  - 最后再考虑显式 memory token

### 后续可继续补充的内容
- 具体代码修改位置
- 数据组织方式
- 训练配置
- 推理配置
- benchmark 与评价指标
- 每轮实验结果记录
