from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .wan_video_dit import flash_attention, modulate, rope_apply
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError("`mixtures` must include both 'video' and 'action' experts.")

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info("Using gradient checkpointing for mixture attention. This will save memory but use more computation.")

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )
        
        logger.info(f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}")
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B")

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (base_mod + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask)

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """
        1、用 _split_modulation(block, t_mod) 得到 MSA/MLP 的 shift、scale、gate。
        2、modulate(norm1(x), shift_msa, scale_msa) 得到自注意力输入，再 q/k/v 投影与 norm_q/norm_k。
        3、对 q、k 施加 rope_apply(freqs)。
        4、返回 q,k,v、未改的 x 作 residual_x、以及 gate_msa 与 MLP 三调制量 和 checkpoint 标志，供后面 拼接混合注意力 与 _apply_expert_post_block 使用。
        """
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._split_modulation(block, t_mod)
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(getattr(expert, "use_gradient_checkpointing", False))
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """
        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.
        per-layer 指的是 MoT 里 video expert 的每一个 Transformer block，且 2 个 expert 的 layer 数相同
        在每一层都要进行混合注意力堆叠。每一层预填时在该 block 上算出一对 k/v 存进 cache

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].  # RoPE，管位置
            video_t_mod: Video time modulation tensor.  # 视频 expert 里各层 DiT block 用的「时间调制」张量，管扩散步时刻对整层行为的缩放/门控
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
            其中 Sv 指的是整段视频 Token 序列长度，等于：潜空间时间长度 f（num_anchor_frames）* 每帧空间 token 数 tokens_per_frame 组成
            tokens_per_frame = (H_lat//p_h) * (W_lat//p_w)，由 pre_dit 在 patchify 之前用 latent 的 H、W 与 patch_size 算出来

        计算得到的 kv_cache 会作为 forward_action_with_video_cache 的 video_kv_cache 参数
        在 infer_action（或同类路径）里 每一步动作去噪 时使用

        具体用法（每层）：
        动作 expert 在该层算出 q_action, k_action, v_action 后，把 预填的 k、v 当作 视频侧 的 key/value，
        与当前步的 k_action、v_action 在序列维上 拼接：k_cat = [k_video_cached, k_action]，v_cat 同理；
        再用 动作行的联合掩码 做 _mixed_attention，让 动作 token 的 query 同时 attend 到锚点视频的 K/V（缓存）和当前动作序列的 K/V。
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )

            # k：该层 video 自注意力里，K 投影 + norm_k + RoPE 后的 key，shape=[B, Sv, H*Dh]
            # v：该层 V 投影后的 value（不做 RoPE），shape=[B, Sv, H*Dh]
            # K 投影、V 投影 指的是 Transformer 自注意力里两条独立的线性变换；
            # K 决定 “和谁对齐”（与Query相乘得到相似度），V 决定 “对齐后取什么内容”
            # Sv：video_tokens.shape[1]，视频 token 数（与 video_attention_mask 的 Sv 一致）
            # H*Dh：num_heads * attn_head_dim，视频 expert 的注意力头维拼接
            kv_cache.append({"k": k, "v": v})

        return kv_cache

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError("MoT requires `action` expert for `forward_action_with_video_cache`.")
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )

        # Use the action query rows from the joint [video+action] mask.
        # 联合序列的 2 值矩阵，形状为 [S, S]，前 Sv 列对应视频，后 Sa 列对应动作；为 每个动作 Q 行 指定 能读哪些 K 列
        # 本函数里 Q 只有动作，因此 mask 只要后面的 Sa 行，但是列仍然取满为 S=Sv+Sa，与 K/V 的拼接长度对齐
        action_attention_mask = attention_mask[video_seq_len:total_seq_len, :total_seq_len]

        # self.mixtures 是 MoT 里按名字挂的 多个 DiT/Transformer expert 的 ModuleDict
        # 这里取出 处理动作 token 的那一套 blocks（与视频 expert 层数、头维一致）
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            # 从当前动作隐状态 只 算出 动作的 q_action（及 k_action、v_action）
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,  # 动作序列上 RoPE（旋转位置编码），给自注意力加上 「第几个动作步 / 序列位置」 信息
                t_mod=action_t_mod,  # AdaLN 调制量（供每层拆成 shift / scale / gate 等），管「当前是第几个扩散步、噪声多强」
            )
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )

            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

            # Mixed attention: action queries attend to cached video K/V plus current action K/V.
            # 混合注意力：query 只用动作侧：q_cat=q_action；key/value 用拼接后的 [视频缓存 | 当前动作]
            # 再用 action_attention_mask（联合掩码里 动作行）约束 “动作 Q 能看哪些 K/V 位置”，最后得到 mixed 进 post-block 更新 x
            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,  # q_action 只对 k_cat/v_cat 做注意力，action_attention_mask 约定了注意力的位置
            )
            # 在 自注意力已经产出 mixed_slice 之后，按层完成 DiT block 后半段 的包装函数
            # 在混合注意力之后，完成 本层 block 的「残差 + 可选 cross-attn + MLP」，得到 下一层 Transformer Block 用的隐状态 x
            # x 指的是当前层 Block 输出之后的动作 Token 隐向量，[B, Sa, D]（D 为 hidden dim），算完之后再传入下一层，得到Q/K/V和混合注意力
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )

        return x

    # self.mot() 调用时的函数
    def forward(
        self,
        embeds_all: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: Dict[str, torch.Tensor],
        context_all: Dict[str, Optional[dict]],
        t_mod_all: Dict[str, torch.Tensor],
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}")
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}")

        # 为各专家（video、action expert）token建立逐层更新的工作副本，避免写回时改动入参 embeds_all。
        tokens_all = {k: v for k, v in embeds_all.items()}  # 新建 dict，键与张量引用同 embeds_all

        # MoT 逐层：各路先算 Q/K/V 并缓存 post 参数，再拼接做混合注意力，再按长度切回并写回各专家 token。
        # mot 按层循环，mot 发生在每一层。2 个 expert 共享层号，先各算本层 Q/K/V，序列维拼接后做一次联合注意力，再拆开做各自 后半段（post），然后进入 下一层
        for layer_idx in range(self.num_layers):  # 遍历当前堆叠层索引
            q_chunks = []  # 本层各专家 Q 列表
            k_chunks = []  # 本层各专家 K 列表
            v_chunks = []  # 本层各专家 V 列表
            cached = {}  # 各专家该层 block 与调制量，供 attention 后 post 使用
            seq_lens = []  # 各专家序列长度，供 mixed 切分

            # 按 expert_order 依次取该层 block，用当前 token 与 RoPE、时间调制构造注意力输入。
            for name in self.expert_order:  # 2 个 name：video 和 action
                expert = self.mixtures[name]  # 名为 name 的专家子模块，按 name 对应的整条专家 DiT
                block = expert.blocks[layer_idx]  # 该专家第 layer_idx 个 DiT block
                x = tokens_all[name]  # 该专家本层输入隐状态，[B, S, D]，其中 S 为 S_v 或者 S_a
                freqs = freqs_all[name]  # 该专家 RoPE 频率表，作用在 Q/K 上，提供位置编码
                t_mod = t_mod_all[name]  # 该专家时间步调制向量，与 block.modulation 相加后拆成 shift/scale/gate。用于 AdaLN 式调制：norm1 后自注意力支路、以及 norm2 后 MLP

                (
                    q,  # [B, S, H*Dh]，S 表示该路的 token 数，H 是头数，Dh 表示每个头的维度
                    k,
                    v,
                    residual_x,  # 进入该 block 的 x 原样保留，[B, S, D]
                    gate_msa,  # 多头自注意力门控，对子层输出做逐维加权再加回，存储的是权重，不是“做不做”的 bool 变量
                    shift_mlp,  # MLP 前对 norm2 做 modulate(..., shift, scale) 的平移/缩放
                    scale_mlp,
                    gate_mlp,  # 同样一套：x + gate * residual，用在 Block 中 MLP 那一个子层
                    use_gradient_checkpointing,  # 是否对该 expert 在「混合注意力之后」的那段 post（投影 o + gate + 可选 cross-attn + MLP）做梯度检查点
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )  # 投影得 Q/K/V，并拆出残差与 AdaLN 等供后续使用

                q_chunks.append(q)  # 追加该专家 Q
                k_chunks.append(k)  # 追加该专家 K
                v_chunks.append(v)  # 追加该专家 V
                seq_lens.append(x.shape[1])  # 记录该路 token 数
                # cached：把「混合注意力之前已算好、但要等混合完才能用」的 per-expert 状态存起来，供切分 mixed 后做 post
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,  # 如果为 True，则少存中间激活、省显存
                }  # 缓存 post 与 checkpoint 开关

            # 在序列维拼接各路 Q/K/V，总长度与 attention_mask 行/列一致。
            q_cat = torch.cat(q_chunks, dim=1)  # [B, S_v+S_a, H*Dh]，与传入的 attention_mask 的维度对应：[S_v + S_a, S_v + S_a]
            k_cat = torch.cat(k_chunks, dim=1)  # 同上
            v_cat = torch.cat(v_chunks, dim=1)  # 同上

            total_seq = q_cat.shape[1]  # 拼接后序列总长
            if attention_mask.shape[0] != total_seq:  # 掩码边长须等于总长
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )  # 不一致则无法对齐注意力

            # 对 q_cat, k_cat, v_cat 调 flash_attention，在总长 S_v+S_a 上做一次注意力
            # 符号 o：SelfAttention 里的 输出线性层 block.self_attn.o，把 注意力输出 [B,S,H*Dh] 投回 隐空间维度 [B,S,D]
            # gate_msa：由 t_mod + block.modulation 拆出的 第 3 个量，喂给 GateModule：x = residual_x + gate_msa * branch_out，控制自注意力分支加多少到残差上（msa：多头自注意力）
            mixed = self._mixed_attention(q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask)  # 单次混合自注意力输出

            start = 0  # mixed 上切片起点
            # 按 expert_order 与 seq_lens 切回各段，再过各专家 post（gate + 可选的cross + MLP）。
            for name, seq_len in zip(self.expert_order, seq_lens):
                end = start + seq_len  # 当前专家段终点
                mixed_slice = mixed[:, start:end, :]  # 该专家对应的 mixed 子序列
                cached_expert = cached[name]  # 取该专家本层缓存
                block = cached_expert["block"]  # 当前 DiT block
                # 如果传入了参数context_all，那么就需要在 post 里做 cross-attn
                context_payload = context_all.get(name)  # context_all 里 expert 对应的可选字典，给 cross-attention 用。context 是条件序列，用作K/V，形状一般为 [B, L, D]

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert["use_gradient_checkpointing"],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )  # 得到该专家更新后的 token

                tokens_all[name] = updated_tokens  # 写回，供下一层或返回
                start = end  # 下一路专家从 end 起切

        return tokens_all
