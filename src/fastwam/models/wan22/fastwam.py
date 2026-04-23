from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        num_anchor_frames: int = 1,
    ):
        super().__init__()
        self.num_anchor_frames = int(num_anchor_frames)
        if self.num_anchor_frames < 1:
            raise ValueError(f"num_anchor_frames must be >= 1, got {self.num_anchor_frames}")
        self.video_expert = video_expert  # 来自 Wan 的 WanVideoDiT（components.dit）
        self.action_expert = action_expert

        # video_expert 和 action_expert，被收进 mot 中（mot.py中），作为 mot 的子模块
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot  # 把 MoT 赋给 dit，供 Trainer 使用

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)  # 仅在配置了 proprio_dim 时创建
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        num_anchor_frames: int = 1,
    ):
        """从 Wan2.2 TI2V 预训练加载组件，组装 VideoDiT + ActionDiT + MoT，返回 ``FastWAM`` 实例。
        TI2V 在 Wan 这条产品线里指 Text-and-Image-to-Video：同时用文本提示 + 参考图像（常见为首帧/条件图）生成视频 的那类模型
        T2V 偏纯文生视频，I2V 偏纯图生视频；TI2V 表示一条管线（pipeline，即数据从输入到输出固定串联起来的一整条处理链条）里两种条件都支持、可一起用
        流程：拉取 Wan 组件 → 构建 ``ActionDiT`` 并校验与 VideoDiT 层数/头维一致 → 封装 ``MoT`` → ``cls(...)`` 构造 ``FastWAM`` 并记录 ``model_paths``。
        """
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        # 返回的结果包括：``dit``、``vae``、``text_encoder``/``tokenizer``（可 ``None``）及各自路径
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit

        # ActionDiT 不是 Wan 仓库里的现成子模块，而是 FastWAM 里单独定义的 动作分支，不应塞进「只拉 Wan 官方组件」的函数里
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        # MoT：VideoDiT 与 ActionDiT 层数/头数/头维一致时逐层对齐；
        # 各专家对自己的 token 算 Q/K/V，在序列维拼成 [video||action] 做一次混合自注意力（flash-attn），再按长度切回两路，各自接 cross-attn（文本等）与 FFN；
        # 两路在同一深度通过该次注意力相互可见。
        # `mot_checkpoint_mixed_attn` 为 True 时仅对这一步混合注意力（不用单独对Token计算QKV）做 gradient checkpoint 以省显存。
        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        # 在 FastWAM 里定义时 cls 即 FastWAM 类本身，因此 cls=FastWAM，从而得到训练/推理里用的整条 FastWAM 模型
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            num_anchor_frames=num_anchor_frames,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        """在线编码文本（若 load_text_encoder=True）：encode_prompt 里把 token 喂进 UMT5 得到 context；
        LIBERO 等任务常 load_text_encoder=false，数据侧直接给预计算 context
        """
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    @torch.no_grad()
    def _encode_multi_image_latents_tensor(
        self,
        input_images: list[torch.Tensor],
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ) -> torch.Tensor:
        """Encode consecutive observation frames via causal VAE.

        Args:
            input_images: Exactly ``4*(num_anchor_frames-1)+1`` consecutive
                observation frames (each [1,3,H,W] or [3,H,W]).  The caller
                is responsible for collecting real sequential frames so the
                causal VAE sees genuine temporal variation, matching the
                training-time encoding of multi-frame video segments.

        Returns:
            Latent tensor [1, C, num_anchor_frames, H_lat, W_lat].
        """
        if len(input_images) == 1:
            return self._encode_input_image_latents_tensor(
                input_images[0], tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
        expected_t = 4 * (self.num_anchor_frames - 1) + 1
        if len(input_images) != expected_t:
            raise ValueError(
                f"_encode_multi_image_latents_tensor expects "
                f"4*(num_anchor_frames-1)+1 = {expected_t} frames, "
                f"got {len(input_images)}"
            )
        # Concatenate real consecutive frames into video: [3, T, H, W]
        frame_slices = []
        for img in input_images:
            frame = img.to(device=self.device)
            if frame.ndim == 4:
                frame = frame[0]  # [1,3,H,W] → [3,H,W]
            frame_slices.append(frame.unsqueeze(1))  # [3, 1, H, W]
        video = torch.cat(frame_slices, dim=1)  # [3, T, H, W]
        z = self.vae.encode(
            [video], device=self.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        """得到供后续前向使用的字典（潜变量、条件、动作等）。
        """
        # sample 必备字段（build_inputs）：video、action、context、context_mask；可选字段：proprio、action_is_pad、image_is_pad
        video = sample["video"]  # [B,3,T,H,W]，多相机拼成的一段视频张量（已归一化等到模型期望范围），T表示像素帧数
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]  # 不直接放字符串；context 是 T5 等预计算文本嵌入 [B,L,D]
        context_mask = sample["context_mask"]  # [B,L] 标哪些位置是真实 token；L表示条件序列长度，即每条样本有多少个「条件 token」位置；D是每个条件位置的向量维度
        proprio = sample.get("proprio", None)  # 仅当 self.proprio_encoder is not None 时必需，[B,T,d]，函数内只用 [:,0,:]
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]  # 形状 [B, T_action, a_dim]；T_action 必须与去噪段 sampled transitions 整除对齐
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        # Multi-anchor: 检查只约束去噪段 sampled transitions。
        # `num_frames` here is `video.shape[2]`, which equals the sampled video length `sampled_T`.
        vae_t_factor = int(self.vae.temporal_downsample_factor)
        denoise_sampled_transitions = (num_frames - 1) - vae_t_factor * (self.num_anchor_frames - 1)
        if denoise_sampled_transitions <= 0:
            raise ValueError(
                f"Denoise sampled transitions must be > 0, got {denoise_sampled_transitions} "
                f"(sampled_T={num_frames}, num_anchor_frames={self.num_anchor_frames})"
            )
        if action_horizon % denoise_sampled_transitions != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by denoise sampled transitions "
                f"({denoise_sampled_transitions}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)  # 不必需，[B,T_action]
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)  # 不必需，[B,T]，与像素帧数 T 对齐
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)  # video 只做 VAE 编码得到 input_latents
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        anchor_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            if self.num_anchor_frames >= input_latents.shape[2]:
                raise ValueError(
                    f"num_anchor_frames ({self.num_anchor_frames}) must be < "
                    f"num_latent_frames ({input_latents.shape[2]})"
                )
            # 从 潜变量时间维前 num_anchor_frames 帧切出；即 潜变量时间维上的连续前 N 帧，不是事后把 N 个独立 latent 张量再拼起来。
            # 如果没有进入上面的 if 条件，不在此处做潜空间 anchor 切片；多锚点其它逻辑在后续 training_loss / pre_dit 等路径上处理
            anchor_latents = input_latents[:, :, 0:self.num_anchor_frames]
            fuse_flag = True  # 同时 fuse_vae_embedding_in_latents 置为 True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            # Dataset already right-shifts `proprio` so step 0 corresponds to raw step
            # `4*action_video_freq_ratio*(num_anchor_frames-1)`; take the first step directly.
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # 以下是出参数
        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,  # [B, C_lat, T_lat, H_lat, W_lat]，后4个分别为（潜空间内的）：通道数（与配置里 VAE / DiT in_dim 一致，如 48）、长度、高、宽
            "anchor_latents": anchor_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,  # [B, T_action, a_dim]，T_action 表示动作序列长度，dtype=self.torch_dtype（上面数据也是）
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
        num_anchor_frames: int = 1,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
            num_anchor_frames=num_anchor_frames,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> anchor-frame video tokens only
        anchor_tokens = min(num_anchor_frames * video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :anchor_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        num_excluded_anchor_steps: int = 0,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        full_video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        video_is_pad = full_video_is_pad[:, num_excluded_anchor_steps:]

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        """依赖 build_inputs；前向结构与下面的 _predict_joint_noise 同构（pre_dit / mot / post_dit）
        """
        # 从 sample 组训练批：潜变量、条件上下文、动作与 padding 掩码。
        inputs = self.build_inputs(sample, tiled=tiled)  # 解析并张量化 batch
        input_latents = inputs["input_latents"]  # 视频潜变量，[B, C_lat, T_lat, H_lat, W_lat]，C_lat：VAE 潜空间通道；T_lat：潜空间时间长度（帧）；H_lat, W_lat：潜空间高宽
        batch_size = input_latents.shape[0]  # 当前批大小

        # 关于“条件嵌入”的条件：除去待去噪的量之外，告诉模型任务/场景是什么，并参与 cross-attn 那一侧的信息
        context = inputs["context"]  # 条件嵌入（如文本，也有其他条件信息），[B, L, D_ctx]
        # 视频、动作 DiT 里条件分支共用该掩码
        context_mask = inputs["context_mask"]  # 条件序列有效位，[B, L]，context 对应的有效位掩码，用于 cross-attention 里屏蔽掉无效/pad 的 token。

        action = inputs["action"]  # 动作轨迹，[B, T_a, A]，T_a：动作步数（时间步）；A：动作维度（各关节/末端等）
        action_is_pad = inputs["action_is_pad"]  # 动作步无效标记，用于 动作损失按步加权，为 pad 步则不算入平均 loss，[B, T_a]
        # T_lat 与像素 T_pix 由 VAE 时间下采样联系；T_a 须被 (T_pix−1) 整除
        image_is_pad = inputs["image_is_pad"]  # 视频帧无效标记，用于 视频重建/扩散损失按帧加权，[B, T_pix]，T_pix：像素视频帧数，与 sample["video"] 的 T 一致（[B,3,T,H,W] 里的 T）

        # 视频分支扩散训练：加噪前向 + 回归目标
        noise_video = torch.randn_like(input_latents)  # 纯高斯噪声，[B, C_lat, T_lat, H_lat, W_lat]
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )  # 每个样本随机采样的扩散步 t，维度为 [B]，每个样本一个标量时间步，内部再除以 num_train_timesteps 得到 sigma（位于0-1的时间参数）
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)  # 加噪后的视频潜变量，作为 DiT 输入，文中 5 式，[B, C_lat, T_lat, H_lat, W_lat]
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)  # 6 式中的 epsl-y，[B, C_lat, T_lat, H_lat, W_lat]，与 input_latents 相同

        # 若开启 anchor 帧与潜变量融合（开启了 fuse，且从 VAE 切出了 anchor 潜变量）：anchor 帧不加噪，保持干净条件
        # anchor_latents 为 None 时（例如未开 fuse_vae_embedding_in_latents），不会走这里的干净写回，
        # 这个时候多锚点通过 num_anchor_frames 与注意力掩码等起作用，相较于这里的 fuse+干净写回 是不同的路径
        if inputs["anchor_latents"] is not None:
            n_anchor = inputs["anchor_latents"].shape[2]
            latents[:, :, 0:n_anchor] = inputs["anchor_latents"]  # 前 n_anchor 个 latent 时间步始终是未加噪的 anchor，后面时间步仍是加噪后的量

        # 动作分支扩散训练：与视频分支类似，在动作空间加噪、采样 t、构造监督目标
        noise_action = torch.randn_like(action)  # 与 action 同形状的高斯噪声，[B, T_a, A]
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )  # 每个样本随机采样的动作扩散步 t，维度=样本数，即 [B]
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)  # t 时刻加噪后的动作，作为动作 DiT 输入，[B, T_a, A]
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)  # 与 train_action_scheduler 一致的回归目标，[B, T_a, A]

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            num_anchor_frames=self.num_anchor_frames,
        )

        action_pre = self.action_expert.pre_dit(  # 动作分支编码
            action_tokens=noisy_action,  # 加噪后的动作，[B, T_a, A]
            timestep=timestep_action,  # 动作分支扩散步，[B]
            context=context,  # 文本条件，[B, L，D_ctx]
            context_mask=context_mask,  # 条件序列掩码，[B, L]
        )  # 包含 action tokens 等信息的字典

        # 从 pre_dit 取双流 token，按帧宽构造联合注意力掩码，再经 MoT 做跨模态 Transformer。
        video_tokens = video_pre["tokens"]  # 对 加噪潜变量 做 patch 嵌入 再展平得到的 视频 token 序列，[B, S_v, D]，S_v = f·h·w（潜空间上 f×高×宽 的 patch 数），D = hidden_dim
        action_tokens = action_pre["tokens"]  # 对 加噪动作 经 action_encoder 得到的 动作 token 序列，供 MoT 动作支路用。[B, S_a, D]，S_a = T_a（动作序列长度，与 action.shape[1] 一致），D 与视频侧 相同（同一 MoT 对齐）。

        # 按帧宽构造联合注意力掩码
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
            num_anchor_frames=self.num_anchor_frames,
        )

        # mot：Video token 和 Action token 先拼接（联合）在一起做联合自注意力，再各自接条件交叉注意力（cross-atten）
        # tokens_out["video"]： 经过 MoT 全部层 后的 视频侧隐状态序列，[B, S_v, D]，与送入时的 video_tokens 同形
        # tokens_out["action"]：动作侧 经 MoT 更新后的 隐状态序列，[B, S_a, D]，与送入时的 action_tokens 同形
        tokens_out = self.mot(  # 联合 DiT（双流注意力）
            embeds_all={
                "video": video_tokens,  # 视频嵌入，一条 token 序列 [B, S_v, D]
                "action": action_tokens,  # 动作嵌入，一条 token 序列 [B, S_a, D]
            },
            attention_mask=attention_mask,  # 联合掩码：约束 video↔video、action↔action、action↔video 等谁能看谁，[S_v + S_a, S_v + S_a]
            freqs_all={
                "video": video_pre["freqs"],  # 视频 token 的 RoPE（旋转位置编码） 相位参数；使注意力带相对位置的信息
                "action": action_pre["freqs"],  # 动作 token 的 RoPE 频率表，作用在已投影的 Q 和 K 上（经过了矩阵相乘的 Q 和 K 上）
            },
            context_all={
                "video": {
                    "context": video_pre["context"],  # [B, L_cond, D]，L_cond 是文本条件序列，可能包含动作；作为 K/V，供 每个视频 token 做 cross-attn
                    "mask": video_pre["context_mask"],  # [B, S_v, L_cond]，每个 视频 query 位置 对 各条件 key 是否可见
                },
                "action": {
                    "context": action_pre["context"],  # [B, L_text, D]，动作 token 的 cross-attn K/V
                    "mask": action_pre["context_mask"],  # [B, S_a, L_text]，文本 pad 等屏蔽

                },
            },
            t_mod_all={  # t_mod = 由 扩散步 timestep_* 经 time_projection 得到的 AdaLN（自适应层归一化） 用调制（modulate）量（拆成 6 份：与 DiT block 里 MSA/MLP 的 shift、scale、gate 等对应）
                "video": video_pre["t_mod"],  # [B, S_v, 6, D]，每个视频 patch token 一套
                "action": action_pre["t_mod"],  # [B, 6, D]，整条动作序列共用一套
            },
        )  # 更新后的 video/action token

        # MoT token 经 post_dit 解码为预测，其中 C_out = C_lat = VAE潜空间通道数 = 48（仓库配置）
        # video_expert / action_expert 各是一套 Video DiT / Action DiT，post_dit 只做 把 MoT 输出的 D 维 token 投回任务空间（潜空间或动作维），不再经过 MoT
        # pred_video（视频潜空间） 和 pre_action（动作序列） 均为流匹配的头，是与训练目标 epsl-y 对齐的模型输出 f_\theta。
        # pred_vedio 是被还原的 VAE 潜空间张量；pred_action 是动作序列预测，与加噪动作 / 训练目标 target_action 对齐，用于 动作 MSE
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)  # [B, C_out, T_lat, H_lat, W_lat]，与 input_latents / target_video 同形（C_out 为 DiT 配置的 out_dim，与 VAE 潜通道对齐）
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)  # [B, T_a, A]，与 action / target_action 同形

        # 根据是否有 anchor 条件计算 Video loss
        n_anchor = 0
        if inputs["anchor_latents"] is not None:
            n_anchor = inputs["anchor_latents"].shape[2]
            # 算 loss 时把 pred_video / target_video 的前 n_anchor 个 latent 时间步去掉，避免对 anchor 段算重建/扩散损失
            pred_video = pred_video[:, :, n_anchor:]
            target_video = target_video[:, :, n_anchor:]

        # 在 pred_video 与 target_video 上算，二者都是 视频 VAE 潜空间 里的张量，形状 [B, C_lat, T_lat, H_lat, W_lat]
        # 如果有 anchor，会先裁掉前 n_anchor 个时间步再算
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,  # 对 pred_video 与 target_video 做逐元素 MSE，先在 C、H、W 上平均，得到每个 潜空间时间步 的误差
            image_is_pad=image_is_pad,  # 对于 shape=[B, T_lat] 的 video_loss_token，按 image_is_pad 映射到潜空间步上做掩码（跳过无效的 padding 时间步）平均
            num_excluded_anchor_steps=n_anchor,
        )  # [B]

        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(  # timestep_video 指视频分支的扩散步，维度是 [B]
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )  # 与损失同设备 dtype，与 timestep_video 同维度，具体是 [B]
        loss_video = (loss_video_per_sample * video_weight).mean()  # 标量

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2)  # [B,T]，逐步 MSE，先把最后一个动作维度平均
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)  # 有效动作步
            valid_sum = valid.sum(dim=1).clamp(min=1.0)  # 每序列有效步数，防除零
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum  # 每样本损失：[B,T] --> [B]
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)  # 无 pad 时对时间维均值

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )  # 动作时间步权重，维度是 [B]
        loss_action = (action_loss_per_sample * action_weight).mean()  # 标量

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action  # 联合优化目标
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),  # 分项（含 λ）
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),  # 分项（含 λ）
        }
        return loss_total, loss_dict  # 总 loss 与可记录字典
        
    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,  # [B, C_lat, T_lat, H_lat, W_lat]
        latents_action: torch.Tensor,  # [B, T_a, A]，T_a 是动作序列的长度
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,  # [B, L, D]
        context_mask: torch.Tensor, # [B, L]
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """推理时调用：单步联合预测（单步推理）
        在已有 latents_video / latents_action 与两侧 timestep_*、context 等前提下，
        走与训练核心相同的 pre_dit → MoT 掩码 → mot → post_dit，
        返回当前步的 pred_video, pred_action，供推理调度器 step 更新潜变量。
        """
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_anchor_frames=self.num_anchor_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_anchor_frames=self.num_anchor_frames,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        # 对当前步视频/动作潜变量给出的 flow-matching / 扩散头输出，是与训练目标 epsl-y 对齐的模型输出 f_\theta
        # 用于计算 flow matching 中的 MSE 损失，其 shape 与输入的 latents_video 和 latents_action 保持一致
        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        anchor_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=anchor_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=anchor_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
            num_anchor_frames=self.num_anchor_frames,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_anchor_frames=self.num_anchor_frames,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: Union[torch.Tensor, list[torch.Tensor]],
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone() if isinstance(input_image, torch.Tensor) else [img.clone() for img in input_image],
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        # 输入检查：图像形状与 num_video_frames（含 T % 4 == 1、H/W 为 16 倍数 等）
        if isinstance(input_image, list):
            for i, img in enumerate(input_image):
                if img.ndim == 3:
                    input_image[i] = img.unsqueeze(0)
            _, _, height, width = input_image[0].shape
        else:
            if input_image.ndim == 3:
                input_image = input_image.unsqueeze(0)
            if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
                )
            _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )

        # 可选 action（作为 视频条件 进 video_expert，注释写明不是给 action 分支当 GT）
        # 是可选的已知动作序列，用来喂给后面循环中的视频 DiT，做 action-conditioned 视频生成（动作信息进 cross-attn / 条件分支 那一侧）
        # 显式提供的一条动作轨迹（语义上经常是真实/示教动作），只用于 条件化视频分支
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)

        # 可选 proprio 拼进 context
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        # 初始化双路的噪声，可用 不同 generator（同一 seed 时仍各自 manual_seed(seed)）分别采样。
        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # 图像经 VAE 得到 anchor_latents
        if isinstance(input_image, list):
            input_image = [img.to(device=self.device, dtype=self.torch_dtype) for img in input_image]
            anchor_latents = self._encode_multi_image_latents_tensor(input_image, tiled=tiled)
        else:
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            anchor_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)

        # 锚点写入视频潜变量，供后面联合预测噪声 _predict_joint_noise 使用
        n_anchor = anchor_latents.shape[2]
        latents_video[:, :, 0:n_anchor] = anchor_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # prompt 或 context/context_mask（二选一）
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        
        # 可选 proprio 是否融入 context 及 context_mask
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # 双调度表：infer_video_scheduler 与 infer_action_scheduler 各建 num_inference_steps 的 (timestep, delta)
        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        # 然后用 ZIP 逐步对齐：一步里视频、动作各有一个时间步与更新量；是联合去噪循环
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            # 对 当前整段 latents_video 与 latents_action，用 本步 timestep_video / timestep_action
            # 走 video pre_dit → action pre_dit → 全量 MoT（非 KV 捷径）→ 双 post_dit
            # 得到 pred_video、pred_action（与训练时联合前向同构）
            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,  # 若传入，会进 视频分支（例如 action-conditioned 视频），用于条件生成，不是替代对 latents_action 的去噪
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            # 2 个 Step 函数更新 latents_video、latents_action
            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

            # 硬约束，防止锚点帧在扩散步中被冲掉
            latents_video[:, :, 0:n_anchor] = anchor_latents.clone()

        # 动作取 latents_action 转 CPU；
        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        # 视频对 latents_video 做 _decode_latents 得到像素 video
        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: Union[torch.Tensor, list[torch.Tensor]],
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        """推理时只生成动作，不联合生成 video 和 action
        """
        self.eval()  # PyTorch 标准行为：关闭 Dropout、BatchNorm 等训练期随机性，固定归一化统计，保证推理时前向行为稳定、可复现。
        
        # 要求 video_attention_mask_mode == "first_frame_causal"
        # first_frame_causal 与 动作只 attend 锚点视频 token（_build_mot_attention_mask 里 action 行只对 [:anchor_tokens] 开）
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        # 检验输入图像的形状，与 H，W 应该为 16 的倍数
        if isinstance(input_image, list):
            for i, img in enumerate(input_image):
                if img.ndim == 3:
                    input_image[i] = img.unsqueeze(0)
            _, _, height, width = input_image[0].shape
        else:
            if input_image.ndim == 3:
                input_image = input_image.unsqueeze(0)
            if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
                raise ValueError(
                    f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
                )
            _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )

        # 如果存在 proprio 则校验并编码进 context
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        # 用 seed 初始化高斯噪声，得到 latents_action（形状 [1, action_horizon, action_dim]）
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        # 锚点：input_image 经 VAE 编成 anchor_latents
        # 条件帧不走单独一条「额外 VAE embedding 分支」，而是融在 latent 序列里并用 t=0 标成「干净锚点」
        if isinstance(input_image, list):
            input_image = [img.to(device=self.device, dtype=self.torch_dtype) for img in input_image]
            anchor_latents = self._encode_multi_image_latents_tensor(input_image, tiled=tiled)
        else:
            input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
            anchor_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        # 从 prompt 或 context/context_mask 得到条件
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        # 可选 proprio 拼进 context
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        # 用 timestep_video = 0 对 anchor（只对 anchor 编码） 做一次 video_expert.pre_dit，得到 video token
        timestep_video = torch.zeros(
            (anchor_latents.shape[0],),
            dtype=anchor_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=anchor_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
            num_anchor_frames=self.num_anchor_frames,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])

        # 再构建 MOT 的 attention_mask，是整段联合序列的掩码
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
            num_anchor_frames=self.num_anchor_frames,
        )

        # 该函数在 mot.py 里逐层跑视频分支自注意力，并把 每层视频 K、V 存进 video_kv_cache
        # video_kv_cache 存的是 各层「锚点视频」在混合注意力里供动作 query 去 attend 的 K/V
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],  # 规定 锚点视频 token 互相之间 谁能看谁
        )

        # 只为动作建 infer_action_scheduler 的推理 schedule（动作 branch 扩散的第 1 步）
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            # 对每个扩散步：用 _predict_action_noise_with_cache（动作噪声预测 + 复用 video_kv_cache）得到 pred_action
            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi
            # 再 infer_action_scheduler.step 更新 latents_action，只有动作 schedule
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        # 只返回 action（搬到 CPU float32），不返回 video
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: Union[torch.Tensor, list[torch.Tensor]],
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            # Multi-anchor metadata: needed so downstream eval/load can detect silent
            # architecture-vs-weights mismatch (N>1 ckpts must NOT be loaded with N=1 runtime,
            # and vice versa, because the anchor/denoise mask + cross-attn group count depend
            # on N and latent alignment is destroyed otherwise).
            "num_anchor_frames": int(getattr(self, "num_anchor_frames", 1)),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        # Multi-anchor compatibility check: ckpt was saved under a specific N, and the anchor
        # mask + temporal-group count baked into the weights are meaningful only for that N.
        ckpt_N = payload.get("num_anchor_frames", None)
        runtime_N = int(getattr(self, "num_anchor_frames", 1))
        if ckpt_N is None:
            if runtime_N != 1:
                raise ValueError(
                    f"Checkpoint has no `num_anchor_frames` metadata (pre-multi-anchor format) "
                    f"but runtime `num_anchor_frames={runtime_N}`. Pass `model.num_anchor_frames=1` "
                    f"or re-train with N={runtime_N}."
                )
        elif int(ckpt_N) != runtime_N:
            raise ValueError(
                f"Checkpoint `num_anchor_frames={int(ckpt_N)}` does NOT match runtime "
                f"`num_anchor_frames={runtime_N}`. Pass `model.num_anchor_frames={int(ckpt_N)}` "
                f"to match the checkpoint."
            )

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
