import hashlib
import os
from typing import Optional
import time
import numpy as np
import traceback
import torch
import torchvision.transforms.functional as transforms_F
from contextlib import contextmanager

from omegaconf import DictConfig, OmegaConf

from hydra.utils import instantiate
from .base_lerobot_dataset import BaseLerobotDataset
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from fastwam.utils.logging_config import get_logger
from fastwam.utils import misc, pytorch_utils
from accelerate import PartialState
logger = get_logger(__name__)


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"

class RobotVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: str = "horizontal", # "horizontal", "vertical", "robotwin", or None
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        action_horizon: Optional[int] = None,
        num_anchor_frames: int = 1,
        num_denoise_latent_frames: Optional[int] = None,
    ):
        # Multi-anchor consistency: derive action window from latent semantics.
        num_anchor_frames = int(num_anchor_frames)
        assert num_anchor_frames >= 1, f"`num_anchor_frames` must be >= 1, got {num_anchor_frames}"
        assert (num_frames - 1) % action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {action_video_freq_ratio}"
        sampled_T = (num_frames - 1) // action_video_freq_ratio + 1
        assert (sampled_T - 1) % 4 == 0, \
            f"sampled video frames-1 must be divisible by 4, got {sampled_T - 1}"
        num_latent_frames = 1 + (sampled_T - 1) // 4
        assert num_latent_frames > num_anchor_frames, \
            f"`num_latent_frames` ({num_latent_frames}) must be > `num_anchor_frames` ({num_anchor_frames})"
        expected_action_horizon = action_video_freq_ratio * 4 * (num_latent_frames - num_anchor_frames)
        if action_horizon is None:
            action_horizon = expected_action_horizon
        action_horizon = int(action_horizon)
        assert action_horizon == expected_action_horizon, (
            f"`action_horizon` ({action_horizon}) must equal "
            f"action_video_freq_ratio*4*(num_latent_frames-num_anchor_frames) ({expected_action_horizon})"
        )
        if num_denoise_latent_frames is not None:
            assert int(num_denoise_latent_frames) == num_latent_frames - num_anchor_frames, (
                f"`num_denoise_latent_frames` ({num_denoise_latent_frames}) must equal "
                f"num_latent_frames - num_anchor_frames ({num_latent_frames - num_anchor_frames})"
            )
        action_start_offset = 4 * action_video_freq_ratio * (num_anchor_frames - 1)

        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=action_horizon,
            action_start_offset=action_start_offset,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
        )

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio
        self.action_horizon = action_horizon
        self.num_anchor_frames = num_anchor_frames
        self.num_denoise_latent_frames = (
            int(num_denoise_latent_frames)
            if num_denoise_latent_frames is not None
            else (num_latent_frames - num_anchor_frames)
        )
        self.action_start_offset = action_start_offset
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        # Multi-anchor diagnostic: under N>1 the observation window is longer (49 raw steps for
        # N=M=2 vs. 33 for N=1) and the action window is right-shifted by `action_start_offset`,
        # so episodes that are shorter than `num_frames + action_start_offset` will produce
        # all-pad samples at every sliding-window start. Emit a one-shot summary on the main
        # process so users can detect data issues (e.g. short demos, dataset corruption) early.
        if PartialState().is_main_process:
            try:
                ep_from = self.lerobot_dataset.episode_data_index["from"].to(torch.long)
                ep_to = self.lerobot_dataset.episode_data_index["to"].to(torch.long)
                ep_lengths = (ep_to - ep_from).tolist()
                if len(ep_lengths) > 0:
                    ep_min = int(min(ep_lengths))
                    ep_max = int(max(ep_lengths))
                    ep_mean = float(sum(ep_lengths)) / float(len(ep_lengths))
                    required_min = int(num_frames)  # enough to cover obs window
                    required_no_pad = int(num_frames + action_start_offset)  # to avoid action pad on first sample
                    num_too_short = sum(1 for L in ep_lengths if L < required_min)
                    num_action_may_pad = sum(1 for L in ep_lengths if L < required_no_pad)
                    tag = "train" if is_training_set else "val"
                    logger.info(
                        "[RobotVideoDataset/%s] episodes=%d | length min=%d mean=%.1f max=%d | "
                        "num_frames=%d, action_start_offset=%d, action_horizon=%d, num_anchor_frames=%d",
                        tag, len(ep_lengths), ep_min, ep_mean, ep_max,
                        num_frames, action_start_offset, action_horizon, num_anchor_frames,
                    )
                    if num_too_short > 0:
                        logger.warning(
                            "[RobotVideoDataset/%s] %d/%d episodes are shorter than num_frames=%d; "
                            "these will produce fully-padded obs windows. Consider filtering them "
                            "out upstream or reducing `num_anchor_frames`/`num_denoise_latent_frames`.",
                            tag, num_too_short, len(ep_lengths), required_min,
                        )
                    elif num_action_may_pad > 0 and num_anchor_frames > 1:
                        logger.info(
                            "[RobotVideoDataset/%s] %d/%d episodes are shorter than "
                            "num_frames+action_start_offset=%d; rolling samples at their tail will "
                            "produce action pad. Enable `skip_padding_as_possible=true` if you want "
                            "to resample those windows.",
                            tag, num_action_may_pad, len(ep_lengths), required_no_pad,
                        )
            except Exception as diag_err:  # defensive; diagnostic should never fail hard
                logger.warning(
                    "[RobotVideoDataset] episode length diagnostic failed: %s", diag_err,
                )

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)
        
    def __len__(self):
        return len(self.lerobot_dataset)

    def _get(self, idx):
        sample_idx = idx
        sample = None
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]

            if not self.skip_padding_as_possible:
                break

            # Evaluate pad masks against the windows actually fed to the model:
            # - action: BaseLerobotDataset already returned the right-shifted `action_horizon` window.
            # - proprio: slice the 32-step window starting at `action_start_offset`.
            # - image : slice to the sampled video frames (decimated by `action_video_freq_ratio`).
            action_is_pad = sample["action_is_pad"]
            image_is_pad = sample["image_is_pad"]
            proprio_is_pad = sample["proprio_is_pad"]
            proprio_is_pad_window = proprio_is_pad[
                self.action_start_offset: self.action_start_offset + self.action_horizon
            ]
            image_is_pad_window = image_is_pad[self.video_sample_indices]
            has_pad = False
            if bool(action_is_pad.any().item()):
                has_pad = True
            if bool(image_is_pad_window.any().item()):
                has_pad = True
            if bool(proprio_is_pad_window.any().item()):
                has_pad = True

            if not has_pad or attempt >= self.max_padding_retry:
                break

            sample_idx = np.random.randint(len(self.lerobot_dataset))

        image_is_pad = sample["image_is_pad"]

        video = sample["pixel_values"]  # [T, C, H, W] or [num_cameras, T, C, H, W]
        num_cameras = 1
        if video.ndim == 5:
            video = video[:, self.video_sample_indices, :, :, :] # [num_cameras, T_video, C, H, W]
            num_cameras, T_video, C, H, W = video.shape
        else:
            assert video.ndim == 4, f"Expected video to have shape [T, C, H, W], but got {video.shape}"
            video = video[self.video_sample_indices, :, :, :] # [T_video, C, H, W]
            T_video, C, H, W = video.shape
        image_is_pad = image_is_pad[self.video_sample_indices]

        video = video.view(num_cameras, T_video, C, H, W)  # [num_cameras, T_video, C, H, W]
        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(
                    f"`concat_multi_camera='robotwin'` requires exactly 3 cameras, got {num_cameras}"
                )
            cam_top = transforms_F.resize(
                video[0],
                size=[256, 320],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 256, 320]
            cam_left = transforms_F.resize(
                video[1],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            cam_right = transforms_F.resize(
                video[2],
                size=[128, 160],
                interpolation=transforms_F.InterpolationMode.BILINEAR,
                antialias=True,
            )  # [T_video, C, 128, 160]
            bottom = torch.cat([cam_left, cam_right], dim=-1)  # [T_video, C, 128, 320]
            video = torch.cat([cam_top, bottom], dim=-2)  # [T_video, C, 384, 320]
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)  # [T_video, C, H, num_cameras*W]
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)  # [T_video, C, num_cameras*H, W]
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)  # [T_video, C, H, W]

        # final resize and normalization
        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)  # [T_video, C, H, W]

        video = video.permute(1, 0, 2, 3) # [C, T_video, H, W], range [-1, 1]

        # Proxy (from lerobot, after BaseLerobotDataset right-shifts action by `action_start_offset`):
        #   action : [action_horizon, action_dim] starting from raw step `action_start_offset`
        #   proprio: [num_frames, proprio_dim] spanning the full obs window; slice to the 32-step
        #            window aligned with `action`.
        action = sample["action"]  # [action_horizon, action_dim]
        proprio_full = sample["proprio"]  # [num_frames, state_dim]
        proprio = proprio_full[self.action_start_offset: self.action_start_offset + self.action_horizon, :]
        # proprio_is_pad key name is rewritten from `state_is_pad` by processor.preprocess().
        proprio_is_pad_full = sample["proprio_is_pad"]
        proprio_is_pad = proprio_is_pad_full[
            self.action_start_offset: self.action_start_offset + self.action_horizon
        ]
        if video.shape[1] <= 1:
            raise ValueError(f"`video` must have at least 2 frames, got shape {tuple(video.shape)}")
        assert action.shape[0] == self.action_horizon, (
            f"`action` shape[0]={action.shape[0]} mismatch with `action_horizon`={self.action_horizon}"
        )
        assert proprio.shape[0] == self.action_horizon, (
            f"`proprio` sliced shape[0]={proprio.shape[0]} mismatch with `action_horizon`={self.action_horizon}"
        )

        task = sample["instruction"]
        
        # FIXME
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)

        context, context_mask = self._get_cached_text_context(instruction)
        # NOTE: to keep consistent with wan2.2's behavior
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        
        data = {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": proprio_is_pad,
        }
        return data

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            data = self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            # trace back
            print(traceback.format_exc())
            random_idx = np.random.randint(len(self))
            data = self._get(random_idx)
        return data
