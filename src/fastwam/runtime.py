import logging
import os
import inspect
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.video_io import save_mp4
from .utils import misc

logger = get_logger(__name__)

# --- 训练/推理运行时：Hydra instantiate 目标、数据集接线、CLI 入口 ---


def _normalize_mixed_precision(mixed_precision: str) -> str:
    """校验并归一化配置里的混合精度字符串。

    要求为 ``str``，strip 后小写，且只能是 ``no`` / ``fp16`` / ``bf16``；否则抛 ``ValueError``。
    返回值供 Trainer、Accelerate、模型权重 dtype 等共用同一套配置语义。

    strip()：去掉字符串首尾的空白字符（空格、\t、\n 等），不改变中间内容
    """
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    """把 ``mixed_precision`` 转成模型加载/存放权重时的 ``torch.dtype``。

    ``no`` → ``float32``；``fp16`` → ``float16``；``bf16`` → ``bfloat16``（内部先经 ``_normalize_mixed_precision``）。
    """
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def create_wan22_model(
    model_id: str,
    tokenizer_model_id: str,
    dit_config,
    tokenizer_max_len: int = 512,
    train_shift: float = 5.0,
    infer_shift: float = 5.0,
    num_train_timesteps: int = 1000,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    # Wan2.2 主干（视频 DiT + tokenizer）；可由 cfg.model._target_ 经 Hydra 调用
    from .models.wan22.wan22 import Wan22Core

    if isinstance(dit_config, DictConfig):
        dit_config = OmegaConf.to_container(dit_config, resolve=True)
    if not isinstance(dit_config, dict):
        raise ValueError(f"`dit_config` must resolve to a dict, got {type(dit_config)}")

    # 从 HF 预训练加载 Wan2.2，并按 dit_config 构建 DiT
    return Wan22Core.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=dit_config,
        train_shift=float(train_shift),
        infer_shift=float(infer_shift),
        num_train_timesteps=int(num_train_timesteps),
    )


def create_fastwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    num_anchor_frames: int = 1,
):
    """Hydra 在 configs/model/fastwam.yaml 里通过 _target_: fastwam.runtime.create_fastwam 指向的模型工厂。
    把 YAML 里的视频 DiT / ActionDiT / scheduler / loss 等参数从 DictConfig 转成普通 dict，做必填校验，
    最后调用 FastWAM.from_wan22_pretrained(...)，得到可在设备上训练的 FastWAM 实例。不负责数据加载。
    """
    from .models.wan22.fastwam import FastWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    # 可选动作 DiT 子配置；空 dict 表示使用 FastWAM 内部默认
    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    # 视频 / 动作分支扩散步数（动作分支必填）
    if isinstance(video_scheduler, DictConfig):
        # resolve=True：在转成普通 dict/list 之前，先把配置里所有 ${...} 插值解析成最终值，得到 可直接喂给 from_wan22_pretrained 的纯 Python 字面量
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        # 后面 FastWAM.from_wan22_pretrained 里用 video_scheduler.get("train_shift", 5.0) 等；没配 video_scheduler 就走这些默认值，写成 None 使用 .get 会报错
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")

    # 动作分支连续扩散调度：三项缺一不可，后续 from_wan22_pretrained 用下标读取，缺键会 KeyError。
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}  # 必填键集合
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())  # 配置里尚未出现的键
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )  # 提前给出可读错误，避免深入构建后再崩

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    # 加载预训练 Wan2.2 并构建 FastWAM 各头与训练 loss 权重
    # 返回的结果是 run_training 函数中的 model 变量，在这里 定义并得到 video_expert、action_expert、mot，再 传给 FastWAM(...)
    return FastWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        num_anchor_frames=int(num_anchor_frames),
    )


def create_fastwam_joint(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    num_anchor_frames: int = 1,
):
    # 联合视频-动作变体（cfg 形态同 create_fastwam，实现类不同）
    from .models.wan22.fastwam_joint import FastWAMJoint

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMJoint.from_wan22_pretrained(  # 与 FastWAM 相同的预训练加载流程
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        num_anchor_frames=int(num_anchor_frames),
    )


def create_fastwam_idm(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    num_anchor_frames: int = 1,
):
    # FastWAM 的反动力学（IDM）变体，结构随任务配置而异
    from .models.wan22.fastwam_idm import (
        FastWAMIDM,
    )

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}")

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}")

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(f"`video_scheduler` must be dict-like, got {type(video_scheduler)}")

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for FastWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(f"`action_scheduler` must be dict-like, got {type(action_scheduler)}")
    required_action_scheduler_keys = {"train_shift", "infer_shift", "num_train_timesteps"}
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    return FastWAMIDM.from_wan22_pretrained(  # IDM 头 + 与 Wan2.2 主干一致的加载约定
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        num_anchor_frames=int(num_anchor_frames),
    )


def build_datasets(data_cfg: DictConfig):
    """
    只接收 data_cfg（即 cfg.data）：用 instantiate(data_cfg.train) 建训练集；
    若配置了 val 则再 instantiate 验证集（并处理 pretrained_norm_stats），否则验证集与训练集同一对象。
    返回 (train_ds, val_ds)。不建模型。
    """
    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        # 未单独配置 val：复用训练集对象（如 LIBERO 常用）
        val_ds = train_ds
    else:
        # 验证集须使用固定的动作/状态归一化统计（本次 run 目录或显式 JSON）
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


def _resolve_train_device() -> str:
    """解析本进程用于**初始**建模型时的设备字符串（``run_training`` → ``instantiate(..., device=...)``）。

    - 无 CUDA：``cpu``。
    - 有 CUDA 且仅 1 卡，或 ``LOCAL_RANK`` 越界/异常：``cuda:0``（单卡或兜底）。
    - 多卡且 ``torchrun``/Accelerate 设置了 ``LOCAL_RANK``：``cuda:{LOCAL_RANK}``，实现一进程绑定一张卡。

    注意：``Wan22Trainer`` 内仍以 ``Accelerator`` 的设备为准做训练；此函数主要对齐工厂建权重时的 ``device`` 参数。
    """
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def run_training(cfg: DictConfig):
    """训练入口：打日志、登记工作目录、把完整 cfg 落盘；
    解析本进程 device 与 mixed_precision → model_dtype；
    instantiate(cfg.model, ...) 间接调用 create_fastwam
    build_datasets(cfg.data) 得到数据集
    构造 Wan22Trainer(cfg, model, train_ds, val_ds) 并 trainer.train()。
    不实现具体训练步，只负责组装与委托。
    """
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )  # 初始化日志；分布式时仅 rank0 视为主进程写控制台

    # 注册全局工作目录并保存本次运行的 config 快照，便于复现与下游写 dataset_stats 等。
    misc.register_work_dir(cfg.output_dir)  # 供 misc 内部解析输出根路径
    config_payload = OmegaConf.to_container(cfg, resolve=True)  # 解析插值为普通容器便于序列化
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)  # 写入 output_dir/config.yaml

    # 本进程设备与混合精度对齐后，按 cfg.model._target_ 工厂建模型，再按 cfg.data 建训练/验证集。
    # cfg.model 和 cfg.task 具体在任务配置文件 configs/task/libero_uncond_2cam224_1e-4.yaml 中给出
    model_device = _resolve_train_device()  # LOCAL_RANK 对应 GPU，无 CUDA 则 cpu
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)  # 归一化 bf16/fp16/no 等写法
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)  # 映射到 torch.dtype 供权重 dtype

    # configs/model/fastwam.yaml 里 _target_: fastwam.runtime.create_fastwam
    # 模型实例化的具体步骤：instantiate(cfg.model) → create_fastwam（模型工厂函数，在本文件中定义）
    # → FastWAM.from_wan22_pretrained（拉 components.dit、建 ActionDiT、包 MoT）
    # → FastWAM(video_expert, action_expert, mot, vae, …)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)  # 调用 create_fastwam 等工厂（在本文件中已定义）

    # train_ds：instantiate(data_cfg.train) 得到的训练用数据集（常为 RobotVideoDataset 等）
    train_ds, val_ds = build_datasets(cfg.data)  # LeRobot 管线等由 data 配置决定

    # 将 cfg、模型与数据集交给 Trainer，内部负责 accelerate、优化器与日志。
    trainer = Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,  # 由 _build_datasets 构建，传给 Wan22Trainer 中用来做数据集
        val_dataset=val_ds,
    )  # 封装训练所需状态

    # 重点考虑 build_datasets 与 Wan22Trainer 的接线方式（codex建议）
    trainer.train()  # 阻塞直至 epoch/step 结束或早停逻辑

def run_inference(cfg: DictConfig):
    # Single-image -> video CLI path (not the LIBERO sim eval script).
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()

    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        # Match training-style resize: cover then center crop to cfg width/height.
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0  # [0,255] -> [-1,1]
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)  # Wan2.2 / FastWAM image-conditioned video sample
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4
