import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf


def create_task_file(output_file: Path, task_suite_names: list[str]) -> Path:
    benchmark_dict = benchmark.get_benchmark_dict()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    total_tasks = 0
    with output_file.open("w", encoding="utf-8") as f:
        for suite_name in task_suite_names:
            task_suite = benchmark_dict[suite_name]()
            n_tasks = int(task_suite.n_tasks)
            print(f"\n{suite_name}:")
            print(f"- Number of tasks: {n_tasks}")
            for task_id in range(n_tasks):
                f.write(f"{suite_name},{task_id}\n")
                total_tasks += 1

    print(f"\nTask list created: {output_file}")
    print(f"Total tasks: {total_tasks}")
    return output_file


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
    }
    if key in blocked_exact:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def collect_worker_overrides() -> list[str]:
    hydra_overrides = list(HydraConfig.get().overrides.task)
    return [ov for ov in hydra_overrides if not _is_blocked_override(ov)]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or str(task_choice).strip() == "":
        raise ValueError(
            "Hydra task choice is empty. Please pass task=... (e.g., task=world_action_model_forward_224)."
        )
    return str(task_choice)


def run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
) -> None:
    script_path = Path("experiments/libero/run_libero_parallel_test.sh")
    if not script_path.exists():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    root_dir = os.getcwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_args = shlex.join(extra_overrides) if extra_overrides else ""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": root_dir,
            "RUN_ID": run_id,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": extra_args,
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
        }
    )

    print("\nStarting evaluation (Hydra manager)...")
    print(f"task: {task_choice}")
    print(f"Checkpoint: {ckpt}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Trials per task: {num_trials}")
    print(f"Max tasks per GPU: {max_tasks_per_gpu}")
    print(f"Output directory: {output_dir}")
    if extra_args:
        print(f"Forwarded overrides: {extra_args}")

    try:
        # 用 subprocess.run(["bash", "experiments/libero/run_libero_parallel_test.sh", str(task_file)], env=...) 起并行脚本
        subprocess.run(
            ["bash", str(script_path), str(task_file)],
            env=env,
            check=True,
            text=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Evaluation script failed with return code: {e.returncode}")
        failed_tasks = output_dir / "failed_tasks.txt"
        if failed_tasks.exists() and failed_tasks.stat().st_size > 0:
            print(f"Failed subtask list: {failed_tasks}")
            print(failed_tasks.read_text(encoding='utf-8'))
        raise


# 固定用 config_name="sim_libero.yaml"，MULTIRUN.task_suite_names（libero_spatial等） 以 configs/sim_libero.yaml 为准。
@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")  # 工作目录与超参自 configs/sim_libero.yaml
def main(cfg: DictConfig):
    # 管理端入口：校验必选项 → 生成任务表与配置快照
    if cfg.ckpt is None:  # 无权重则无法跑仿真
        raise ValueError("ckpt must not be None.")
    if cfg.EVALUATION.output_dir is None:  # 子进程与写盘都依赖此根目录
        raise ValueError("EVALUATION.output_dir must not be None.")

    task_choice = _resolve_worker_task_choice()  # 当前选中的 task= 配置名，经环境交给并行脚本
    manager = cfg.MULTIRUN  # 含 task_suite、GPU 数、任务表路径、create_only 等

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))  # 展开 ~ 与 $VAR
    output_dir.mkdir(parents=True, exist_ok=True)  # 先建目录，便于写 tasks 与 config 备份

    task_file_cfg = manager.get("task_file")  # 可选：自定义任务表文件路径
    if task_file_cfg:  # 显式则用之
        task_file = Path(os.path.expanduser(os.path.expandvars(str(task_file_cfg))))
    else:  # 否则默认与本次 run 的 output 放一起
        task_file = output_dir / "tasks.txt"
    task_file = create_task_file(task_file, list(manager.task_suite_names))  # 将各 suite 展平为 suite,task_id 行

    OmegaConf.save(config=cfg, f=str(output_dir / "manager_config.yaml"))  # 存本次运行完整 cfg 便于复现

    if bool(manager.get("create_only", False)):  # 只生成任务表与 config，不 spawn 子进程
        print("create_only=True, only create the task list and exit.")
        return

    run_evaluation(  # 调 bash 为各 GPU/worker 注入环境变量并执行并行评测
        task_file=task_file,
        task_choice=task_choice,
        ckpt=str(cfg.ckpt),
        num_gpus=int(manager.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(manager.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=collect_worker_overrides(),  # 透传非 MULTIRUN/hydra 等屏蔽项外的 CLI 覆盖
    )


if __name__ == "__main__":
    """文件调用概述：
    1、run_libero_manager.main（sim_libero.yaml）

    2、create_task_file → 生成 tasks.txt（suite,task_id 每行）

    3、run_evaluation
    
    4、run_libero_parallel_test.sh（run_libero_eval）：读任务表、tmux/多 GPU 调度、对每个子任务：
    python eval_libero_single.py，Hydra：task=CONFIG、ckpt、EVALUATION.task_suite_name / task_id / gpu_id 等
    eval_single_process（eval_libero_single 的 @hydra.main 入口，约 783+ 行）
    加载 checkpoint、FastWAMProcessor、任务 suite/task（LIBERO benchmark）
    run_single_task
    每个 trial：run_single_episode（LIBERO env、观测 → 模型 → env.step）
    
    5、全部子任务结束后，shell 可跑 experiments/libero/summarize_results.py --output_dir=...（约 636–638 行）
    """
    main()
