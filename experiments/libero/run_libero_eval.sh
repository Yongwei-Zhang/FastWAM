#!/usr/bin/env bash
set -euo pipefail

# 火山引擎自定义任务的容器镜像是精简环境，未预装 tmux
# run_libero_parallel_test.sh 依赖 tmux 创建 session/pane 来调度子任务
if ! command -v tmux &>/dev/null; then
  echo "[info] tmux not found, installing..."
  apt-get update -qq && apt-get install -y -qq tmux
fi

# MuJoCo EGL 离屏渲染（容器中可能缺少默认配置）
# 在脚本中设置 MUJOCO_GL=egl 和 LD_LIBRARY_PATH，确保 EGL 库可被找到。
# 如果容器里确实没有 libEGL.so，还需要在入口命令中加 apt-get install -y libegl1-mesa-dev。
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# =========================
# 从 Github 的 Issues 中 Copy 下来：https://github.com/yuantianyuan01/FastWAM/issues/25
# Default evaluation config
# =========================
TASK="libero_uncond_2cam224_1e-4"

# 给出的模型权重；动作/观测归一化统计
# CKPT="./checkpoints/fastwam_release/libero_uncond_2cam224.pt"
# DATASET_STATS_PATH="./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"

# （切换到根目录执行）评测自己训练的模型，DATASET_STATS_PATH 必须和该次训练用的统计一致
CKPT="./runs/libero_uncond_2cam224_1e-4/2026-04-09_08-25-37/checkpoints/weights/step_020000.pt"
DATASET_STATS_PATH="./runs/libero_uncond_2cam224_1e-4/2026-04-09_08-25-37/dataset_stats.json"

########################################
# 脚本评估参数
# configs/sim_libero.yaml 中关于 MULTIRUN 的设置
NUM_GPUS="4"  # 参数设置，等号2边不能有空格
# Optional: specify visible GPUs directly here (e.g. "0,2,5").
# Leave empty to use NUM_GPUS scheduling (0..NUM_GPUS-1).
# GPU_IDS="0,1,2,3"

# 每张 GPU 上的最大任务数（官方脚本给的是2）
MAX_TASKS_PER_GPU=2

# 可选：覆盖 MULTIRUN.task_suite_names（1～4 个，逗号分隔，无空格）。
# 例：libero_spatial,libero_object  或  libero_10
# 留空则沿用 Hydra 默认（configs/sim_libero.yaml 中的列表）。
TASK_SUITE_NAMES=""
# TASK_SUITE_NAMES="libero_spatial"

########################################

# Resolve project paths relative to this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 本脚本所在目录的绝对路径
FASTWAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # 从 SCRIPT_DIR 向上2级：fastwam 仓库根目录
WORKSPACE_ROOT="$(cd "${FASTWAM_ROOT}/.." && pwd)"  # FASTWAM_ROOT 的上一级，即 包含 FastWAM 仓库的那层目录
LIBERO_ROOT="${WORKSPACE_ROOT}/LIBERO"  # WORKSPACE 中与 FastWAM 同级的 LIBERO 目录

if [[ ! -d "${LIBERO_ROOT}" ]]; then
  echo "Error: LIBERO root not found at ${LIBERO_ROOT}"
  exit 1
fi

# Make sure python can import libero.libero.* reliably.
# 把 LIBERO 仓库根目录 加入 Python 的模块搜索路径
# PYTHONPATH：环境变量，列出 Python 在标准库和已安装包之外还要搜索模块的目录列表，让解释器能在这些路径下找到包/模块
if [[ -z "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${LIBERO_ROOT}"
else
  export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"
fi

cd "${FASTWAM_ROOT}"

# LIBERO 首次 import 时若缺少 config.yaml 会 input()，非交互环境会 EOFError
# 确定 每次运行前都已有合法 config.yaml（~/.libero/config.yaml），告诉 libero 包仿真资源路径在哪
_LIBERO_CFG_ROOT="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
_LIBERO_CFG_FILE="${_LIBERO_CFG_ROOT}/config.yaml"
if [[ ! -f "${_LIBERO_CFG_FILE}" ]]; then
  printf 'N\n' | python -c "from libero.libero import benchmark"  # 在无配置文件时 自动生成默认配置
fi

# Usage:
#   1) Modify variables in this script, then run:
#      bash experiments/libero/run_libero_eval.sh
#   2) Optionally pass extra Hydra overrides:
#      bash experiments/libero/run_libero_eval.sh MULTIRUN.max_tasks_per_gpu=2

# 若设置了 GPU_IDS：清洗列表、限制可见 GPU、并用列表长度覆盖 NUM_GPUS。
if [[ -n "${GPU_IDS:-}" ]]; then  # 未设置 GPU_IDS 时视为空，与 set -u 兼容
  GPU_IDS="$(echo "${GPU_IDS}" | tr -d '[:space:]')"  # 去掉空格等空白，避免误解析
  if [[ -z "${GPU_IDS}" ]]; then  # 清洗后为空则视为非法
    echo "Error: empty GPU list."
    exit 1
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"  # 仅暴露这些物理卡；进程内编号从 0 起
  NUM_GPUS="$(echo "${GPU_IDS}" | tr ',' '\n' | awk 'NF' | wc -l | tr -d ' ')"  # 逗号分隔项数 = 实际用卡数
  echo "Using specified GPUs: ${CUDA_VISIBLE_DEVICES} (count=${NUM_GPUS})"
fi

ARGS=(
  "task=${TASK}"
  "ckpt=${CKPT}"
  "EVALUATION.dataset_stats_path=${DATASET_STATS_PATH}"
  "MULTIRUN.num_gpus=${NUM_GPUS}"
  "MULTIRUN.max_tasks_per_gpu=${MAX_TASKS_PER_GPU}"
)

# 设置评测任务的种类
if [[ -n "${TASK_SUITE_NAMES// /}" ]]; then
  _norm="${TASK_SUITE_NAMES// /}"
  IFS=',' read -ra _SUITES <<< "${_norm}"
  _n="${#_SUITES[@]}"
  if [[ "${_n}" -lt 1 || "${_n}" -gt 4 ]]; then
    echo "Error: TASK_SUITE_NAMES must be 1..4 comma-separated suite names (got ${_n})."
    exit 1
  fi
  _hydra_list="[${_SUITES[0]}"
  for ((i = 1; i < _n; i++)); do
    _hydra_list+=",${_SUITES[i]}"
  done
  _hydra_list+="]"
  ARGS+=("MULTIRUN.task_suite_names=${_hydra_list}")
fi

python experiments/libero/run_libero_manager.py "${ARGS[@]}" "$@"