# 火山引擎自定义任务：LIBERO 评估 CPU/GPU 利用率接近 0 的原因与处理

本文档总结在**火山引擎机器学习平台「自定义任务」**中运行 `experiments/libero/run_libero_eval.sh` 时，监控上 **CPU、GPU 长期接近 0** 的常见原因及已在仓库中落地的修复思路，便于与本地开发机（已装 tmux、已配置 shell、已配显示/EGL）行为对齐。

---

## 1. 现象

- 任务状态为「运行中」，但平台监控里 **CPU / GPU 利用率长时间接近 0**。
- 或日志里出现 **`tmux: command not found`**、**`ModuleNotFoundError: No module named 'hydra'`**、**LIBERO 首次运行时的交互提示**、**MuJoCo / OpenGL / EGL 相关 Traceback** 等。

---

## 2. 根因（为何「看起来在调度」却几乎不占 GPU）

评测流水线依赖 `run_libero_parallel_test.sh`：**用 tmux 在多个 pane 里启动 `eval_libero_single.py`**。若任一步在子 shell 中失败或未真正启动 Python，则：

- 主进程 / 调度脚本仍可能继续打印「已分配 GPU、已启动任务」；
- **实际仿真与模型推理进程未跑起来**，监控上就会长期接近 0。

下面分条对应我们中途遇到的问题。

---

## 3. 问题与对策

### 3.1 容器内未安装 `tmux`

**原因**：平台镜像多为精简环境，**默认不含 tmux**。脚本里凡调用 `tmux new-session` / `tmux send-keys` 都会失败，子任务根本不会启动。

**处理**（已写入 `run_libero_eval.sh`）：启动时检测 `tmux`，不存在则 `apt-get install`。

**可选**：在自定义任务「入口命令」里显式安装：

```bash
apt-get update -qq && apt-get install -y -qq tmux
```

---

### 3.2 tmux pane 内未激活 Conda（`hydra` 等缺失）

**原因**：入口命令往往是 `source .../conda.sh && conda activate fastwam && ...`，但 **tmux 新开 pane 默认不会继承你已激活的环境**；若仅依赖 `source ~/.bashrc && conda activate fastwam`，而容器里 **`.bashrc` 未做 `conda init`**，则 `conda` 不可用，`python` 落到 base 或其它环境 → **`ModuleNotFoundError: hydra`**。

**处理**（已写入 `run_libero_parallel_test.sh`）：在 `run_libero_eval()` 内根据当前环境的 **`CONDA_EXE` 推导 `conda.sh` 路径**（`CONDA_INIT_SH`），在 `tmux send-keys` 里使用：

```bash
source $CONDA_INIT_SH && conda activate fastwam && ...
```

这样不依赖容器是否把 conda 写进 `~/.bashrc`。

---

### 3.3 LIBERO 首次运行交互 / 非交互环境**

**原因**：若 `~/.libero/config.yaml` 不存在，部分 LIBERO 导入路径会 **阻塞在 `input()`**（例如询问是否自定义数据集路径）。**无 TTY 的自定义任务**无法输入，进程卡住，表现为**长时间低 CPU/GPU**。

**处理**：在任务启动前准备好 `~/.libero/config.yaml`，或在本机/容器里先跑一次 LIBERO 初始化完成配置，再提交云端任务。

---

### 3.4 PyTorch 2.6+ `torch.load` 与 `weights_only`（LIBERO 侧）

**原因**：PyTorch 2.6 起 `torch.load` 默认 `weights_only=True`，LIBERO 加载 **含 numpy 等对象的初始化状态** 可能报错（`UnpicklingError`）。

**处理**：在 **LIBERO 安装目录**（如 `libero/libero/benchmark/__init__.py`）对可信的 `init_states` 文件使用 `torch.load(..., weights_only=False)`。该修改属于 **LIBERO 源码**，不在 FastWAM 仓库内；升级 LIBERO 时需留意是否被覆盖。

---

### 3.5 MuJoCo / EGL 离屏渲染（OpenGL `eglQueryString` 等）

**原因**：仿真使用 **EGL 离屏渲染**。容器内若缺少 **EGL 相关库** 或 `LD_LIBRARY_PATH` 未包含系统 EGL 路径，会在 `OpenGL.EGL` / `mujoco.egl` 处失败，例如 **`AttributeError: 'NoneType' object has no attribute 'eglQueryString'`**。

**处理**（已写入脚本与 tmux 子命令）：

- 设置 `MUJOCO_GL=egl`；
- 将 `/usr/lib/x86_64-linux-gnu` 加入 `LD_LIBRARY_PATH`（见 `run_libero_eval.sh` 与 `run_libero_parallel_test.sh` 内 `export`）。

若仍缺库，在**入口命令**中安装（示例）：

```bash
apt-get update -qq && apt-get install -y -qq libegl1-mesa-dev
```

完整入口命令示例：

```bash
apt-get update -qq && apt-get install -y -qq tmux libegl1-mesa-dev && \
source /home/YongweiZhang/miniconda3/etc/profile.d/conda.sh && \
conda activate fastwam && \
cd /home/YongweiZhang/Github_Projects/FastWAM && \
bash experiments/libero/run_libero_eval.sh
```

（路径按你实例上的 `conda` 与仓库路径调整。）

---

## 4. 排查顺序建议

1. 看 **`evaluate_results/.../task_logs/*.log`**：真实错误在子进程日志里，不在 Hydra 主进程最后一行。
2. 确认 **`tmux` 可用**（无 `command not found`）。
3. 确认子日志里 **`python` 来自 `fastwam` 环境**（无 `hydra` 缺失）。
4. 确认无 **交互阻塞**（LIBERO 配置已就绪）。
5. 若报 **EGL / OpenGL / EGL 相关 Traceback**：按 3.5 补库与环境变量。

---

## 5. 与本仓库相关的脚本

| 文件 | 作用 |
|------|------|
| `experiments/libero/run_libero_eval.sh` | 评测入口：可选安装 tmux、设置 `MUJOCO_GL` / `LD_LIBRARY_PATH` |
| `experiments/libero/run_libero_parallel_test.sh` | tmux 多 GPU 调度、`CONDA_INIT_SH`、在 pane 内导出 EGL 相关环境 |

---

## 6. 小结

火山引擎自定义任务与本地开发机差异主要在：**精简镜像（缺 tmux、缺 EGL 包）、非交互 shell、tmux 子 shell 不自动继承 conda**。上述问题都会导致 **评测子进程未真正跑仿真与推理**，从而表现为 **CPU/GPU 利用率长期接近 0**；按日志逐项消除后，任务应能正常占满 GPU。
