"""
GPU keepalive: 在后台周期性地在每张可见GPU上执行少量矩阵运算，
使火山引擎监控系统检测到GPU利用率 > 20%，防止自动关机。

用法:
    # 后台运行（默认对所有可见GPU保活）
    nohup python eva_bash/gpu_keepalive.py &

    # 仅对指定GPU保活
    CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python eva_bash/gpu_keepalive.py &

    # 停止
    kill $(cat /tmp/gpu_keepalive.pid)
"""

import os
import signal
import sys
import time

import torch

PID_FILE = "/tmp/gpu_keepalive.pid"
# 每隔多少秒做一次GPU运算
INTERVAL = 5
# 矩阵大小 — 足够让GPU利用率被采样到，但不影响正常推理
MATRIX_SIZE = 2048
# 每次做多少轮matmul
NUM_ITERS = 50


def _write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _cleanup(*_):
    try:
        os.remove(PID_FILE)
    except FileNotFoundError:
        pass
    sys.exit(0)


def main():
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)
    _write_pid()

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU found, exiting.")
        return

    print(f"GPU keepalive started (pid={os.getpid()}), {num_gpus} GPU(s), "
          f"interval={INTERVAL}s, matrix={MATRIX_SIZE}, iters={NUM_ITERS}")

    # 预分配张量
    tensors = []
    for i in range(num_gpus):
        a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=f"cuda:{i}", dtype=torch.float16)
        b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=f"cuda:{i}", dtype=torch.float16)
        tensors.append((a, b))

    while True:
        for i, (a, b) in enumerate(tensors):
            for _ in range(NUM_ITERS):
                torch.mm(a, b)
            torch.cuda.synchronize(i)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
