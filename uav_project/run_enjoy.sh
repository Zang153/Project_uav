#!/bin/bash
# 自动使用 conda 环境内的 C++ 库来启动 enjoy_rl.py 测试脚本
# 这可以避免 matplotlib 报 CXXABI 版本过低的错误

echo "[INFO] 正在启动模型测试..."
LD_PRELOAD=/home/zyx/miniconda3/envs/mujoco-sim/lib/libstdc++.so.6 python enjoy_rl.py
