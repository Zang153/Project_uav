#!/bin/bash

# 获取操作系统名称
OS="$(uname)"
# SCRIPT_PATH="code/uav_code.py"
SCRIPT_PATH="uav_control/main.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 找不到脚本文件 $SCRIPT_PATH"
    echo "请确保你在项目根目录下运行此脚本"
    exit 1
fi

echo "正在检测运行环境..."

if [ "$OS" == "Darwin" ]; then
    echo "检测到 macOS 系统"
    
    # 检查是否安装了 mjpython
    if command -v mjpython &> /dev/null; then
        echo "使用 mjpython 启动仿真 (macOS 推荐方式)..."
        mjpython "$SCRIPT_PATH"
    else
        echo "警告: 未找到 mjpython 命令。"
        echo "尝试使用 python 启动 (可能会报错)..."
        python "$SCRIPT_PATH"
    fi
else
    echo "检测到 Linux/Other 系统"
    echo "使用 python 启动仿真..."
    python "$SCRIPT_PATH"
fi

echo "仿真结束。"
