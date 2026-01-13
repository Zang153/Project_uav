#!/bin/bash

# Script to run the UAV simulation

# Check for mjpython (required for macOS GUI)
if command -v mjpython &> /dev/null; then
    PYTHON_CMD="mjpython"
else
    # Try to find it in common conda locations if not in path
    if [ -f "$CONDA_PREFIX/bin/mjpython" ]; then
        PYTHON_CMD="$CONDA_PREFIX/bin/mjpython"
    else
        PYTHON_CMD="python3"
    fi
fi

echo "Using python command: $PYTHON_CMD"

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Project root is the parent directory
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Running simulation from project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
$PYTHON_CMD -m uav_project.main
