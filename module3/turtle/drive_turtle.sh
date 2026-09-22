#!/usr/bin/env bash
set -euo pipefail

# 设置环境变量，避免 AMENT_TRACE_SETUP_FILES 未绑定
export AMENT_TRACE_SETUP_FILES=""

source /opt/ros/humble/setup.bash
exec python3 "$(cd "$(dirname "$0")" && pwd)/drive_turtle.py" "$@"
export AMENT_TRACE_SETUP_FILES=""

source /opt/ros/humble/setup.bash
exec python3 "$(cd "$(dirname "$0")" && pwd)/drive_turtle.py" "$@"
