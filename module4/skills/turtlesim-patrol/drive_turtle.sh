#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
exec python3 "$(cd "$(dirname "$0")" && pwd)/drive_turtle.py" "$@"
