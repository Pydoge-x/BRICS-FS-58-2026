#!/usr/bin/env bash
# 在 Ubuntu 22.04 (jammy) 上安装 ROS 2 Humble Desktop + turtlesim
# 优先使用 packages/ros2/debs 离线包；无离线包时走清华 apt 源。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEB_DIR="${ROOT}/ros2/debs"
KEYRING=/usr/share/keyrings/ros-archive-keyring.gpg

export DEBIAN_FRONTEND=noninteractive

if [[ "$(. /etc/os-release && echo "$VERSION_CODENAME")" != "jammy" ]]; then
  echo "警告: 当前非 Ubuntu 22.04 jammy，安装可能失败。" >&2
fi

if [[ -d "${DEB_DIR}" ]] && compgen -G "${DEB_DIR}/*.deb" >/dev/null; then
  echo "==> 离线安装 ROS 2 Humble（debs）..."
  sudo dpkg -i "${DEB_DIR}"/*.deb || true
  sudo apt-get install -y -f
else
  echo "==> 在线安装 ROS 2 Humble（清华源）..."
  sudo apt-get update -qq
  sudo apt-get install -y curl gnupg lsb-release ca-certificates
  if [[ -f "${ROOT}/ros2/ros.key" ]]; then
    sudo cp "${ROOT}/ros2/ros.key" "${KEYRING}"
  else
    curl -fsSL https://mirrors.tuna.tsinghua.edu.cn/rosdistro/ros.key \
      | sudo gpg --dearmor -o "${KEYRING}"
  fi
  echo "deb [arch=$(dpkg --print-architecture) signed-by=${KEYRING}] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu jammy main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y ros-humble-desktop ros-humble-turtlesim ros-dev-tools
fi

grep -q '/opt/ros/humble/setup.bash' "${HOME}/.bashrc" 2>/dev/null \
  || echo 'source /opt/ros/humble/setup.bash' >> "${HOME}/.bashrc"

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
ros2 pkg executables turtlesim | head -5
echo "ROS 2 Humble 安装完成。验证: source /opt/ros/humble/setup.bash && ros2 run turtlesim turtlesim_node"
