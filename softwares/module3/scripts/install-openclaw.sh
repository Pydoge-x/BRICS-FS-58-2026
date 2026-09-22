#!/usr/bin/env bash
# 加载 OpenClaw Docker 镜像并启动 Gateway（默认 18789）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAR="${ROOT}/openclaw/openclaw-image.tar"
IMAGE="${OPENCLAW_IMAGE:-eia/openclaw:latest}"
NAME="${OPENCLAW_CONTAINER:-lab-openclaw}"
PORT="${OPENCLAW_PORT:-18789}"

DOCKER=docker
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

if [[ -f "${TAR}" ]]; then
  echo "==> 加载 OpenClaw 镜像: ${TAR}"
  ${DOCKER} load -i "${TAR}"
else
  echo "未找到 ${TAR}，尝试直接使用本地镜像 ${IMAGE}"
fi

TOKEN="${OPENCLAW_GATEWAY_TOKEN:-$(openssl rand -hex 16)}"

${DOCKER} rm -f "${NAME}" >/dev/null 2>&1 || true
echo "==> 启动 ${NAME} (端口 ${PORT})"
# 官方入口脚本的 gateway start --host 在新版 CLI 上会失败；这里前台启动并启用 token。
${DOCKER} run -d --name "${NAME}" \
  -p "${PORT}:18789" \
  -e OPENCLAW_GATEWAY_TOKEN="${TOKEN}" \
  -e OPENCLAW_PRIMARY_MODEL="${OPENCLAW_PRIMARY_MODEL:-deepseek/deepseek-chat}" \
  ${DEEPSEEK_API_KEY:+-e DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY}"} \
  --entrypoint bash \
  "${IMAGE}" \
  -lc "openclaw gateway --bind lan --port 18789 --allow-unconfigured --auth token --token \"\${OPENCLAW_GATEWAY_TOKEN}\" 2>&1 | tee /tmp/gw.log"

echo "Gateway Token: ${TOKEN}"
echo "等待 Gateway ..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    echo "OpenClaw Gateway 就绪: http://127.0.0.1:${PORT}/"
    exit 0
  fi
  sleep 2
done
echo "警告: 端口已映射，但 HTTP 尚未就绪，请查看: ${DOCKER} logs ${NAME}" >&2
exit 0
