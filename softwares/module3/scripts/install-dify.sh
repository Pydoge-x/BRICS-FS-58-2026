#!/usr/bin/env bash
# 加载 Dify 离线镜像并 docker compose up
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIFY_DIR="${ROOT}/dify"
TAR="${DIFY_DIR}/images/dify-images.tar"
CONFIG="${DIFY_DIR}/config"

DOCKER=docker
if ! docker info >/dev/null 2>&1; then
  DOCKER="sudo docker"
fi

if [[ ! -f "${TAR}" ]]; then
  echo "错误: 缺少 ${TAR}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}/docker-compose.yaml" ]]; then
  echo "错误: 缺少 ${CONFIG}/docker-compose.yaml" >&2
  exit 1
fi

echo "==> 加载 Dify 镜像 ..."
${DOCKER} load -i "${TAR}"

if [[ ! -f "${CONFIG}/.env" ]]; then
  cp "${CONFIG}/.env.example" "${CONFIG}/.env"
fi

mkdir -p "${CONFIG}/volumes/app/storage" \
         "${CONFIG}/volumes/plugin_daemon" \
         "${CONFIG}/volumes/db/data" \
         "${CONFIG}/volumes/redis/data" \
         "${CONFIG}/volumes/weaviate" \
         "${CONFIG}/volumes/sandbox/dependencies" \
         "${CONFIG}/volumes/sandbox/conf"

# DinD / 非 root 时尽量放宽卷权限
chmod -R a+rwX "${CONFIG}/volumes" 2>/dev/null || true

echo "==> 启动 Dify ..."
cd "${CONFIG}"
${DOCKER} compose up -d

PORT=$(grep -E '^EXPOSE_NGINX_PORT=' .env 2>/dev/null | cut -d= -f2 || echo 3000)
echo ""
echo "Dify: http://127.0.0.1:${PORT}/install  （首次初始化）"
echo "      http://127.0.0.1:${PORT}/signin"
echo "状态: cd ${CONFIG} && ${DOCKER} compose ps"
