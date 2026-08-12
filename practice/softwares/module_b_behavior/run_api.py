"""启动模块 B REST API + Web 前端。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.app import app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="模块 B API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081, help="默认 8081（8000/8080 常被占用）")
    args = parser.parse_args()

    import uvicorn

    print(f"Web UI:  http://localhost:{args.port}")
    print(f"Swagger: http://localhost:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
