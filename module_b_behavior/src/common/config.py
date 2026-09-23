"""配置加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def module_root() -> Path:
    return Path(__file__).resolve().parents[2]


def models_dir() -> Path:
    return module_root() / "models"


def resolve_model_path(path: str | Path) -> Path:
    """将配置中的模型路径解析为绝对路径（相对 module_root）。"""
    p = Path(path)
    if p.is_absolute():
        return p
    root = module_root()
    candidate = root / p
    if candidate.exists():
        return candidate
    # 兼容旧布局：裸文件名在项目根目录
    legacy = root / p.name
    if legacy.exists():
        return legacy
    return candidate
