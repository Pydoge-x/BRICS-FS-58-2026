"""课外行为标签与 Kinetics/UCF 预训练类别映射。"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from module_b_behavior.src.common.config import module_root, resolve_model_path

EXTRACURRICULAR_LABELS = (
    "sport",
    "play",
    "chat",
    "walk",
    "unknown",
)

# 校园学生行为（MMAction2 归类标注）
CAMPUS_STUDENT_LABELS = (
    "walk",
    "run",
    "sit",
    "fall",
    "roughhouse",
    "basketball",
    "soccer",
    "volleyball",
    "table_tennis",
    "gymnastics",
    "unknown",
)

CAMPUS_LABEL_ZH: dict[str, str] = {
    "walk": "行走",
    "run": "跑步",
    "sit": "坐",
    "fall": "倒地",
    "roughhouse": "打架打闹",
    "basketball": "打篮球",
    "soccer": "踢足球",
    "volleyball": "排球",
    "table_tennis": "打乒乓球",
    "gymnastics": "做体操",
    "unknown": "未知",
}

# UCF101 类名（粗映射，兼容旧 school 模式）
PRETRAIN_TO_SCHOOL: dict[str, str] = {
    "Basketball": "sport",
    "SoccerJuggling": "sport",
    "JumpingJack": "sport",
    "VolleyballSpiking": "sport",
    "TennisSwing": "sport",
    "WalkingWithDog": "walk",
    "Walking": "walk",
    "Talking": "chat",
    "TalkingOnPhone": "chat",
    "SoccerPenalty": "play",
    "PlayingGuitar": "play",
}

# Kinetics-400 关键词 → 校园课外行为（旧 school 模式）
_KINETICS_KEYWORDS: list[tuple[str, str]] = [
    ("basketball", "sport"),
    ("soccer", "sport"),
    ("volleyball", "sport"),
    ("tennis", "sport"),
    ("jumping jack", "sport"),
    ("skiing", "sport"),
    ("swimming", "sport"),
    ("skateboard", "sport"),
    ("gymnastics", "sport"),
    ("juggling", "sport"),
    ("dunking", "sport"),
    ("hurdling", "sport"),
    ("archery", "sport"),
    ("wrestling", "sport"),
    ("arm wrestling", "sport"),
    ("walking the dog", "walk"),
    ("jogging", "walk"),
    ("riding a bike", "walk"),
    ("walking", "walk"),
    ("marching", "walk"),
    ("talking", "chat"),
    ("texting", "chat"),
    ("laughing", "chat"),
    ("shaking hands", "chat"),
    ("arguing", "chat"),
    ("news anchoring", "chat"),
    ("trampoline", "play"),
    ("hopscotch", "play"),
    ("playing", "play"),
    ("soccer juggling", "play"),
    ("hide and seek", "play"),
    ("tickling", "play"),
    ("frisbee", "play"),
    ("catching or throwing", "play"),
    ("throwing ball", "play"),
    ("dodgeball", "play"),
    ("exercising", "sport"),
    ("aerobics", "sport"),
    ("yoga", "sport"),
    ("tai chi", "sport"),
    ("dancing", "play"),
    ("somersaulting", "sport"),
    ("cartwheeling", "sport"),
]


@lru_cache(maxsize=1)
def _load_campus_map() -> dict[str, Any]:
    path = module_root() / "configs" / "campus_behavior_map.yaml"
    if not path.exists():
        return {"labels": []}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"labels": []}


@lru_cache(maxsize=1)
def _campus_exact_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for item in _load_campus_map().get("labels", []):
        campus_id = item.get("id")
        if not campus_id:
            continue
        for raw in item.get("kinetics_exact", []) or []:
            index[str(raw).strip().lower()] = campus_id
        for raw in item.get("ucf_exact", []) or []:
            index[str(raw).strip().lower()] = campus_id
    return index


@lru_cache(maxsize=1)
def _campus_keyword_rules() -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for item in _load_campus_map().get("labels", []):
        campus_id = item.get("id")
        if not campus_id:
            continue
        for kw in item.get("kinetics_keywords", []) or []:
            rules.append((str(kw).strip().lower(), campus_id))
    rules.sort(key=lambda x: len(x[0]), reverse=True)
    return rules


def campus_label_zh(label_id: str | None) -> str:
    if not label_id:
        return CAMPUS_LABEL_ZH["unknown"]
    return CAMPUS_LABEL_ZH.get(label_id, label_id)


def map_kinetics_to_campus(raw_label: str) -> str:
    """将 Kinetics-400 原始类映射为校园学生行为 ID。"""
    if not raw_label:
        return "unknown"

    norm = raw_label.strip().lower()
    norm = re.sub(r"\s+", " ", norm)

    exact = _campus_exact_index()
    if norm in exact:
        return exact[norm]

    for key, campus_id in _campus_keyword_rules():
        if key in norm:
            return campus_id

    return "unknown"


def map_kinetics_scores_to_campus(
    scores: Any,
    labels: list[str],
    *,
    top_k: int = 5,
) -> tuple[str, float, str]:
    """从 Top-K 预测中选取首个可映射的校园行为（提高归类召回）。"""
    import numpy as np

    arr = np.asarray(scores).flatten()
    if arr.size == 0 or not labels:
        return "unknown", 0.0, ""

    order = arr.argsort()[::-1][:top_k]
    for idx in order:
        raw = labels[int(idx)] if int(idx) < len(labels) else f"class_{int(idx)}"
        campus = map_kinetics_to_campus(raw)
        if campus != "unknown":
            return campus, float(arr[int(idx)]), raw

    best_idx = int(order[0])
    raw = labels[best_idx] if best_idx < len(labels) else f"class_{best_idx}"
    return "unknown", float(arr[best_idx]), raw


def map_pretrain_label(raw_label: str) -> str:
    """将 Kinetics/UCF 原始类别映射为校园课外标签（旧 school 粗粒度）。"""
    if not raw_label:
        return "unknown"

    if raw_label in PRETRAIN_TO_SCHOOL:
        return PRETRAIN_TO_SCHOOL[raw_label]

    norm = raw_label.strip().lower()
    norm = re.sub(r"\s+", " ", norm)

    for key, school in _KINETICS_KEYWORDS:
        if key in norm:
            return school

    for key, school in PRETRAIN_TO_SCHOOL.items():
        if key.lower() in norm or norm in key.lower():
            return school

    return "unknown"


def load_kinetics_labels(label_file: str | Path | None = None) -> list[str]:
    """加载 Kinetics-400 原始类别表。"""
    path = resolve_model_path(
        label_file if label_file else "models/mmaction2/label_map_k400.txt"
    )
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_display_label(raw_label: str | None, label_mode: str = "kinetics") -> str:
    """按模式返回展示用语义标签。"""
    if not raw_label:
        return "unknown"
    if label_mode == "campus":
        return map_kinetics_to_campus(raw_label)
    if label_mode == "school":
        return map_pretrain_label(raw_label)
    return raw_label


def resolve_display_label_zh(behavior_id: str | None, label_mode: str = "kinetics") -> str:
    """返回中文展示名（campus 模式）。"""
    if label_mode == "campus":
        return campus_label_zh(behavior_id)
    return behavior_id or "unknown"


def get_campus_behavior_info() -> list[dict[str, Any]]:
    """供 API 返回校园行为定义与 Kinetics 映射摘要。"""
    items: list[dict[str, Any]] = []
    for entry in _load_campus_map().get("labels", []):
        items.append({
            "id": entry.get("id"),
            "name_zh": entry.get("name_zh"),
            "kinetics_exact_count": len(entry.get("kinetics_exact") or []),
            "kinetics_keywords": entry.get("kinetics_keywords") or [],
        })
    return items
