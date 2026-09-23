"""兼容层：规则行为推断转发至 classroom.rules。"""

from __future__ import annotations

from module_b_behavior.src.classroom.rules import (
    CLASSROOM_LABELS,
    infer_behavior_from_pose,
    majority_behavior,
)

# 向后兼容
BEHAVIOR_LABELS = CLASSROOM_LABELS

# UCF101 类别 → 规则行为粗映射（用于弱验证）
UCF101_BEHAVIOR_HINT = {
    "WritingOnBoard": {"write", "bow_head"},
    "Reading": {"bow_head", "sit_listen"},
    "WalkingWithDog": {"stand"},
    "Yoga": {"sit_listen", "stand"},
    "Basketball": {"stand", "raise_hand"},
}

__all__ = [
    "BEHAVIOR_LABELS",
    "CLASSROOM_LABELS",
    "UCF101_BEHAVIOR_HINT",
    "infer_behavior_from_pose",
    "majority_behavior",
]
