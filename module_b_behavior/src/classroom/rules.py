"""课堂课内行为规则（基于 YOLO-Pose 关键点）。"""

from __future__ import annotations

import numpy as np

CLASSROOM_LABELS = (
    "sit_listen",
    "raise_hand",
    "write",
    "bow_head",
    "stand",
    "unknown",
)


def infer_behavior_from_pose(kpts: np.ndarray) -> tuple[str, float]:
    if kpts.shape[0] < 17:
        return "unknown", 0.0

    def visible(idx: int) -> bool:
        return float(kpts[idx, 2]) > 0.3

    def pt(idx: int) -> np.ndarray:
        return kpts[idx, :2]

    # ============================================================
    # 姿态行为规则引擎（请补充代码）
    #
    # COCO 17 关键点索引：
    #   0=nose  5=l_shoulder  6=r_shoulder  9=l_wrist  10=r_wrist
    #   11=l_hip  12=r_hip
    #
    # 需要实现的行为判断（按优先级）：
    #   1. raise_hand —— 手腕高于同侧肩膀 30px 以上
    #      e.g. visible(lw) and pt(lw)[1] < pt(ls)[1] - 30
    #   2. write —— 躯干长度 > 120 且手腕低于肩膀 40px 以上
    #   3. stand —— 躯干长度 > 120 但未满足写字条件
    #   4. bow_head —— 鼻子 Y 坐标低于肩膀 Y 坐标 20px
    #   5. sit_listen —— 默认坐姿听讲
    #
    # 提示：
    #   - shoulder_y = (pt(ls)[1] + pt(rs)[1]) / 2
    #   - hip_y = 臀部中点或 shoulder_y + 80（臀部不可见时）
    #   - torso_len = abs(hip_y - shoulder_y)
    #   - 每个行为返回 (标签, 置信度)
    # ============================================================
    pass  # TODO: 请在此补充姿态行为规则引擎代码 


def majority_behavior(behaviors: list[str]) -> str:
    if not behaviors:
        return "unknown"
    from collections import Counter
    return Counter(behaviors).most_common(1)[0][0]
