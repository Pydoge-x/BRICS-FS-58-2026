"""可视化：检测框、骨架、行为标签。"""

from __future__ import annotations

import cv2
import numpy as np

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]

COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255)]


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, color: tuple[int, int, int]) -> None:
    for i, j in SKELETON:
        if kpts[i, 2] > 0.3 and kpts[j, 2] > 0.3:
            p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(frame, p1, p2, color, 2)
    for i in range(min(17, len(kpts))):
        if kpts[i, 2] > 0.3:
            cv2.circle(frame, (int(kpts[i, 0]), int(kpts[i, 1])), 3, color, -1)


def draw_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    label: str,
    bconf: float,
    det_conf: float,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    text = f"{label} {bconf:.2f} det:{det_conf:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, text, (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
    )


def draw_person(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float] | None,
    kpts: np.ndarray,
    behavior: str,
    bconf: float,
    det_conf: float,
    color: tuple[int, int, int],
) -> None:
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        label = f"{behavior} {bconf:.2f} det:{det_conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    draw_skeleton(frame, kpts, color)
