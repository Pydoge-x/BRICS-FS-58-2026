"""人体姿态估计（Ultralytics YOLO-Pose 预训练）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from module_b_behavior.src.common.config import resolve_model_path


@dataclass
class PoseResult:
    """单人姿态结果。"""

    keypoints: np.ndarray  # (17, 3) x, y, conf
    bbox: tuple[float, float, float, float] | None
    confidence: float


class PoseEstimator:
    """YOLO-Pose 封装。"""

    def __init__(
        self,
        model_name: str = "models/yolo/yolov8n-pose.pt",
        conf: float = 0.25,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(str(resolve_model_path(model_name)))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def predict(self, frame: np.ndarray) -> list[PoseResult]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        poses: list[PoseResult] = []
        if results.keypoints is None:
            return poses

        for i, kpt in enumerate(results.keypoints.data.cpu().numpy()):
            box = results.boxes[i] if results.boxes is not None else None
            bbox = tuple(box.xyxy[0].tolist()) if box is not None else None
            conf = float(box.conf[0]) if box is not None else 0.5
            poses.append(PoseResult(keypoints=kpt, bbox=bbox, confidence=conf))
        return poses

    @staticmethod
    def visible_keypoint_ratio(keypoints: np.ndarray, thresh: float = 0.3) -> float:
        if keypoints.size == 0:
            return 0.0
        return float(np.mean(keypoints[:, 2] > thresh))
