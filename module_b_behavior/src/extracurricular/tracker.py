"""课外场景人体检测 + ByteTrack（无姿态）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from module_b_behavior.src.common.config import resolve_model_path
from module_b_behavior.src.visualize.draw import COLORS


@dataclass
class TrackedDetection:
    track_id: int
    bbox: tuple[float, float, float, float]
    det_conf: float
    color: tuple[int, int, int]


class DetectTracker:
    """YOLO 检测 + ByteTrack，仅输出 bbox。"""

    def __init__(
        self,
        model_name: str = "models/yolo/yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        tracker: str = "bytetrack.yaml",
        classes: list[int] | None = None,
    ) -> None:
        self.model = YOLO(str(resolve_model_path(model_name)))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker
        self.classes = classes or [0]

    def predict(self, frame: np.ndarray, persist: bool = True) -> list[TrackedDetection]:
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker,
            persist=persist,
            classes=self.classes,
            verbose=False,
        )[0]

        detections: list[TrackedDetection] = []
        if results.boxes is None:
            return detections

        ids = results.boxes.id
        use_track = persist and self.tracker

        for i, box in enumerate(results.boxes):
            if use_track and ids is not None:
                tid = int(ids[i].item())
                if tid < 0:
                    continue
            else:
                tid = i + 1
            bbox = tuple(box.xyxy[0].tolist())
            det_conf = float(box.conf[0])
            detections.append(
                TrackedDetection(
                    track_id=tid,
                    bbox=bbox,
                    det_conf=det_conf,
                    color=COLORS[tid % len(COLORS)],
                )
            )
        return detections

    def reset(self) -> None:
        self.model.predictor = None  # type: ignore[attr-defined]
