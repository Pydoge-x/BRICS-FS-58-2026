"""人体检测（Ultralytics YOLO 预训练）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from module_b_behavior.src.common.config import resolve_model_path


@dataclass
class PersonDetection:
    """单人检测结果。"""

    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2
    confidence: float
    class_id: int = 0


class PersonDetector:
    """YOLO 人体检测封装。"""

    def __init__(
        self,
        model_name: str = "models/yolo/yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        person_class_id: int = 0,
    ) -> None:
        self.model = YOLO(str(resolve_model_path(model_name)))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.person_class_id = person_class_id

    def predict(self, frame: np.ndarray) -> list[PersonDetection]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )[0]
        detections: list[PersonDetection] = []
        if results.boxes is None:
            return detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                )
            )
        return detections

    def predict_batch(self, source: str | Path) -> Any:
        """Ultralytics 批量预测（用于 val）。"""
        return self.model.predict(
            source=str(source),
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=[self.person_class_id],
            device=self.device,
            verbose=False,
        )

    def validate(self, data_yaml: str | Path) -> dict[str, float]:
        """在 YOLO 格式数据集上验证，返回核心指标。"""
        metrics = self.model.val(
            data=str(data_yaml),
            split="val",
            imgsz=self.imgsz,
            batch=8,
            device=self.device,
            verbose=False,
        )
        box = metrics.box
        return {
            "map50": float(box.map50),
            "map50_95": float(box.map),
            "precision": float(box.mp),
            "recall": float(box.mr),
        }
