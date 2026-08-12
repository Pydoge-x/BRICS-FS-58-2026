"""ByteTrack 多目标跟踪（基于 Ultralytics YOLO-Pose）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.classroom.rules import infer_behavior_from_pose
from src.common.config import resolve_model_path
from src.visualize.draw import COLORS


@dataclass
class TrackedPerson:
    track_id: int
    keypoints: np.ndarray
    bbox: tuple[float, float, float, float] | None
    det_conf: float
    behavior: str
    behavior_conf: float
    color: tuple[int, int, int]


class PoseTracker:
    """YOLO-Pose + ByteTrack 一体化跟踪。"""

    def __init__(
        self,
        model_name: str = "models/yolo/yolov8n-pose.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        tracker: str = "bytetrack.yaml",
    ) -> None:
        self.model = YOLO(str(resolve_model_path(model_name)))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.tracker = tracker

    def predict(self, frame: np.ndarray, persist: bool = True) -> list[TrackedPerson]:
        results = self.model.track(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            tracker=self.tracker,
            persist=persist,
            verbose=False,
        )[0]

        persons: list[TrackedPerson] = []
        if results.keypoints is None or results.boxes is None:
            return persons

        # ============================================================
        # YOLO 检测结果解析（请补充代码）
        #
        # 目标：遍历检测结果，构造 TrackedPerson 列表
        #
        # 步骤：
        #   1. 获取跟踪 ID：results.boxes.id（有则用，无则按 i+1 分配）
        #   2. 遍历 results.keypoints.data.cpu().numpy()
        #   3. 对每个检测结果：
        #      - 提取 bbox（box.xyxy[0].tolist()）
        #      - 提取 det_conf（box.conf[0]）
        #      - 调用 infer_behavior_from_pose(kpt) 获取行为标签
        #      - 用 COLORS[tid % len(COLORS)] 分配颜色
        #      - 构造 TrackedPerson 加入 persons 列表
        #   4. 返回 persons
        #
        # 提示：
        #   - use_track = persist and self.tracker
        #   - TrackedPerson(track_id, keypoints, bbox, det_conf,
        #     behavior, behavior_conf, color)
        # ============================================================
        pass  # TODO: 请在此补充 YOLO 检测结果解析代码 
        return persons

    def reset(self) -> None:
        """新视频前重置跟踪器状态。"""
        # Ultralytics 在 persist=True 时依赖 stream 内部状态；显式 predict 一次空帧重置
        self.model.predictor = None  # type: ignore[attr-defined]
