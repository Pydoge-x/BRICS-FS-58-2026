"""课外流水线：YOLO 检测 + ByteTrack + MMAction2 行为识别。"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from module_b_behavior.src.common.config import load_config, module_root, resolve_model_path
from module_b_behavior.src.common.segments import TrackAccumulator, build_segments
from module_b_behavior.src.common.video_io import open_video_writer, reencode_to_h264
from module_b_behavior.src.extracurricular.mmaction import MMActionRecognizer
from module_b_behavior.src.extracurricular.tracker import DetectTracker
from module_b_behavior.src.visualize.draw import draw_bbox


@dataclass
class FrameStats:
    frame_idx: int
    num_persons: int
    pipeline_ms: float
    behaviors: list[str] = field(default_factory=list)


@dataclass
class SceneBehavior:
    label: str = "unknown"
    score: float = 0.0
    raw_label: str | None = None
    label_mode: str = "kinetics"


class ExtracurricularPipeline:
    """YOLO 检测 + ByteTrack + MMAction2 片段行为识别。"""

    SCENE = "extracurricular"

    def __init__(
        self,
        config_path: str | Path | None = None,
        label_mode: str | None = None,
    ) -> None:
        root = module_root()
        cfg_path = Path(config_path) if config_path else root / "configs" / "extracurricular.yaml"
        self.cfg = load_config(cfg_path)
        if label_mode:
            self.cfg.setdefault("mmaction", {})["label_mode"] = label_mode
        self.root = root

        inf = self.cfg["inference"]
        trk = self.cfg.get("tracking", {})
        self.tracking_enabled = trk.get("enabled", True)
        self.tracker_name = trk.get("tracker", "bytetrack.yaml")
        mma = self.cfg.get("mmaction", {})
        self.clip_len = int(mma.get("clip_len", 16))
        self.infer_interval = int(mma.get("infer_interval", 48))
        self.max_tracks_per_infer = int(mma.get("max_tracks_per_infer", 2))
        self.infer_mode = str(mma.get("infer_mode", "scene"))
        self.warmup_full_video = bool(mma.get("warmup_full_video", True))

        self.tracker = DetectTracker(
            model_name=str(resolve_model_path(self.cfg["models"]["detect"])),
            conf=inf["conf"],
            iou=inf.get("iou", 0.45),
            imgsz=inf["imgsz"],
            device=inf.get("device", "cpu"),
            tracker=self.tracker_name,
            classes=inf.get("classes", [0]),
        )
        self.recognizer = MMActionRecognizer(self.cfg)

    def _bootstrap_scene(self, video_path: Path) -> SceneBehavior:
        if not self.warmup_full_video or not self.recognizer.available:
            return SceneBehavior(label_mode=self.recognizer.label_mode)
        detail = self.recognizer.predict_detailed(video_path)
        return SceneBehavior(
            label=detail.get("behavior", "unknown"),
            score=float(detail.get("score", 0.0)),
            raw_label=detail.get("raw_label"),
            label_mode=detail.get("label_mode", self.recognizer.label_mode),
        )

    def _should_run_scene_infer(self, frame_idx: int, buffer_len: int) -> bool:
        if buffer_len < self.clip_len:
            return False
        if frame_idx == self.clip_len - 1:
            return True
        return frame_idx > 0 and frame_idx % self.infer_interval == 0

    def _update_scene_behavior(
        self,
        scene_buffer: list[np.ndarray],
        frame_idx: int,
        scene_behavior: SceneBehavior,
    ) -> SceneBehavior:
        if not self._should_run_scene_infer(frame_idx, len(scene_buffer)):
            return scene_behavior
        detail = self.recognizer.predict_detailed_frames(scene_buffer[-self.clip_len :])
        return SceneBehavior(
            label=detail.get("behavior", "unknown"),
            score=float(detail.get("score", 0.0)),
            raw_label=detail.get("raw_label"),
            label_mode=detail.get("label_mode", self.recognizer.label_mode),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        scene_buffer: list[np.ndarray],
        scene_behavior: SceneBehavior,
        clip_buffers: dict[int, list[np.ndarray]],
        behavior_cache: dict[int, tuple[str, float]],
    ) -> tuple[list[dict], FrameStats, SceneBehavior]:
        # ============================================================
        # 室外流水线单帧处理（请补充代码）
        #
        # 目标：对单帧执行 YOLO 检测、场景行为推理、结果组装
        #
        # 步骤：
        #   1. 调用 self.tracker.predict(frame, persist=...) 获取跟踪结果
        #   2. 将当前帧加入 scene_buffer（超长时裁剪）
        #   3. 根据 self.infer_mode 分支处理：
        #
        #      scene 模式（推荐）：
        #       - 调用 self._update_scene_behavior() 更新场景行为
        #       - 将所有检测目标的行为缓存为 scene_behavior
        #
        #      track 模式：
        #       - 对每个检测目标裁剪人体区域并存入 clip_buffers
        #       - 按 buffer 长度排序，取前 max_tracks_per_infer 个
        #       - 对 buffer 满足 clip_len 的目标，调用 recognizer.predict(buf)
        #       - 结果存入 behavior_cache
        #
        #   4. 遍历 tracked，从 behavior_cache 获取每个人行为
        #      构造 persons 列表（每项含 track_id, bbox, det_conf,
        #      behavior, behavior_conf, color）
        #   5. 构造 FrameStats 返回
        #
        # 提示：
        #   - _crop_person(frame, bbox) 裁剪人体区域（已实现）
        #   - scene 模式：behavior_cache[det.track_id] = (label, score)
        #   - track 模式：person 裁剪后用 recognizer.predict(buf) 推理
        # ============================================================
        pass  # TODO: 请在此补充室外流水线单帧处理代码 

    def run_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        visualize: bool = True,
        export_json: bool = True,
        export_keyframes: bool = False,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total = min(total, max_frames)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        writer = None
        vis_path = output_dir / "vis" / f"{Path(video_path).stem}_tracked.mp4"
        if visualize:
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            writer = open_video_writer(vis_path, fps, (width, height))

        track_map: dict[int, TrackAccumulator] = defaultdict(
            lambda: TrackAccumulator(track_id=0)
        )
        clip_buffers: dict[int, list[np.ndarray]] = {}
        behavior_cache: dict[int, tuple[str, float]] = {}
        scene_buffer: list[np.ndarray] = []

        video_path_obj = Path(video_path)
        scene_behavior = self._bootstrap_scene(video_path_obj)

        frame_idx = 0
        all_stats: list[FrameStats] = []
        color_tracker_map: dict[int, tuple[int, int, int]] = {}
        total_pbar = max_frames if max_frames else total
        pbar = tqdm(total=total_pbar, desc=f"推理中", unit="帧")

        while True:
            if max_frames and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            persons, stats, scene_behavior = self.process_frame(
                frame, frame_idx, scene_buffer, scene_behavior,
                clip_buffers, behavior_cache,
            )
            pbar.update(1)
            pbar.set_postfix(人=stats.num_persons, 耗时=f"{stats.pipeline_ms:.0f}ms")

            track_map[frame_idx].track_id = frame_idx
            for p in persons:
                tid = p["track_id"]
                if tid not in color_tracker_map:
                    color_tracker_map[tid] = p["color"]
                acc = track_map.setdefault(tid, TrackAccumulator(track_id=tid))
                acc.frames.append(frame_idx)
                acc.behaviors.append(p["behavior"])
                acc.confidences.append(p["behavior_conf"])
                acc.det_confs.append(p["det_conf"])

            if visualize and writer:
                for p in persons:
                    draw_bbox(
                        frame,
                        p["bbox"],
                        p["label"],
                        p["behavior_conf"],
                        p["det_conf"],
                        p["color"],
                    )
                scene_text = f"Scene: {scene_behavior.label} ({scene_behavior.score:.2f})"
                cv2.putText(
                    frame, scene_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )
                writer.write(frame)

            all_stats.append(stats)
            frame_idx += 1

        pbar.close()
        cap.release()
        if writer:
            writer.release()

        if visualize and vis_path.exists():
            reencode_to_h264(vis_path)

        tracks_json = {}
        for tid, acc in track_map.items():
            if not acc.frames:
                continue
            segs = build_segments(acc, fps)
            tracks_json[str(tid)] = {
                "track_id": tid,
                "total_frames": len(acc.frames),
                "segments": segs,
            }

        avg_ms = float(np.mean([s.pipeline_ms for s in all_stats])) if all_stats else 0
        result = {
            "video_id": Path(video_path).stem,
            "media_type": "video",
            "scene": self.SCENE,
            "fps": fps,
            "resolution": [width, height],
            "total_frames": frame_idx,
            "performance": {
                "avg_pipeline_ms": round(avg_ms, 2),
                "avg_pipeline_fps": round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0,
                "total_person_detections": sum(s.num_persons for s in all_stats),
                "unique_track_ids": len(track_map),
            },
            "meta": {
                "stage": "extracurricular_yolo_mmaction",
                "model_detect": self.cfg["models"]["detect"],
                "tracker": self.tracker_name,
                "mmaction": self.recognizer.status(),
                "infer_mode": self.infer_mode,
                "infer_interval": self.infer_interval,
                "scene_behavior": {
                    "behavior": scene_behavior.label,
                    "score": scene_behavior.score,
                    "raw_label": scene_behavior.raw_label,
                    "label_mode": scene_behavior.label_mode,
                },
            },
            "tracks": tracks_json,
            "keyframes": {"count": 0, "manifest": None},
        }

        if export_json:
            json_path = output_dir / "results" / f"{Path(video_path).stem}_tracked.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            result["json_output"] = str(json_path)

        return result

    def run_image(self, *args, **kwargs) -> dict[str, Any]:
        raise ValueError("课外场景仅支持视频分析（MMAction2 需要时序片段）")


def _crop_person(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2].copy()
