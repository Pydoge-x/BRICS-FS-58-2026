"""课堂流水线：YOLO-Pose + ByteTrack + 规则行为。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from src.analysis.keyframes import pick_keyframe_indices, save_keyframes
from src.classroom.rules import majority_behavior
from src.common.config import load_config, module_root, resolve_model_path
from src.common.segments import TrackAccumulator, build_segments
from src.common.video_io import open_video_writer, reencode_to_h264
from src.track.tracker import PoseTracker
from src.visualize.draw import draw_person


@dataclass
class FrameStats:
    frame_idx: int
    num_persons: int
    pipeline_ms: float
    behaviors: list[str] = field(default_factory=list)


class ClassroomPipeline:
    """YOLO-Pose + ByteTrack + 规则行为 + 可选关键帧。"""

    SCENE = "classroom"

    def __init__(self, config_path: str | Path | None = None) -> None:
        root = module_root()
        cfg_path = Path(config_path) if config_path else root / "configs" / "classroom.yaml"
        if not cfg_path.exists():
            cfg_path = root / "configs" / "pretrained.yaml"
        self.cfg = load_config(cfg_path)
        self.root = root

        inf = self.cfg["inference"]
        trk = self.cfg.get("tracking", {})
        self.tracking_enabled = trk.get("enabled", True)
        self.tracker_name = trk.get("tracker", "bytetrack.yaml")
        self.keyframe_interval = int(self.cfg.get("keyframes", {}).get("interval", 150))

        self.tracker = PoseTracker(
            model_name=str(resolve_model_path(self.cfg["models"]["pose"])),
            conf=inf["conf"],
            iou=inf.get("iou", 0.45),
            imgsz=inf["imgsz"],
            device=inf.get("device", "cpu"),
            tracker=self.tracker_name,
        )

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[list[dict], FrameStats]:
        t0 = time.perf_counter()
        tracked = self.tracker.predict(frame, persist=self.tracking_enabled)
        elapsed = (time.perf_counter() - t0) * 1000

        persons = []
        behaviors = []
        for p in tracked:
            label = f"#{p.track_id} {p.behavior}"
            persons.append({
                "track_id": p.track_id,
                "keypoints": p.keypoints,
                "bbox": p.bbox,
                "det_conf": p.det_conf,
                "behavior": p.behavior,
                "label": label,
                "behavior_conf": p.behavior_conf,
                "color": p.color,
            })
            behaviors.append(p.behavior)

        stats = FrameStats(
            frame_idx=frame_idx,
            num_persons=len(persons),
            pipeline_ms=elapsed,
            behaviors=behaviors,
        )
        return persons, stats

    def run_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        visualize: bool = True,
        export_json: bool = True,
        export_keyframes: bool = True,
        max_frames: int | None = None,
    ) -> dict[str, Any]:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg["clip"]["fps"]
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        vis_path = output_dir / "vis" / f"{video_path.stem}_tracked.mp4"
        if visualize:
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            writer = open_video_writer(vis_path, fps, (w, h))

        track_map: dict[int, TrackAccumulator] = {}
        frame_stats: list[FrameStats] = []
        keyframe_candidates: dict[int, tuple[np.ndarray, list[dict]]] = {}
        total_pbar = min(total, max_frames) if max_frames is not None else total
        frame_idx = 0
        pbar = tqdm(total=total_pbar, desc=f"推理中", unit="帧")

        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            persons, stats = self.process_frame(frame, frame_idx)
            frame_stats.append(stats)
            pbar.update(1)
            pbar.set_postfix(人=stats.num_persons, 耗时=f"{stats.pipeline_ms:.0f}ms")

            for p in persons:
                tid = p["track_id"]
                if tid not in track_map:
                    track_map[tid] = TrackAccumulator(track_id=tid)
                acc = track_map[tid]
                acc.frames.append(frame_idx)
                acc.behaviors.append(p["behavior"])
                acc.confidences.append(p["behavior_conf"])
                acc.det_confs.append(p["det_conf"])

                draw_person(
                    frame,
                    p["bbox"],
                    p["keypoints"],
                    p["label"],
                    p["behavior_conf"],
                    p["det_conf"],
                    p["color"],
                )

            if writer is not None:
                cv2.putText(
                    frame,
                    f"classroom | frame {frame_idx} | persons {len(persons)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                writer.write(frame)

            frame_idx += 1

        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()
        if visualize and vis_path.exists():
            reencode_to_h264(vis_path)

        kf_manifest = []
        if export_keyframes and frame_stats:
            stats_dicts = [
                {"frame_idx": s.frame_idx, "num_persons": s.num_persons, "behaviors": s.behaviors}
                for s in frame_stats
            ]
            max_kf = int(self.cfg.get("keyframes", {}).get("max_total", 20))
            kf_indices = pick_keyframe_indices(
                stats_dicts, interval=self.keyframe_interval, max_total=max_kf
            )
            cap2 = cv2.VideoCapture(str(video_path))
            for target in sorted(kf_indices):
                cap2.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, kf_frame = cap2.read()
                if not ok:
                    continue
                kf_persons, _ = self.process_frame(kf_frame, target)
                keyframe_candidates[target] = (kf_frame.copy(), kf_persons)
            cap2.release()
            if keyframe_candidates:
                kf_manifest = save_keyframes(
                    keyframe_candidates, output_dir, video_path.stem, fps
                )

        total_ms = sum(s.pipeline_ms for s in frame_stats)
        avg_fps = frame_idx / (total_ms / 1000) if total_ms > 0 else 0.0

        result: dict[str, Any] = {
            "video_id": video_path.stem,
            "media_type": "video",
            "scene": self.SCENE,
            "fps": fps,
            "resolution": [w, h],
            "total_frames": frame_idx,
            "performance": {
                "avg_pipeline_ms": round(float(np.mean([s.pipeline_ms for s in frame_stats])), 2) if frame_stats else 0,
                "avg_pipeline_fps": round(avg_fps, 2),
                "total_person_detections": sum(s.num_persons for s in frame_stats),
                "unique_track_ids": len(track_map),
            },
            "meta": {
                "stage": "classroom_yolo_pose_rules",
                "model_pose": self.cfg["models"]["pose"],
                "tracker": self.tracker_name if self.tracking_enabled else "none",
            },
            "tracks": [],
            "keyframes": {
                "count": len(kf_manifest),
                "manifest": str(output_dir / "keyframes" / video_path.stem / "keyframes_manifest.json"),
            },
        }

        for tid, acc in sorted(track_map.items()):
            segs = build_segments(acc, fps, min_frames=self.cfg["clip"].get("min_frames", 15))
            if not segs:
                continue
            result["tracks"].append({"track_id": tid, "segments": segs})

        if export_json:
            json_path = output_dir / "results" / f"{video_path.stem}_tracked.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["json_output"] = str(json_path)
        if visualize:
            result["visualization"] = str(vis_path)

        return result

    def run_image(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        visualize: bool = True,
        export_json: bool = True,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        h, w = frame.shape[:2]
        persons, stats = self.process_frame(frame, 0)

        vis_path = None
        if visualize:
            vis_path = output_dir / "vis" / f"{image_path.stem}_annotated.jpg"
            vis_path.parent.mkdir(parents=True, exist_ok=True)
            annotated = frame.copy()
            for p in persons:
                draw_person(
                    annotated,
                    p["bbox"],
                    p["keypoints"],
                    p["label"],
                    p["behavior_conf"],
                    p["det_conf"],
                    p["color"],
                )
            cv2.imwrite(str(vis_path), annotated)

        tracks = []
        for p in persons:
            tracks.append({
                "track_id": p["track_id"],
                "segments": [{
                    "behavior": p["behavior"],
                    "behavior_confidence": round(p["behavior_conf"], 3),
                    "detection_confidence_mean": round(p["det_conf"], 3),
                    "start_frame": 0,
                    "end_frame": 0,
                    "start_timestamp": "00:00.000",
                    "end_timestamp": "00:00.000",
                    "duration_sec": 0,
                }],
            })

        result: dict[str, Any] = {
            "video_id": image_path.stem,
            "media_type": "image",
            "scene": self.SCENE,
            "fps": 0,
            "resolution": [w, h],
            "total_frames": 1,
            "performance": {
                "avg_pipeline_ms": round(stats.pipeline_ms, 2),
                "avg_pipeline_fps": round(1000 / stats.pipeline_ms, 2) if stats.pipeline_ms else 0,
                "total_person_detections": stats.num_persons,
                "unique_track_ids": len(persons),
            },
            "meta": {
                "stage": "classroom_yolo_pose_rules",
                "model_pose": self.cfg["models"]["pose"],
                "tracker": self.tracker_name if self.tracking_enabled else "none",
            },
            "persons": [
                {
                    "track_id": p["track_id"],
                    "behavior": p["behavior"],
                    "behavior_confidence": round(p["behavior_conf"], 3),
                    "detection_confidence": round(p["det_conf"], 3),
                    "bbox": list(p["bbox"]) if p["bbox"] else None,
                }
                for p in persons
            ],
            "tracks": tracks,
            "keyframes": {"count": 0, "manifest": None},
        }

        if export_json:
            json_path = output_dir / "results" / f"{image_path.stem}_annotated.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            result["json_output"] = str(json_path)
        if visualize and vis_path:
            result["visualization"] = str(vis_path)

        return result
