"""MMAction2 预训练行为识别（TSN Kinetics-400）。"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from module_b_behavior.src.common.config import resolve_model_path
from module_b_behavior.src.extracurricular.labels import (
    campus_label_zh,
    map_kinetics_scores_to_campus,
    resolve_display_label,
)

# CPU 快速推理管线（相对默认 TenCrop×25 大幅提速）
_FAST_TEST_PIPELINE = [
    dict(io_backend='disk', type='DecordInit'),
    dict(clip_len=1, frame_interval=1, num_clips=3, test_mode=True, type='SampleFrames'),
    dict(type='DecordDecode'),
    dict(scale=(-1, 256), type='Resize'),
    dict(crop_size=224, type='CenterCrop'),
    dict(input_format='NCHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]


class MMActionRecognizer:
    """封装 MMAction2 TSN 推理。"""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg.get("mmaction", {})
        self.clip_len = int(self.cfg.get("clip_len", 16))
        self.input_size = int(self.cfg.get("input_size", 224))
        self.fast_mode = bool(self.cfg.get("fast_mode", True))
        self.label_mode = str(self.cfg.get("label_mode", "kinetics"))
        self.finetuned_campus = False
        self.model = None
        self.labels: list[str] = []
        self.available = False
        self.note = "mmaction2_not_installed"
        self.last_infer_ms = 0.0
        self._try_init()

    def _load_labels(self, label_file: Path) -> None:
        if label_file.exists():
            self.labels = [
                line.strip()
                for line in label_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def _try_init(self) -> None:
        config_path = self.cfg.get("config")
        checkpoint = self.cfg.get("checkpoint")
        if not config_path or not checkpoint:
            self.note = "mmaction_config_missing"
            return

        try:
            from mmaction.apis import init_recognizer  # type: ignore[import-untyped]
        except ImportError:
            return

        cfg_file = resolve_model_path(config_path)
        ckpt_file = resolve_model_path(checkpoint)
        label_path = resolve_model_path(
            self.cfg.get("label_file", "models/mmaction2/label_map_k400.txt")
        )

        if self.label_mode == "campus":
            campus_ckpt = resolve_model_path(self.cfg.get("campus_checkpoint", ""))
            campus_cfg = resolve_model_path(self.cfg.get("campus_config", ""))
            campus_labels = resolve_model_path(self.cfg.get("campus_label_file", ""))
            if campus_ckpt.exists() and campus_cfg.exists():
                cfg_file = campus_cfg
                ckpt_file = campus_ckpt
                if campus_labels.exists():
                    label_path = campus_labels
                self.finetuned_campus = True

        if not cfg_file.exists() or not ckpt_file.exists():
            self.note = "mmaction_weights_missing"
            return

        self._load_labels(label_path)

        try:
            device = self.cfg.get("device", "cpu")
            self.model = init_recognizer(str(cfg_file), str(ckpt_file), device=device)
            if self.fast_mode and not str(cfg_file).endswith("tsn_fast_cpu.py"):
                self.model.cfg.test_pipeline = _FAST_TEST_PIPELINE
            self.available = True
            if self.finetuned_campus:
                self.note = "mmaction2_campus_finetuned"
            else:
                self.note = "mmaction2_ready_fast" if self.fast_mode else "mmaction2_ready"
        except Exception as exc:
            self.note = f"mmaction_init_failed:{exc}"

    def _top_prediction(self, scores: np.ndarray) -> tuple[str, float, str]:
        if self.label_mode == "campus" and self.finetuned_campus and self.labels:
            idx = int(scores.argmax())
            display = self.labels[idx] if idx < len(self.labels) else "unknown"
            score = float(scores[idx])
            return display, score, display
        if self.label_mode == "campus" and self.labels:
            display, score, raw = map_kinetics_scores_to_campus(scores, self.labels)
            return display, score, raw

        idx = int(scores.argmax())
        raw = self.labels[idx] if self.labels and idx < len(self.labels) else f"class_{idx}"
        score = float(scores[idx])
        display = resolve_display_label(raw, self.label_mode)
        return display, score, raw

    def _result_dict(self, display: str, score: float, raw: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "behavior": display,
            "score": round(score, 3),
            "raw_label": raw,
            "label_mode": self.label_mode,
        }
        if self.label_mode == "campus":
            out["behavior_zh"] = campus_label_zh(display)
        return out

    def _infer_video_path(self, video_path: str | Path) -> tuple[str, float, str]:
        from mmaction.apis import inference_recognizer  # type: ignore[import-untyped]

        assert self.model is not None
        t0 = time.perf_counter()
        result = inference_recognizer(self.model, str(video_path))
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        scores = result.pred_score.detach().cpu().numpy()
        return self._top_prediction(scores)

    def _frames_to_temp_video(self, frames: list[np.ndarray], fps: float = 10.0) -> str:
        if not frames:
            raise ValueError("empty frames")
        h, w = frames[0].shape[:2]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        writer = cv2.VideoWriter(
            tmp.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )
        if not writer.isOpened():
            raise RuntimeError("无法创建临时视频")
        for frame in frames[-self.clip_len :]:
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        writer.release()
        return tmp.name

    def predict_video(self, video_path: str | Path) -> tuple[str, float]:
        """对完整视频片段推理，返回 (展示标签, 置信度)。"""
        if not self.available or self.model is None:
            return "unknown", 0.0
        try:
            display, score, _ = self._infer_video_path(video_path)
            return display, score
        except Exception:
            return "unknown", 0.0

    def predict(self, frames: list[np.ndarray]) -> tuple[str, float]:
        """对帧序列推理。"""
        if not frames:
            return "unknown", 0.0
        if not self.available or self.model is None:
            return "unknown", 0.0

        tmp_path = None
        try:
            tmp_path = self._frames_to_temp_video(frames)
            display, score, _ = self._infer_video_path(tmp_path)
            return display, score
        except Exception:
            return "unknown", 0.0
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def predict_detailed(self, video_path: str | Path) -> dict[str, Any]:
        """返回展示标签与 Kinetics 原始类。"""
        if not self.available or self.model is None:
            return {"behavior": "unknown", "score": 0.0, "raw_label": None, "label_mode": self.label_mode}
        try:
            display, score, raw = self._infer_video_path(video_path)
            return self._result_dict(display, score, raw)
        except Exception as exc:
            return {"behavior": "unknown", "score": 0.0, "raw_label": None, "error": str(exc), "label_mode": self.label_mode}

    def predict_detailed_frames(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """对帧序列推理。"""
        if not self.available or self.model is None:
            return {"behavior": "unknown", "score": 0.0, "raw_label": None, "label_mode": self.label_mode}
        tmp_path = None
        try:
            tmp_path = self._frames_to_temp_video(frames)
            display, score, raw = self._infer_video_path(tmp_path)
            return self._result_dict(display, score, raw)
        except Exception as exc:
            return {"behavior": "unknown", "score": 0.0, "raw_label": None, "error": str(exc), "label_mode": self.label_mode}
        finally:
            if tmp_path:
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except OSError:
                    pass

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "note": self.note,
            "fast_mode": self.fast_mode,
            "clip_len": self.clip_len,
            "num_labels": len(self.labels),
            "model": self.cfg.get("config"),
            "last_infer_ms": round(self.last_infer_ms, 1),
            "label_mode": self.label_mode,
        }
