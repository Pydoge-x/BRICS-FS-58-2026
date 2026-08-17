"""统一推理入口（兼容旧代码，默认路由至课堂流水线）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from module_b_behavior.src.classroom.pipeline import ClassroomPipeline
from module_b_behavior.src.common.config import load_config
from module_b_behavior.src.common.segments import TrackAccumulator

# 向后兼容别名
PretrainedPipeline = ClassroomPipeline


def run_pretrained(
    video_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    visualize: bool = True,
    export_json: bool = True,
    export_keyframes: bool = True,
    max_frames: int | None = None,
) -> dict[str, Any]:
    pipeline = ClassroomPipeline(config_path)
    return pipeline.run_video(
        video_path, output_dir, visualize, export_json, export_keyframes, max_frames
    )


def run_pretrained_image(
    image_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None = None,
    visualize: bool = True,
    export_json: bool = True,
) -> dict[str, Any]:
    pipeline = ClassroomPipeline(config_path)
    return pipeline.run_image(image_path, output_dir, visualize, export_json)


__all__ = [
    "PretrainedPipeline",
    "ClassroomPipeline",
    "TrackAccumulator",
    "load_config",
    "run_pretrained",
    "run_pretrained_image",
]
