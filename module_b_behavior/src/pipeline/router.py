"""按场景路由推理流水线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from module_b_behavior.src.api.schemas import MediaType

Scene = Literal["classroom", "extracurricular"]
SCENES: tuple[str, ...] = ("classroom", "extracurricular")


def get_pipeline(scene: str = "classroom", label_mode: str | None = None):
    if scene == "extracurricular":
        from module_b_behavior.src.extracurricular.pipeline import ExtracurricularPipeline
        return ExtracurricularPipeline(label_mode=label_mode)
    from module_b_behavior.src.classroom.pipeline import ClassroomPipeline
    return ClassroomPipeline()


def run_analysis(
    scene: str,
    media_type: MediaType,
    input_path: str | Path,
    output_dir: str | Path,
    *,
    visualize: bool = True,
    export_json: bool = True,
    export_keyframes: bool = True,
    max_frames: int | None = None,
    label_mode: str | None = None,
) -> dict[str, Any]:
    pipeline = get_pipeline(scene, label_mode=label_mode)
    if scene == "extracurricular" and media_type == MediaType.IMAGE:
        raise ValueError("课外场景仅支持视频，请上传视频文件或切换为课堂场景")
    if media_type == MediaType.IMAGE:
        return pipeline.run_image(input_path, output_dir, visualize=visualize, export_json=export_json)
    return pipeline.run_video(
        input_path,
        output_dir,
        visualize=visualize,
        export_json=export_json,
        export_keyframes=export_keyframes if scene == "classroom" else False,
        max_frames=max_frames,
    )
