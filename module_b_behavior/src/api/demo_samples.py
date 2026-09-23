"""内置 demo 样本，供 Web 一键测试。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoSample:
    id: str
    name: str
    description: str
    relative_path: str
    analysis_mode: str
    media_type: str
    max_frames: int | None = None
    preview_path: str | None = None


DEMO_SAMPLES: tuple[DemoSample, ...] = (
    DemoSample(
        id="classroom_video",
        name="课堂视频",
        description="1 分钟课堂片段 · YOLO-Pose + 6 类规则",
        relative_path="data/demo/classroom01_clip_8min_1min.mp4",
        analysis_mode="classroom",
        media_type="video",
        max_frames=120,
        preview_path="data/demo/images/classroom01_frame_00000.jpg",
    ),
    DemoSample(
        id="classroom_image",
        name="课堂图片",
        description="单帧课堂场景 · 姿态行为识别",
        relative_path="data/demo/images/classroom01_frame_00000.jpg",
        analysis_mode="classroom",
        media_type="image",
        preview_path="data/demo/images/classroom01_frame_00000.jpg",
    ),
    DemoSample(
        id="extracurricular_video",
        name="课外视频",
        description="体育场景 · MMAction2 TSN（Kinetics-400）",
        relative_path="data/demo/extracurricular/sports.mp4",
        analysis_mode="extracurricular",
        media_type="video",
        max_frames=60,
        preview_path="data/demo/extracurricular/images/sports_frame0.jpg",
    ),
)

_BY_ID = {s.id: s for s in DEMO_SAMPLES}


def get_sample(sample_id: str) -> DemoSample | None:
    return _BY_ID.get(sample_id)


def resolve_path(root: Path, relative: str) -> Path:
    return (root / relative).resolve()


def list_samples(root: Path) -> list[dict]:
    items: list[dict] = []
    for s in DEMO_SAMPLES:
        path = resolve_path(root, s.relative_path)
        item = asdict(s)
        item["available"] = path.is_file()
        item["preview_url"] = (
            f"/api/v1/demo-samples/{s.id}/preview" if s.preview_path else None
        )
        items.append(item)
    return items
