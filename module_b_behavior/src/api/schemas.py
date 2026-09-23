"""REST API 请求/响应模型。"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class Scene(str, Enum):
    CLASSROOM = "classroom"
    EXTRACURRICULAR = "extracurricular"


class AnalyzeOptions(BaseModel):
    max_frames: int | None = Field(None, ge=1, le=10000, description="限制处理帧数，用于快速预览")
    visualize: bool = True
    export_keyframes: bool = True


class JobSummary(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    media_type: MediaType
    scene: Scene = Scene.CLASSROOM
    label_mode: str | None = Field(None, description="课外标签模式：kinetics | campus")
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    progress_message: str | None = None


class JobDetail(JobSummary):
    result_summary: dict[str, Any] | None = None
    urls: dict[str, str] | None = None


class BehaviorInfo(BaseModel):
    scene: Scene
    labels: list[str]
    description: str
    engine: str


class HealthResponse(BaseModel):
    status: str
    version: str
    stage: str
    scenes: list[Scene] = [Scene.CLASSROOM, Scene.EXTRACURRICULAR]
    supports: list[Literal["image", "video"]] = ["image", "video"]
