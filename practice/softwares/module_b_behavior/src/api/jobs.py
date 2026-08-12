"""异步分析任务管理（内存队列 + 单线程推理）。"""

from __future__ import annotations

import json
import mimetypes
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.api.schemas import JobStatus, MediaType, Scene
from src.pipeline.router import run_analysis

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def detect_media_type(filename: str) -> MediaType:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTS:
        return MediaType.VIDEO
    raise ValueError(f"不支持的文件格式: {ext}")


@dataclass
class Job:
    job_id: str
    filename: str
    input_path: Path
    output_dir: Path
    media_type: MediaType
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=_utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    progress_message: str | None = None
    result: dict[str, Any] | None = None
    max_frames: int | None = None
    visualize: bool = True
    export_keyframes: bool = True
    scene: Scene = Scene.CLASSROOM
    label_mode: str | None = None


class JobManager:
    """管理图片/视频分析任务的生命周期。"""

    def __init__(self, root: Path, max_workers: int = 1) -> None:
        self.root = root
        self.upload_dir = root / "data" / "api" / "uploads"
        self.output_base = root / "output" / "api"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_base.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infer")
    def create_job(
        self,
        filename: str,
        file_bytes: bytes,
        media_type: MediaType | None = None,
        max_frames: int | None = None,
        visualize: bool = True,
        export_keyframes: bool = True,
        scene: Scene = Scene.CLASSROOM,
        label_mode: str | None = None,
    ) -> Job:
        mt = media_type or detect_media_type(filename)
        job_id = uuid.uuid4().hex[:12]
        safe_name = Path(filename).name
        job_upload = self.upload_dir / job_id
        job_upload.mkdir(parents=True, exist_ok=True)
        input_path = job_upload / safe_name
        input_path.write_bytes(file_bytes)

        job = Job(
            job_id=job_id,
            filename=safe_name,
            input_path=input_path,
            output_dir=self.output_base / job_id,
            media_type=mt,
            max_frames=max_frames,
            visualize=visualize,
            export_keyframes=export_keyframes if mt == MediaType.VIDEO else False,
            scene=scene,
            label_mode=label_mode,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run_job, job_id)
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        scene_label = "课外·MMAction2" if job.scene == Scene.EXTRACURRICULAR else "课堂"
        self._update(
            job_id,
            status=JobStatus.RUNNING,
            started_at=_utcnow(),
            progress_message=f"加载{scene_label}模型…",
        )
        try:
            if job.media_type == MediaType.IMAGE:
                self._update(job_id, progress_message="分析图片中…")
            else:
                self._update(job_id, progress_message="分析视频中…")
            result = run_analysis(
                job.scene.value,
                job.media_type,
                job.input_path,
                job.output_dir,
                visualize=job.visualize,
                export_json=True,
                export_keyframes=job.export_keyframes,
                max_frames=job.max_frames,
                label_mode=job.label_mode,
            )
            self._update(
                job_id,
                status=JobStatus.COMPLETED,
                completed_at=_utcnow(),
                progress_message="完成",
                result=result,
            )
        except Exception as exc:
            self._update(
                job_id,
                status=JobStatus.FAILED,
                completed_at=_utcnow(),
                error=str(exc),
                progress_message="失败",
            )

    def build_urls(self, job: Job, base_url: str = "") -> dict[str, str]:
        prefix = f"{base_url}/api/v1/media/{job.job_id}"
        urls: dict[str, str] = {
            "result": f"{base_url}/api/v1/jobs/{job.job_id}/result",
            "status": f"{base_url}/api/v1/jobs/{job.job_id}",
            "original": f"{prefix}/original",
        }
        if job.result and job.result.get("visualization"):
            urls["annotated"] = f"{prefix}/annotated"
            if job.media_type == MediaType.VIDEO:
                urls["video"] = urls["annotated"]
            else:
                urls["image"] = urls["annotated"]
        if job.media_type == MediaType.VIDEO:
            kf_dir = job.output_dir / "keyframes"
            if kf_dir.exists():
                urls["keyframes_manifest"] = f"{prefix}/keyframes/manifest"
        return urls

    def build_summary(self, job: Job) -> dict[str, Any] | None:
        if not job.result:
            return None
        r = job.result
        behavior_counts: dict[str, int] = {}
        tracks = r.get("tracks", [])
        if isinstance(tracks, dict):
            track_list = list(tracks.values())
        else:
            track_list = tracks
        for track in track_list:
            if not isinstance(track, dict):
                continue
            for seg in track.get("segments", []):
                b = seg.get("behavior", "unknown")
                behavior_counts[b] = behavior_counts.get(b, 0) + 1
        summary = {
            "scene": r.get("scene", job.scene.value),
            "label_mode": job.label_mode or r.get("meta", {}).get("mmaction", {}).get("label_mode"),
            "media_type": r.get("media_type", job.media_type.value),
            "video_id": r.get("video_id"),
            "total_frames": r.get("total_frames"),
            "fps": r.get("fps"),
            "resolution": r.get("resolution"),
            "performance": r.get("performance"),
            "person_count": r.get("performance", {}).get("unique_track_ids", 0),
            "track_count": len(r.get("tracks", [])),
            "behavior_segment_counts": behavior_counts,
            "keyframes_count": r.get("keyframes", {}).get("count", 0),
        }
        scene_beh = r.get("meta", {}).get("scene_behavior")
        if scene_beh:
            summary["scene_behavior"] = scene_beh
        return summary

    def load_result_json(self, job: Job) -> dict[str, Any] | None:
        if job.result:
            return job.result
        json_dir = job.output_dir / "results"
        if not json_dir.exists():
            return None
        files = list(json_dir.glob("*_tracked.json"))
        if not files:
            files = list(json_dir.glob("*_annotated.json"))
        if not files:
            files = list(json_dir.glob("*.json"))
        if not files:
            return None
        with open(files[0], encoding="utf-8") as f:
            return json.load(f)

    def get_original_path(self, job: Job) -> Path | None:
        if job.input_path.exists():
            return job.input_path
        return None

    def get_annotated_path(self, job: Job) -> Path | None:
        if job.result and job.result.get("visualization"):
            p = Path(job.result["visualization"])
            if p.exists():
                return p
        vis_dir = job.output_dir / "vis"
        if vis_dir.exists():
            for pattern in ("*_annotated.jpg", "*.jpg", "*.mp4", "*.webm"):
                files = list(vis_dir.glob(pattern))
                if files:
                    return files[0]
        return None

    def get_video_path(self, job: Job) -> Path | None:
        p = self.get_annotated_path(job)
        if p and p.suffix.lower() in VIDEO_EXTS:
            return p
        return None

    def get_image_path(self, job: Job) -> Path | None:
        p = self.get_annotated_path(job)
        if p and p.suffix.lower() in IMAGE_EXTS:
            return p
        return None

    def get_keyframes_dir(self, job: Job) -> Path | None:
        kf = job.output_dir / "keyframes"
        if kf.exists():
            subdirs = [d for d in kf.iterdir() if d.is_dir()]
            if subdirs:
                return subdirs[0]
        return None

    @staticmethod
    def guess_mime(path: Path) -> str:
        mime, _ = mimetypes.guess_type(str(path))
        return mime or "application/octet-stream"

    def delete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        upload = self.upload_dir / job_id
        if upload.exists():
            shutil.rmtree(upload, ignore_errors=True)
        if job.output_dir.exists():
            shutil.rmtree(job.output_dir, ignore_errors=True)
        with self._lock:
            self._jobs.pop(job_id, None)
        return True
