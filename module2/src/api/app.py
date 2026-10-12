"""模块 B RESTful API 服务。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from module_b_behavior.src.api.demo_samples import get_sample, list_samples, resolve_path
from module_b_behavior.src.api.jobs import IMAGE_EXTS, VIDEO_EXTS, Job, JobManager, detect_media_type
from module_b_behavior.src.api.schemas import BehaviorInfo, HealthResponse, JobDetail, JobStatus, JobSummary, MediaType, Scene
from module_b_behavior.src.classroom.rules import CLASSROOM_LABELS
from module_b_behavior.src.extracurricular.labels import EXTRACURRICULAR_LABELS, load_kinetics_labels

ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT / "src" / "web"

job_manager = JobManager(ROOT)

app = FastAPI(
    title="模块 B · 人体行为识别 API",
    description="课堂 YOLO-Pose + 规则 / 课外 YOLO + MMAction2 双路由行为识别",
    version="1.2.0-infer",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _job_to_summary(job: Job) -> JobSummary:
    return JobSummary(
        job_id=job.job_id,
        status=job.status,
        filename=job.filename,
        media_type=job.media_type,
        scene=job.scene,
        label_mode=job.label_mode,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        progress_message=job.progress_message,
    )


def _parse_analysis_mode(
    analysis_mode: str | None,
    scene: Scene,
) -> tuple[Scene, str | None]:
    """解析分析场景：课堂 / 课外（MMAction2）。"""
    mode = (analysis_mode or "").strip().lower()
    if mode == "classroom":
        return Scene.CLASSROOM, None
    if mode in ("extracurricular", "extracurricular_kinetics", "extra_kinetics", "kinetics"):
        return Scene.EXTRACURRICULAR, "kinetics"
    if scene == Scene.EXTRACURRICULAR:
        return Scene.EXTRACURRICULAR, "kinetics"
    return Scene.CLASSROOM, None


def _job_to_detail(job: Job) -> JobDetail:
    return JobDetail(
        **_job_to_summary(job).model_dump(),
        result_summary=job_manager.build_summary(job),
        urls=job_manager.build_urls(job) if job.status == JobStatus.COMPLETED else None,
    )


def _require_job(job_id: str) -> Job:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"任务不存在: {job_id}")
    return job


@app.get("/api/v1/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.2.0-infer",
        stage="infer_only:classroom_yolo+extracurricular_mmaction",
    )


@app.get("/api/v1/behaviors", response_model=BehaviorInfo)
def behaviors(
    scene: Scene = Query(Scene.CLASSROOM, description="场景：classroom 或 extracurricular"),
    label_mode: str = Query("kinetics", description="课外固定为 kinetics（MMAction2 预训练）"),
) -> BehaviorInfo:
    if scene == Scene.EXTRACURRICULAR:
        kinetics = load_kinetics_labels()
        return BehaviorInfo(
            scene=scene,
            labels=kinetics[:30] if kinetics else list(EXTRACURRICULAR_LABELS),
            description=(
                f"Kinetics-400 行为（共 {len(kinetics) or len(EXTRACURRICULAR_LABELS)} 类），"
                "MMAction2 TSN 预训练，仅支持视频"
            ),
            engine="yolo+mmaction2+kinetics400",
        )
    return BehaviorInfo(
        scene=scene,
        labels=list(CLASSROOM_LABELS),
        description="课堂行为标签（YOLO-Pose + 姿态规则，支持图片与视频）",
        engine="yolo_pose+rules",
    )


@app.get("/api/v1/demo-samples")
def demo_samples() -> list[dict]:
    """返回内置 demo 样本列表（一键测试）。"""
    return list_samples(ROOT)


@app.get("/api/v1/demo-samples/{sample_id}/preview")
def demo_sample_preview(sample_id: str) -> FileResponse:
    sample = get_sample(sample_id)
    if not sample or not sample.preview_path:
        raise HTTPException(404, f"样本不存在或无预览: {sample_id}")
    path = resolve_path(ROOT, sample.preview_path)
    if not path.is_file():
        raise HTTPException(404, "预览图不存在")
    return FileResponse(path, media_type=job_manager.guess_mime(path))


@app.post("/api/v1/behavior/analyze-demo", response_model=JobSummary, status_code=202)
async def analyze_demo(
    sample_id: str = Form(..., description="内置样本 ID"),
    max_frames: int | None = Form(None, description="覆盖默认最大帧数"),
    visualize: bool = Form(True),
    export_keyframes: bool = Form(True),
) -> JobSummary:
    # ============================================================
    # 内置样本分析任务创建（请补充代码）
    #
    # 目标：使用内置 demo 样本发起行为分析
    #
    # 步骤：
    #   1. 调用 get_sample(sample_id) 获取样本，不存在则 404
    #   2. 调用 resolve_path(ROOT, sample.relative_path) 获取文件路径
    #   3. 调用 _parse_analysis_mode(sample.analysis_mode, ...) 解析场景
    #   4. 调用 detect_media_type(path.name) 判断图片/视频
    #   5. 调用 job_manager.create_job(...) 创建分析任务
    #   6. 返回 _job_to_summary(job)
    #
    # 提示：
    #   - create_job 参数: filename, file_bytes, media_type,
    #     max_frames, visualize, export_keyframes, scene, label_mode
    #   - sample.max_frames 可作默认 max_frames
    # ============================================================
    pass  # TODO: 请在此补充 analyze_demo 代码 


@app.post("/api/v1/behavior/analyze", response_model=JobSummary, status_code=202)
async def analyze_behavior(
    media: UploadFile | None = File(None, description="待分析图片或视频"),
    video: UploadFile | None = File(None, description="兼容旧字段：视频"),
    max_frames: int | None = Form(None, description="限制处理帧数（仅视频）"),
    visualize: bool = Form(True),
    export_keyframes: bool = Form(True),
    scene: Scene = Form(Scene.CLASSROOM, description="兼容字段：classroom 或 extracurricular"),
    analysis_mode: str | None = Form(
        None,
        description="分析场景：classroom | extracurricular",
    ),
) -> JobSummary:
    # ============================================================
    # 上传文件创建分析任务（请补充代码）
    #
    # 目标：上传图片/视频文件并创建行为分析任务
    #
    # 步骤：
    #   1. 取 upload = media or video，检查 filename → 400
    #   2. 校验扩展名 ext in IMAGE_EXTS | VIDEO_EXTS → 400
    #   3. await upload.read() 读取内容，检查最小尺寸
    #      （图片 > 100 字节，视频 > 1024 字节）
    #   4. 调用 _parse_analysis_mode(analysis_mode, scene) 解析场景
    #   5. 课外 + 图片 → 400（MMAction2 需要时序帧）
    #   6. 调用 job_manager.create_job(...) 创建任务
    #   7. 返回 _job_to_summary(job)
    #
    # 提示：
    #   - detect_media_type(upload.filename) 判断媒体类型
    #   - create_job 参数同 analyze_demo
    # ============================================================
    pass  # TODO: 请在此补充 analyze_behavior 代码 


@app.get("/api/v1/jobs", response_model=list[JobSummary])
def list_jobs(limit: int = 20) -> list[JobSummary]:
    return [_job_to_summary(j) for j in job_manager.list_jobs(limit=limit)]


@app.get("/api/v1/jobs/{job_id}", response_model=JobDetail)
def get_job(job_id: str) -> JobDetail:
    return _job_to_detail(_require_job(job_id))


@app.get("/api/v1/jobs/{job_id}/result")
def get_job_result(job_id: str) -> JSONResponse:
    job = _require_job(job_id)
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"任务未完成，当前状态: {job.status}")
    data = job_manager.load_result_json(job)
    if not data:
        raise HTTPException(404, "结果文件不存在")
    return JSONResponse(content=data)


@app.get("/api/v1/media/{job_id}/original")
def get_original_media(job_id: str) -> FileResponse:
    """获取上传的原始图片/视频（用于预览）。"""
    job = _require_job(job_id)
    path = job_manager.get_original_path(job)
    if not path or not path.exists():
        raise HTTPException(404, "原始文件不存在")
    return FileResponse(path, media_type=job_manager.guess_mime(path), filename=path.name)


@app.get("/api/v1/media/{job_id}/annotated")
def get_annotated_media(job_id: str) -> FileResponse:
    """获取推理后的标注图片或视频。"""
    job = _require_job(job_id)
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"任务未完成，当前状态: {job.status}")
    path = job_manager.get_annotated_path(job)
    if not path or not path.exists():
        raise HTTPException(404, "标注结果不存在")
    return FileResponse(path, media_type=job_manager.guess_mime(path), filename=path.name)


@app.get("/api/v1/media/{job_id}/video")
def get_job_video(job_id: str) -> FileResponse:
    job = _require_job(job_id)
    path = job_manager.get_video_path(job)
    if not path or not path.exists():
        raise HTTPException(404, "可视化视频不存在")
    return FileResponse(path, media_type=job_manager.guess_mime(path), filename=path.name)


@app.get("/api/v1/media/{job_id}/keyframes/manifest")
def get_keyframes_manifest(job_id: str) -> JSONResponse:
    job = _require_job(job_id)
    kf_dir = job_manager.get_keyframes_dir(job)
    if not kf_dir:
        raise HTTPException(404, "关键帧不存在")
    manifest_path = kf_dir / "keyframes_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "关键帧清单不存在")
    with open(manifest_path, encoding="utf-8") as f:
        items = json.load(f)
    for item in items:
        ann_name = Path(item["annotated"]).name
        item["annotated_url"] = f"/api/v1/media/{job_id}/keyframes/{ann_name}"
        item["raw_url"] = f"/api/v1/media/{job_id}/keyframes/{Path(item['raw']).name}"
    return JSONResponse(content=items)


@app.get("/api/v1/media/{job_id}/keyframes/{filename}")
def get_keyframe_image(job_id: str, filename: str) -> FileResponse:
    job = _require_job(job_id)
    kf_dir = job_manager.get_keyframes_dir(job)
    if not kf_dir:
        raise HTTPException(404, "关键帧目录不存在")
    path = (kf_dir / filename).resolve()
    if not str(path).startswith(str(kf_dir.resolve())):
        raise HTTPException(403, "非法路径")
    if not path.exists():
        raise HTTPException(404, "图片不存在")
    return FileResponse(path, media_type=job_manager.guess_mime(path))


@app.delete("/api/v1/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    if not job_manager.delete_job(job_id):
        raise HTTPException(404, f"任务不存在: {job_id}")
    return {"message": "已删除", "job_id": job_id}


if WEB_DIR.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="web-assets")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
