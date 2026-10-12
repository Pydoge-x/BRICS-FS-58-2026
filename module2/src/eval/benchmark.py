"""性能评测：检测、姿态、流水线。"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from module_b_behavior.src.detect.person_detector import PersonDetector
from module_b_behavior.src.pipeline.inference import PretrainedPipeline
from module_b_behavior.src.pose.pose_estimator import PoseEstimator


def eval_detection(data_yaml: Path, device: str = "cpu") -> dict:
    """YOLO 人体检测在 COCO mini 上的指标。"""
    if not data_yaml.exists():
        return {"status": "skipped", "reason": f"缺少 {data_yaml}"}

    detector = PersonDetector(device=device)
    t0 = time.perf_counter()
    metrics = detector.validate(data_yaml)
    elapsed = time.perf_counter() - t0
    metrics["status"] = "ok"
    metrics["eval_time_sec"] = round(elapsed, 2)
    return metrics


def eval_pose_on_images(image_dir: Path, max_images: int = 50, device: str = "cpu") -> dict:
    """姿态估计：可见关键点比例、检测人数。"""
    if not image_dir.exists():
        return {"status": "skipped", "reason": f"缺少 {image_dir}"}

    estimator = PoseEstimator(device=device)
    images = sorted(image_dir.glob("*.jpg"))[:max_images]
    if not images:
        return {"status": "skipped", "reason": "无图片"}

    ratios = []
    person_counts = []
    latencies = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        t0 = time.perf_counter()
        poses = estimator.predict(frame)
        latencies.append((time.perf_counter() - t0) * 1000)
        person_counts.append(len(poses))
        for p in poses:
            ratios.append(PoseEstimator.visible_keypoint_ratio(p.keypoints))

    return {
        "status": "ok",
        "num_images": len(images),
        "avg_persons_per_image": round(float(np.mean(person_counts)), 2),
        "avg_visible_kpt_ratio": round(float(np.mean(ratios)), 3) if ratios else 0,
        "avg_pose_ms": round(float(np.mean(latencies)), 2),
    }


def eval_pipeline_video(video_path: Path, max_frames: int = 30, device: str = "cpu") -> dict:
    """端到端流水线 FPS 与行为输出。"""
    if not video_path.exists():
        return {"status": "skipped", "reason": f"缺少 {video_path}"}

    cfg = ROOT / "configs" / "pretrained.yaml"
    import yaml
    with open(cfg, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["inference"]["device"] = device
    tmp_cfg = ROOT / "output" / "benchmark_pretrained.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        yaml.dump(c, f)

    pipeline = PretrainedPipeline(tmp_cfg)
    out_dir = ROOT / "output" / "benchmark"
    result = pipeline.run_video(
        video_path, out_dir, visualize=True, export_json=True, max_frames=max_frames
    )
    perf = result.get("performance", {})
    return {
        "status": "ok",
        "video": str(video_path),
        "frames_processed": result.get("total_frames", 0),
        **perf,
        "tracks": len(result.get("tracks", [])),
        "visualization": result.get("visualization"),
    }


def run_all(device: str = "cpu", max_images: int = 100, max_video_frames: int = 30) -> dict:
    data_yaml = ROOT / "annotations" / "coco_person_mini.yaml"
    image_dir = ROOT / "data" / "raw" / "coco_mini" / "images" / "val2017"
    demo_video = ROOT / "data" / "demo" / "coco_person_demo.mp4"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "detection": eval_detection(data_yaml, device),
        "pose": eval_pose_on_images(image_dir, max_images=min(50, max_images), device=device),
        "pipeline": eval_pipeline_video(demo_video, max_frames=max_video_frames, device=device),
    }
    return report


def save_report(report: dict) -> Path:
    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark_report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = out_dir / "benchmark_report.md"
    lines = [
        "# 模块 B 预训练模型 Benchmark 报告",
        "",
        f"- 时间: {report['timestamp']}",
        f"- 设备: {report['device']}",
        "",
        "## 1. 人体检测（YOLOv8n @ COCO mini）",
        "",
    ]
    det = report["detection"]
    if det.get("status") == "ok":
        lines += [
            f"| 指标 | 值 |",
            f"| :--- | ---: |",
            f"| mAP@0.5 | {det.get('map50', 0):.4f} |",
            f"| mAP@0.5:0.95 | {det.get('map50_95', 0):.4f} |",
            f"| Precision | {det.get('precision', 0):.4f} |",
            f"| Recall | {det.get('recall', 0):.4f} |",
            f"| 评估耗时 | {det.get('eval_time_sec', 0)}s |",
            "",
        ]
    else:
        lines.append(f"跳过: {det.get('reason', 'unknown')}\n")

    pose = report["pose"]
    lines += ["## 2. 姿态估计（YOLOv8n-pose @ COCO mini）", ""]
    if pose.get("status") == "ok":
        lines += [
            f"- 评测图片数: {pose['num_images']}",
            f"- 平均人数/图: {pose['avg_persons_per_image']}",
            f"- 平均可见关键点比例: {pose['avg_visible_kpt_ratio']}",
            f"- 平均推理耗时: {pose['avg_pose_ms']} ms/图",
            "",
        ]
    else:
        lines.append(f"跳过: {pose.get('reason', 'unknown')}\n")

    pipe = report["pipeline"]
    lines += ["## 3. 端到端流水线（检测+姿态+行为）", ""]
    if pipe.get("status") == "ok":
        lines += [
            f"- 视频: `{pipe.get('video')}`",
            f"- 处理帧数: {pipe.get('frames_processed')}",
            f"- 平均检测: {pipe.get('avg_detect_ms')} ms/帧",
            f"- 平均姿态: {pipe.get('avg_pose_ms')} ms/帧",
            f"- 流水线 FPS: {pipe.get('avg_pipeline_fps')}",
            f"- 可视化: `{pipe.get('visualization', '')}`",
            "",
        ]
    else:
        lines.append(f"跳过: {pipe.get('reason', 'unknown')}\n")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="预训练模型性能评测")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-images", type=int, default=100)
    parser.add_argument("--max-video-frames", type=int, default=30)
    args = parser.parse_args()

    report = run_all(args.device, args.max_images, args.max_video_frames)
    path = save_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告已保存: {path}")
    print(f"Markdown: {ROOT / 'reports' / 'benchmark_report.md'}")


if __name__ == "__main__":
    main()
