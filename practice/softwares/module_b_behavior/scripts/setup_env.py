"""一键环境检查：确认 demo 数据就绪 → benchmark。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def check_data() -> None:
    required = [
        ROOT / "data" / "demo" / "coco_person_demo.mp4",
        ROOT / "data" / "demo" / "classroom01_clip_8min_1min.mp4",
        ROOT / "models" / "yolo" / "yolov8n-pose.pt",
    ]
    print("\n>>> 检查 demo 与推理权重", flush=True)
    for path in required:
        if path.exists():
            print(f"  OK: {path.relative_to(ROOT)}", flush=True)
        else:
            raise FileNotFoundError(f"缺少: {path.relative_to(ROOT)}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="推理版环境检查并运行 benchmark")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--max-video-frames", type=int, default=30)
    args = parser.parse_args()

    check_data()

    demo = ROOT / "data" / "demo" / "coco_person_demo.mp4"
    if not demo.exists():
        run([PY, "scripts/make_demo_video.py", "--max-frames", str(min(args.max_video_frames * 2, 60))])

    run([
        PY, "scripts/run_benchmark.py",
        "--max-images", str(args.max_images),
        "--max-video-frames", str(args.max_video_frames),
    ])


if __name__ == "__main__":
    main()
