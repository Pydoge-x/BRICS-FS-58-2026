"""命令行推理入口（课堂 / 课外，图片 / 视频）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module_b_behavior.src.api.jobs import IMAGE_EXTS, VIDEO_EXTS  # noqa: E402
from module_b_behavior.src.api.schemas import MediaType  # noqa: E402
from module_b_behavior.src.pipeline.router import run_analysis  # noqa: E402


def detect_media_type(path: Path) -> MediaType:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return MediaType.IMAGE
    if ext in VIDEO_EXTS:
        return MediaType.VIDEO
    raise ValueError(f"不支持的媒体格式: {ext}（支持图片/视频）")


def main() -> None:
    parser = argparse.ArgumentParser(description="模块 B 命令行推理（备份版：课堂 + 课外）")
    parser.add_argument("--media", required=True, help="图片或视频路径")
    parser.add_argument("--output", default="output/infer", help="输出目录")
    parser.add_argument(
        "--scene",
        choices=["classroom", "extracurricular"],
        default="classroom",
        help="课堂=YOLO-Pose+规则；课外=MMAction2",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="视频最大处理帧数")
    parser.add_argument("--no-vis", action="store_true", help="不生成标注图/视频")
    parser.add_argument("--no-json", action="store_true", help="不导出 JSON")
    parser.add_argument("--no-keyframes", action="store_true", help="不导出关键帧（仅课堂视频）")
    args = parser.parse_args()

    media = Path(args.media)
    if not media.exists():
        raise FileNotFoundError(f"文件不存在: {media}")

    label_mode = "kinetics" if args.scene == "extracurricular" else None
    result = run_analysis(
        args.scene,
        detect_media_type(media),
        media,
        args.output,
        visualize=not args.no_vis,
        export_json=not args.no_json,
        export_keyframes=not args.no_keyframes,
        max_frames=args.max_frames,
        label_mode=label_mode,
    )
    summary = {k: v for k, v in result.items() if k != "tracks"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
