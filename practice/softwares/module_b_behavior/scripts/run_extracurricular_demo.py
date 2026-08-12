"""课外场景 demo：YOLO + MMAction2 预训练推理。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extracurricular.mmaction import MMActionRecognizer  # noqa: E402
from src.extracurricular.pipeline import ExtracurricularPipeline  # noqa: E402
from src.common.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="课外行为识别 demo（MMAction2 预训练）")
    parser.add_argument(
        "--video",
        default=str(ROOT / "data" / "demo" / "extracurricular" / "mmaction_official_demo.mp4"),
        help="输入视频路径",
    )
    parser.add_argument("--output", default=str(ROOT / "output" / "extracurricular_demo"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--mmaction-only", action="store_true", help="仅跑 MMAction2（不跑 YOLO 流水线）")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(f"视频不存在: {video}")

    cfg = load_config(ROOT / "configs" / "extracurricular.yaml")

    if args.mmaction_only:
        rec = MMActionRecognizer(cfg)
        print("MMAction 状态:", json.dumps(rec.status(), ensure_ascii=False))
        detail = rec.predict_detailed(video)
        print("识别结果:", json.dumps(detail, ensure_ascii=False, indent=2))
        return

    pipeline = ExtracurricularPipeline(ROOT / "configs" / "extracurricular.yaml")
    result = pipeline.run_video(
        video,
        args.output,
        visualize=True,
        export_json=True,
        max_frames=args.max_frames,
    )
    summary = {k: v for k, v in result.items() if k != "tracks"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
