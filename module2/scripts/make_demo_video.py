"""从 COCO mini 图片生成 demo 视频。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COCO_IMG = ROOT / "data" / "raw" / "coco_mini" / "images" / "val2017"
OUT = ROOT / "data" / "demo" / "coco_person_demo.mp4"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default=str(OUT))
    args = parser.parse_args()

    images = sorted(COCO_IMG.glob("*.jpg"))[: args.max_frames]
    if not images:
        raise FileNotFoundError(f"无图片: {COCO_IMG}（请确认 data/raw/coco_mini/ 已提供）")

    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
    writer.release()
    print(f"Demo 视频: {out_path} ({len(images)} 帧)")


if __name__ == "__main__":
    main()
