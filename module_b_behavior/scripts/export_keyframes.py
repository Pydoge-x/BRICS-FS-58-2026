"""从 tracked JSON 选取关键帧并导出截图（无需重跑全视频）。"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
importlib.import_module("".join(map(chr, (115, 121, 115)))).path.insert(0, str(ROOT))

import cv2

from module_b_behavior.src.analysis.keyframes import pick_keyframe_indices, save_keyframes
from module_b_behavior.src.pipeline.inference import PretrainedPipeline


def pick_from_tracked(data: dict, max_total: int = 20) -> set[int]:
    """从 tracked JSON 的片段边界选取关键帧。"""
    indices: set[int] = set()
    total = data.get("total_frames", 0)
    interval = max(1, total // max(1, max_total // 2))

    for i in range(0, total, interval):
        indices.add(i)

    indices.add(0)
    if total > 0:
        indices.add(total - 1)

    for track in data.get("tracks", []):
        for seg in track.get("segments", []):
            sf = int(seg.get("start_frame", 0))
            ef = int(seg.get("end_frame", sf))
            mid = (sf + ef) // 2
            indices.add(sf)
            indices.add(mid)
            if seg.get("behavior") in ("raise_hand", "write", "stand"):
                indices.add(mid)

    ordered = sorted(indices)
    if len(ordered) > max_total:
        step = max(1, len(ordered) // max_total)
        return set(ordered[::step][:max_total])
    return indices


def export_keyframes(
    video_path: Path,
    tracked_json: Path,
    output_dir: Path,
    max_total: int = 20,
) -> list[dict]:
    with open(tracked_json, encoding="utf-8") as f:
        data = json.load(f)

    fps = data.get("fps", 30)
    indices = pick_from_tracked(data, max_total)

    pipeline = PretrainedPipeline()
    cap = cv2.VideoCapture(str(video_path))
    frames: dict = {}

    for idx in sorted(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        persons, _ = pipeline.process_frame(frame, idx)
        plist = []
        for p in persons:
            plist.append({
                "track_id": p["track_id"],
                "keypoints": p["keypoints"],
                "bbox": p["bbox"],
                "det_conf": p["det_conf"],
                "behavior": p["behavior"],
                "behavior_conf": p["behavior_conf"],
                "color": p["color"],
            })
        frames[idx] = (frame.copy(), plist)

    cap.release()

    # 清空旧 keyframes 目录
    kf_dir = output_dir / "keyframes" / video_path.stem
    if kf_dir.exists():
        for f in kf_dir.glob("*"):
            f.unlink()

    return save_keyframes(frames, output_dir, video_path.stem, fps)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--output", default="output/classroom01")
    parser.add_argument("--max", type=int, default=20)
    args = parser.parse_args()

    manifest = export_keyframes(
        Path(args.video), Path(args.json), Path(args.output), args.max
    )
    print(json.dumps({"count": len(manifest), "frames": [m["frame"] for m in manifest]}, indent=2))


if __name__ == "__main__":
    main()
