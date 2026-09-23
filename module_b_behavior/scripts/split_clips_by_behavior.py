"""按行为片段切分视频 clip，写入阶段 C 校园数据集（弱监督）。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SCHOOL = ROOT / "data" / "raw" / "school" / "classroom"
MIN_FRAMES = 15
MAX_FRAMES = 240          # 单 clip 最长约 12s @20fps
SKIP_BEHAVIORS = {"unknown"}
# sit_listen 已有整段视频；仅导出区分度高的行为 clip
EXPORT_BEHAVIORS = {"raise_hand", "write", "stand", "bow_head"}


def extract_clip(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    out_path: Path,
    fps: float,
    size: tuple[int, int],
) -> bool:
    w, h = size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 0


def split_by_behavior(
    video_path: Path,
    tracked_json: Path,
    min_frames: int = MIN_FRAMES,
) -> dict:
    with open(tracked_json, encoding="utf-8") as f:
        data = json.load(f)

    fps = data.get("fps", 30)
    w, h = data.get("resolution", [1920, 1200])
    stem = video_path.stem
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    manifest: dict = {"clips": [], "counts": {}}
    if (ROOT / "annotations" / "school_manifest.json").exists():
        with open(ROOT / "annotations" / "school_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
    manifest.setdefault("samples", {})
    manifest.setdefault("clips_from_segments", [])

    exported = 0
    for track in data.get("tracks", []):
        tid = track["track_id"]
        for i, seg in enumerate(track.get("segments", [])):
            beh = seg.get("behavior", "unknown")
            if beh in SKIP_BEHAVIORS:
                continue
            sf = int(seg.get("start_frame", 0))
            ef = int(seg.get("end_frame", sf))
            if beh not in EXPORT_BEHAVIORS:
                continue
            if ef - sf + 1 < min_frames:
                continue
            if ef - sf + 1 > MAX_FRAMES:
                ef = sf + MAX_FRAMES - 1

            out_name = f"{stem}_t{tid}_s{i}_{beh}_f{sf}-{ef}.mp4"
            out_path = SCHOOL / beh / out_name
            if extract_clip(cap, sf, ef, out_path, fps, (w, h)):
                rel = str(out_path.relative_to(ROOT / "data" / "raw" / "school")).replace("\\", "/")
                manifest["samples"].setdefault(beh, [])
                if rel not in manifest["samples"][beh]:
                    manifest["samples"][beh].append(rel)
                manifest["clips_from_segments"].append({
                    "file": rel,
                    "track_id": tid,
                    "behavior": beh,
                    "start_frame": sf,
                    "end_frame": ef,
                    "duration_sec": seg.get("duration_sec"),
                })
                exported += 1

    cap.release()
    manifest["counts"] = {k: len(v) for k, v in manifest["samples"].items()}
    manifest_path = ROOT / "annotations" / "school_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {"exported_clips": exported, "manifest": str(manifest_path), "counts": manifest["counts"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="按行为切分 clip 到 school 数据集")
    parser.add_argument("--video", required=True)
    parser.add_argument("--json", required=True, help="tracked JSON")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = parser.parse_args()

    info = split_by_behavior(Path(args.video), Path(args.json), args.min_frames)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
