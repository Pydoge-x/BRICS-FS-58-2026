"""关键帧抽取：用于检查漏检 / 误检 / 典型行为。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from module_b_behavior.src.visualize.draw import draw_person


def save_keyframes(
    frames: dict[int, tuple[np.ndarray, list[dict]]],
    output_dir: Path,
    video_stem: str,
    fps: float,
) -> list[dict]:
    """
    保存关键帧截图（原图 + 标注图）。
    frames: frame_idx -> (frame_bgr, persons)
    """
    out = Path(output_dir) / "keyframes" / video_stem
    out.mkdir(parents=True, exist_ok=True)
    manifest = []

    for frame_idx, (frame, persons) in sorted(frames.items()):
        tag = f"frame_{frame_idx:05d}"
        raw_path = out / f"{tag}_raw.jpg"
        ann_path = out / f"{tag}_ann.jpg"

        cv2.imwrite(str(raw_path), frame)
        annotated = frame.copy()
        for p in persons:
            label = f"#{p['track_id']} {p['behavior']}"
            draw_person(
                annotated,
                p["bbox"],
                p["keypoints"],
                label,
                p["behavior_conf"],
                p["det_conf"],
                p["color"],
            )
        cv2.imwrite(str(ann_path), annotated)

        ts = frame_idx / fps if fps else 0
        manifest.append({
            "frame": frame_idx,
            "timestamp_sec": round(ts, 2),
            "num_persons": len(persons),
            "behaviors": [p["behavior"] for p in persons],
            "track_ids": [p["track_id"] for p in persons],
            "raw": str(raw_path),
            "annotated": str(ann_path),
        })

    manifest_path = out / "keyframes_manifest.json"
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def pick_keyframe_indices(
    frame_stats: list[dict],
    interval: int = 150,
    top_k_busy: int = 3,
    max_total: int = 20,
) -> set[int]:
    """选取关键帧：固定间隔 + 人数峰值 + 特殊行为，总数上限 max_total。"""
    if not frame_stats:
        return set()

    indices: set[int] = set()
    n = len(frame_stats)

    for i in range(0, n, interval):
        indices.add(frame_stats[i]["frame_idx"])

    by_count = sorted(frame_stats, key=lambda x: x["num_persons"], reverse=True)
    for item in by_count[:top_k_busy]:
        indices.add(item["frame_idx"])

    # 人数最少（且低于平均一半）的一帧
    counts = [s["num_persons"] for s in frame_stats]
    avg = sum(counts) / len(counts) if counts else 0
    low = [s for s in frame_stats if s["num_persons"] < avg * 0.5]
    if low:
        indices.add(min(low, key=lambda x: x["num_persons"])["frame_idx"])

    for item in frame_stats:
        if set(item.get("behaviors", [])) & {"raise_hand", "write", "stand"}:
            indices.add(item["frame_idx"])

    indices.add(frame_stats[0]["frame_idx"])
    indices.add(frame_stats[-1]["frame_idx"])

    # 限制总数：优先保留间隔帧与首尾，再补 busy/special
    if len(indices) > max_total:
        ordered = sorted(indices)
        step = max(1, len(ordered) // max_total)
        indices = set(ordered[::step][:max_total])
    return indices
