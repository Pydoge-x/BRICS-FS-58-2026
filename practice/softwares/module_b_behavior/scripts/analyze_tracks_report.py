"""从 tracked JSON 生成跟踪分析报告。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def build_report(data: dict, output_path: Path) -> Path:
    tracks = data.get("tracks", [])
    perf = data.get("performance", {})
    fps = data.get("fps", 30)

    behavior_counts: Counter = Counter()
    duration_by_beh: defaultdict[str, list[float]] = defaultdict(list)
    low_conf: list[dict] = []
    raise_hand_segs: list[dict] = []

    for t in tracks:
        tid = t["track_id"]
        for seg in t.get("segments", []):
            beh = seg.get("behavior", "unknown")
            behavior_counts[beh] += 1
            duration_by_beh[beh].append(seg.get("duration_sec", 0))
            conf = seg.get("behavior_confidence", 0)
            if conf < 0.5:
                low_conf.append({"track_id": tid, **seg})
            if beh == "raise_hand":
                raise_hand_segs.append({"track_id": tid, **seg})

    lines = [
        f"# 课堂视频跟踪分析报告 — {data.get('video_id', 'unknown')}",
        "",
        "## 1. 概览",
        "",
        f"| 指标 | 值 |",
        f"| :--- | ---: |",
        f"| 总帧数 | {data.get('total_frames', 0)} |",
        f"| 帧率 | {fps:.2f} fps |",
        f"| 唯一 track 数 | {perf.get('unique_track_ids', len(tracks))} |",
        f"| 流水线 FPS | {perf.get('avg_pipeline_fps', 0)} |",
        f"| 平均检测人数/帧 | {perf.get('total_person_detections', 0) / max(data.get('total_frames', 1), 1):.1f} |",
        "",
        "## 2. 行为片段分布",
        "",
        "| 行为 | 片段数 | 平均时长(s) |",
        "| :--- | ---: | ---: |",
    ]
    for beh, cnt in behavior_counts.most_common():
        durs = duration_by_beh[beh]
        avg_d = sum(durs) / len(durs) if durs else 0
        lines.append(f"| {beh} | {cnt} | {avg_d:.1f} |")

    lines += ["", "## 3. 举手片段（核心行为）", ""]
    if raise_hand_segs:
        lines.append("| track_id | 起始 | 结束 | 时长(s) | 置信度 |")
        lines.append("| ---: | :--- | :--- | ---: | ---: |")
        for s in raise_hand_segs[:20]:
            lines.append(
                f"| {s['track_id']} | {s.get('start_timestamp', '-')} | "
                f"{s.get('end_timestamp', '-')} | {s.get('duration_sec', 0):.1f} | "
                f"{s.get('behavior_confidence', 0):.2f} |"
            )
    else:
        lines.append("未检测到举手片段。")

    lines += ["", "## 4. 低置信度片段（需人工复核）", ""]
    if low_conf:
        lines.append(f"共 {len(low_conf)} 个片段置信度 < 0.5，建议查看 keyframes 或对应 clip。")
        lines.append("")
        lines.append("| track_id | 行为 | 置信度 | 时长(s) |")
        lines.append("| ---: | :--- | ---: | ---: |")
        for s in sorted(low_conf, key=lambda x: x.get("behavior_confidence", 0))[:15]:
            lines.append(
                f"| {s['track_id']} | {s.get('behavior', '-')} | "
                f"{s.get('behavior_confidence', 0):.2f} | {s.get('duration_sec', 0):.1f} |"
            )
    else:
        lines.append("无低置信度片段。")

    lines += [
        "",
        "## 5. 建议",
        "",
        "- 漏检：查看 `keyframes/` 中人数最少帧",
        "- 误检行为：对比 `_ann.jpg` 与 `_raw.jpg`",
        "- 阶段 C：运行 `split_clips_by_behavior.py` 自动切分训练 clip",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="tracked JSON 路径")
    parser.add_argument("--output", default=None, help="报告输出路径")
    args = parser.parse_args()

    json_path = Path(args.json)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    out = Path(args.output) if args.output else json_path.parent.parent / "reports" / "track_analysis.md"
    path = build_report(data, out)
    print(f"报告: {path}")


if __name__ == "__main__":
    main()
