"""行为片段聚合与切分。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class TrackAccumulator:
    track_id: int
    frames: list[int] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    det_confs: list[float] = field(default_factory=list)


def smooth_behaviors(behaviors: list[str], k: int = 3) -> list[str]:
    if not behaviors:
        return []
    out = []
    for i in range(len(behaviors)):
        window = behaviors[max(0, i - k + 1) : i + 1]
        out.append(Counter(window).most_common(1)[0][0])
    return out


def frame_timestamp(frame: int, fps: float) -> str:
    if not fps:
        return "00:00.000"
    sec = frame / fps
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:06.3f}"


def build_segments(
    acc: TrackAccumulator,
    fps: float,
    min_frames: int = 15,
    smooth_k: int = 3,
) -> list[dict]:
    # ============================================================
    # 行为片段聚合算法（请补充代码）
    #
    # 目标：将平滑后的行为序列合并为行为片段（segment）
    #       每人连续相同行为的起止时间和持续时间
    #
    # 已有数据：
    #   - smoothed = smooth_behaviors(acc.behaviors, smooth_k)  # 已平滑行为
    #   - acc.frames / acc.confidences / acc.det_confs
    #   - frame_timestamp(frame, fps) → "00:00.000" 格式时间戳
    #
    # 步骤：
    #   1. 定义内嵌函数 flush(end_i, behavior, confs)：
    #      将 [seg_start_i, end_i] 区间写入 segments
    #      每个 segment 的字典结构见下方提示
    #   2. 遍历 smoothed[1:]，检测行为变化：
    #      - 行为变了 → flush 当前段 → 开始新段
    #      - 行为不变 → 累积 confs
    #   3. flush 最后一段
    #   4. 兜底：如果 segments 为空，用全体数据构造一个片段
    #
    # 每个 segment 结构：
    #   {
    #     "behavior": "sit_listen",
    #     "behavior_confidence": 0.55,
    #     "detection_confidence_mean": 0.87,
    #     "start_frame": 0, "end_frame": 240,
    #     "start_timestamp": "00:00.000",
    #     "end_timestamp": "00:08.000",
    #     "duration_sec": 8.0
    #   }
    #
    # 提示：
    #   - min_frames 过滤过短片段（但第一个片段不跳过）
    #   - flush 中的 seg_start_i 引用外层变量
    #   - 兜底用 Counter(acc.behaviors).most_common(1)[0][0] 取众数
    # ============================================================
    if not acc.frames:
        return []
    pass  # TODO: 请在此补充行为片段聚合代码 
