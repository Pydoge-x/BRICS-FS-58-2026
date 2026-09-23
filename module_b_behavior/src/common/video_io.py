"""视频读写工具。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def open_video_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """创建视频写入器，优先 H.264，不行则降级到 mp4v。"""
    # 直接使用 mp4v（软件编码），避免虚拟机 v4l2m2m 硬件编码器报错
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer
    # 降级：avc1
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if writer.isOpened():
        return writer
    raise RuntimeError(f"无法创建视频写入器: {path}")


def reencode_to_h264(input_path: Path) -> None:
    """将 mp4v 编码的视频重编码为浏览器兼容的 H.264。

    先用 OpenCV 的 mp4v 写入，再用 imageio-ffmpeg 重编码。
    重编码后的视频会覆盖原文件。
    """
    if not input_path.exists():
        return

    tmp_path = input_path.with_suffix(".tmp.mp4")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        tmp_path.replace(input_path)
    except subprocess.CalledProcessError:
        # 重编码失败，保留原始 mp4v 文件
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
