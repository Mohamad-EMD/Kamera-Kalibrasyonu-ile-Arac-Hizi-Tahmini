"""Modül 1 — Video okuma (FG-1).

Videoyu açar; FPS, çözünürlük ve kare sayısını verir; kareleri zaman damgasıyla döndürür.
Ham .h264 dosyalarında FPS bilgisi olmayabilir, bu yüzden config'ten FPS verilebilir.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int  # ham akışlarda 0 ya da yaklaşık olabilir
    fps_source: str  # "dosya" veya "config"


class VideoReader:
    """Kareleri (kare_no, t_saniye, görüntü) olarak döndüren basit okuyucu."""

    def __init__(self, path: str | Path, fps: float | None = None,
                 start_frame: int = 0, max_frames: int | None = None):
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Video açılamadı: {self.path}")

        file_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps:
            used_fps, src = float(fps), "config"
        elif file_fps > 0:
            used_fps, src = float(file_fps), "dosya"
        else:
            raise ValueError("Videodan FPS okunamadı; config'te 'video.fps' verin.")

        self.info = VideoInfo(
            path=self.path,
            fps=used_fps,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=max(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0),
            fps_source=src,
        )
        self.start_frame = start_frame
        self.max_frames = max_frames

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        idx = 0
        # Ham h264'te konuma atlamak güvenilir olmadığı için baştaki kareler okunup atlanır.
        while idx < self.start_frame:
            if not self.cap.grab():
                return
            idx += 1
        produced = 0
        while self.max_frames is None or produced < self.max_frames:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, idx / self.info.fps, frame
            idx += 1
            produced += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
