"""Modül 3 — Çoklu araç takibi (FG-4).

Ultralytics içindeki ByteTrack uygulamasını, tespit modülünden gelen NumPy dizileriyle besler.
Böylece takip, tespit modelinden bağımsızdır (KG-4): YOLO yerine başka bir dedektör de kullanılabilir.

ByteTrack önce yüksek skorlu tespitleri mevcut izlerle eşleştirir, sonra düşük skorlu tespitleri
kalan izlerle eşleştirir; bu sayede kısmen örtülen araçlar kaybolmaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .detector import Detections


class _BoxesAdapter:
    """BYTETracker'ın beklediği arayüz: conf, cls, xywh, xyxy ve boolean indeksleme."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_BoxesAdapter":
        return _BoxesAdapter(self.xyxy[idx], self.conf[idx], self.cls[idx])


@dataclass
class Tracks:
    xyxy: np.ndarray   # (M, 4)
    ids: np.ndarray    # (M,) kalıcı araç kimliği
    conf: np.ndarray   # (M,)
    cls: np.ndarray    # (M,)

    def __len__(self) -> int:
        return len(self.ids)


class Tracker:
    def __init__(self, fps: float, track_high_thresh: float = 0.25, track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.25, track_buffer_s: float = 0.5,
                 match_thresh: float = 0.8, fuse_score: bool = True):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.track_buffer = max(1, round(track_buffer_s * fps))  # saniye → kare
        args = SimpleNamespace(
            tracker_type="bytetrack",
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=match_thresh,
            fuse_score=fuse_score,
        )
        # Bu Ultralytics sürümünde track_buffer doğrudan kare sayısı olarak kullanılır.
        self._bt = BYTETracker(args)

    def update(self, det: Detections, frame: np.ndarray | None = None) -> Tracks:
        out = self._bt.update(_BoxesAdapter(det.xyxy, det.conf, det.cls), frame)
        if out is None or len(out) == 0:
            return Tracks(np.zeros((0, 4), np.float32), np.zeros(0, np.int64),
                          np.zeros(0, np.float32), np.zeros(0, np.int32))
        out = np.asarray(out)
        return Tracks(xyxy=out[:, :4].astype(np.float32), ids=out[:, 4].astype(np.int64),
                      conf=out[:, 5].astype(np.float32), cls=out[:, 6].astype(np.int32))
