"""Modül 6 — Görselleştirme ve çıktı (FG-6, FG-7).

Şimdilik: takip edilen araçların kutusu, kimliği ve kısa izi çizilir.
Hız yazısı ve hız sınırı ihlali vurgusu Oturum 3'te eklenecek.
"""

from __future__ import annotations

from collections import defaultdict, deque

import cv2
import numpy as np

from .detector import VEHICLE_CLASSES


def _color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(track_id) * 7919)
    return tuple(int(c) for c in rng.integers(64, 255, 3))


class TrackDrawer:
    def __init__(self, trail_len: int = 30):
        self.trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))

    def draw(self, frame: np.ndarray, xyxy: np.ndarray, ids: np.ndarray, cls: np.ndarray,
             labels: dict[int, str] | None = None) -> np.ndarray:
        img = frame.copy()
        scale = max(img.shape[1] / 1920, 0.5)
        for box, tid, c in zip(xyxy, ids, cls):
            x1, y1, x2, y2 = map(int, box)
            col = _color(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, max(1, int(2 * scale)))
            self.trails[int(tid)].append(((x1 + x2) // 2, y2))
            pts = np.array(self.trails[int(tid)], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, col, max(1, int(2 * scale)))
            text = (labels or {}).get(int(tid), f"ID {tid} {VEHICLE_CLASSES.get(int(c), '')}")
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, max(1, int(2 * scale)))
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0),
                        max(1, int(2 * scale)), cv2.LINE_AA)
        return img
