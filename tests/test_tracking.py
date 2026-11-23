"""Takip ve tespit veri yapıları için birim testleri (GPU / model gerektirmez)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiztahmini.detector import Detections  # noqa: E402
from hiztahmini.tracker import Tracker  # noqa: E402


def _dets(boxes, conf=0.9):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return Detections(boxes, np.full(len(boxes), conf, np.float32), np.full(len(boxes), 2, np.int32))


def test_bottom_center():
    d = _dets([[10, 20, 30, 60]])
    assert np.allclose(d.bottom_center(), [[20, 60]])


def test_two_moving_cars_keep_their_ids():
    trk = Tracker(fps=30, track_buffer_s=0.5)
    ids_a, ids_b = set(), set()
    for k in range(40):
        a = [100 + 5 * k, 100, 180 + 5 * k, 160]      # sağa giden araç
        b = [800, 600 - 4 * k, 900, 680 - 4 * k]      # yukarı giden araç
        out = trk.update(_dets([a, b]))
        for box, tid in zip(out.xyxy, out.ids):
            (ids_a if box[0] < 500 else ids_b).add(int(tid))
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b


def test_short_occlusion_keeps_id():
    trk = Tracker(fps=30, track_buffer_s=0.5)  # 15 kare tampon
    seen = set()
    for k in range(60):
        box = [100 + 6 * k, 100, 180 + 6 * k, 160]
        dets = _dets([]) if 25 <= k < 33 else _dets([box])  # 8 kare görünmez
        for tid in trk.update(dets).ids:
            seen.add(int(tid))
    assert len(seen) == 1
