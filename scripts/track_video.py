"""Videoda araçları tespit edip takip eder; izler.csv ve isteğe bağlı takip videosu üretir.

Örnek:
    python scripts/track_video.py --config configs/utfpr_set01.yaml --max-frames 900 --save-video
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiztahmini.config import load_config  # noqa: E402
from hiztahmini.detector import VEHICLE_CLASSES, Detector  # noqa: E402
from hiztahmini.tracker import Tracker  # noqa: E402
from hiztahmini.video_io import VideoReader  # noqa: E402
from hiztahmini.visualize import TrackDrawer  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--weights", default=None, help="config'teki modeli geçersiz kılar")
    ap.add_argument("--imgsz", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    vcfg, dcfg, tcfg = cfg["video"], cfg["detector"], cfg["tracker"]
    if args.weights:
        dcfg["weights"] = args.weights
    if args.imgsz:
        dcfg["imgsz"] = args.imgsz

    reader = VideoReader(vcfg["path"], fps=vcfg.get("fps"), start_frame=args.start_frame,
                         max_frames=args.max_frames)
    info = reader.info
    print(f"Video: {info.width}x{info.height}, FPS={info.fps} ({info.fps_source}), kare={info.frame_count}")

    detector = Detector(dcfg["weights"], imgsz=dcfg["imgsz"], conf=dcfg["conf"], iou=dcfg["iou"],
                        half=dcfg.get("half", True), road_polygon=dcfg.get("road_polygon"),
                        edge_margin=dcfg.get("edge_margin"), agnostic_nms=dcfg.get("agnostic_nms", False))
    tracker = Tracker(fps=info.fps, **tcfg)

    out_dir = ROOT / cfg["output"]["dir"] / Path(vcfg["path"]).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    drawer = TrackDrawer()
    if args.save_video:
        writer = cv2.VideoWriter(str(out_dir / "takip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 info.fps, (info.width, info.height))

    rows = []
    t0 = time.perf_counter()
    n = 0
    total = args.max_frames or (info.frame_count - args.start_frame if info.frame_count else None)
    for idx, t, frame in tqdm(reader, total=total, unit="kare"):
        det = detector(frame)
        trk = tracker.update(det, frame)
        for box, tid, conf, c in zip(trk.xyxy, trk.ids, trk.conf, trk.cls):
            rows.append(dict(kare=idx, t=round(t, 4), id=int(tid), sinif=VEHICLE_CLASSES.get(int(c), str(c)),
                             guven=round(float(conf), 3), x1=round(float(box[0]), 1), y1=round(float(box[1]), 1),
                             x2=round(float(box[2]), 1), y2=round(float(box[3]), 1)))
        if writer is not None:
            writer.write(drawer.draw(frame, trk.xyxy, trk.ids, trk.cls))
        n += 1
    elapsed = time.perf_counter() - t0
    reader.close()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "izler.csv", index=False)
    n_ids = df["id"].nunique() if len(df) else 0
    print(f"{n} kare, {elapsed:.1f} sn, ortalama {n / elapsed:.1f} FPS (video yazma dahil: {args.save_video})")
    print(f"Farklı araç kimliği: {n_ids}  ->  {out_dir}")


if __name__ == "__main__":
    main()
