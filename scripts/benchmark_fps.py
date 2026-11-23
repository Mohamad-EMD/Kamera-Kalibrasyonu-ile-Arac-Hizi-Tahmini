"""İşleme hızı (FPS) ölçümü (KG-3).

Her model / giriş boyutu için aynı kareler üzerinde ölçülür:
  - okuma   : sadece video çözme
  - tespit  : okuma + YOLO
  - takip   : okuma + YOLO + ByteTrack (uçtan uca, çıktı videosu yazmadan)
İlk birkaç kare GPU ısınması için ölçüme katılmaz. Sonuç tablo olarak yazdırılır ve CSV'ye kaydedilir.

Örnek:
    python scripts/benchmark_fps.py --config configs/utfpr_set01.yaml --frames 600
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiztahmini.config import load_config  # noqa: E402
from hiztahmini.detector import Detector  # noqa: E402
from hiztahmini.tracker import Tracker  # noqa: E402
from hiztahmini.video_io import VideoReader  # noqa: E402

WARMUP = 20


def run(cfg: dict, frames: int, mode: str, weights: str | None = None, imgsz: int | None = None) -> float:
    v, d = cfg["video"], cfg["detector"]
    reader = VideoReader(v["path"], fps=v.get("fps"), max_frames=frames + WARMUP)
    det = (Detector(weights, imgsz=imgsz, conf=d["conf"], iou=d["iou"], half=d.get("half", True),
                    road_polygon=d.get("road_polygon"), edge_margin=d.get("edge_margin"),
                    agnostic_nms=d.get("agnostic_nms", False)) if weights else None)
    trk = Tracker(fps=reader.info.fps, **cfg["tracker"]) if mode == "takip" else None
    n, t0 = 0, None
    for i, (_, _, frame) in enumerate(reader):
        if i == WARMUP:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0, n = time.perf_counter(), 0
        if det is not None:
            dets = det(frame)
            if trk is not None:
                trk.update(dets, frame)
        n += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    reader.close()
    return n / (time.perf_counter() - t0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 960])
    args = ap.parse_args()
    cfg = load_config(args.config)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Donanım: {gpu} | video: {Path(cfg['video']['path']).name} | {args.frames} kare")
    rows = [dict(model="-", imgsz="-", asama="okuma", fps=run(cfg, args.frames, "okuma"))]
    for m in args.models:
        for s in args.sizes:
            for mode in ("tespit", "takip"):
                fps = run(cfg, args.frames, mode, m, s)
                rows.append(dict(model=m.replace(".pt", ""), imgsz=s, asama=mode, fps=fps))
                print(f"  {m:12s} {s:4d} {mode:7s} {fps:6.1f} FPS")
    df = pd.DataFrame(rows)
    df["fps"] = df["fps"].round(1)
    out = ROOT / cfg["output"]["dir"] / Path(cfg["video"]["path"]).stem
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "fps_olcumu.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
