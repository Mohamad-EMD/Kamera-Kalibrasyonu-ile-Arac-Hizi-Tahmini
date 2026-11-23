"""UTFPR örnek veri setini ve demo videosunu indirir (herkese açık Google Drive / doğrudan link).

Örnek:
    python scripts/download_data.py --sets 1            # sadece Set01 (~522 MB)
    python scripts/download_data.py --sets 2 3 4 5      # kalan setler (~6,6 GB)
    python scripts/download_data.py --demo              # Roboflow demo videosu (~35 MB)
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import gdown

# Luvizon, Nassu, Minetto (2016) örnek paketi: drive.google.com/drive/folders/1blvOlthdZX62j7SO84Y8lqJlbxOW_p-6
UTFPR_FILES = {
    1: ("1KqcDiD5ic7FichExp6o0jO-zZbRyuVuR", "15WrsIT0AQtPz-EdxW3o4R1U-gySzfoWd"),
    2: ("1N0k0gTF9oJ4Ow4oFXD2G57_oxUu8Ge38", "1_aEfU3qfJkWPstiia0rWY3nZleAzIP8D"),
    3: ("1Rk3MQiHPNNMfJwKc8uwS5ZThRYdgdKua", "1Wz8xEU4g-gf3oIfdZ58m3aBp8B7zsXNs"),
    4: ("1QLZ_CFdJagbJ_BKmCxIP6BbgomPHvxAH", "1gzSjp1lrMPSy7dpkO0PMH2JksgS81Ytl"),
    5: ("1kcI5ErQ-B5YY3Xs_AnmV6SNUF6NXG3KB", "1zdRNwKE4ojkMUg2XXc7RBaoRQKPLhTzc"),
}
DEMO_URL = "https://media.roboflow.com/supervision/video-examples/vehicles.mp4"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", type=int, default=[])
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--out", default="C:/veri")
    args = ap.parse_args()
    out = Path(args.out)

    for s in args.sets:
        vid_id, xml_id = UTFPR_FILES[s]
        d = out / "UTFPR"
        d.mkdir(parents=True, exist_ok=True)
        for fid, name in ((xml_id, f"Set{s:02d}_video01.xml"), (vid_id, f"Set{s:02d}_video01.h264")):
            if (d / name).exists():
                print("zaten var:", name)
                continue
            gdown.download(id=fid, output=str(d / name), quiet=False)

    if args.demo:
        d = out / "demo"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "vehicles.mp4").exists():
            urllib.request.urlretrieve(DEMO_URL, d / "vehicles.mp4")
        print("demo:", d / "vehicles.mp4")


if __name__ == "__main__":
    main()
