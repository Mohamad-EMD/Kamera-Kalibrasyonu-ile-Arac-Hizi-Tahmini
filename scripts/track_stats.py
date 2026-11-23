"""izler.csv için takip istatistikleri; varsa UTFPR XML yer gerçeğiyle kaba karşılaştırma.

Örnek:
    python scripts/track_stats.py outputs/Set01_video01/izler.csv --gt C:/veri/UTFPR/Set01_video01.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

import pandas as pd


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("izler")
    ap.add_argument("--gt", default=None)
    ap.add_argument("--min-len", type=int, default=15, help="'uzun iz' sayılması için en az kare")
    args = ap.parse_args()

    df = pd.read_csv(args.izler)
    g = df.groupby("id").agg(n=("kare", "size"), ilk=("kare", "min"), son=("kare", "max"),
                             sinif=("sinif", lambda s: s.mode()[0]))
    print(f"Toplam iz (ID): {len(g)}")
    print(f"  {args.min_len} kareden uzun: {(g.n >= args.min_len).sum()}   5 kareden kısa: {(g.n < 5).sum()}")
    print(f"  İz uzunluğu medyanı: {g.n.median():.0f} kare")
    print(f"  Sınıflar: {g.sinif.value_counts().to_dict()}")

    if args.gt:
        vehicles = ET.parse(args.gt).getroot().findall(".//vehicle")
        radar = [v.find("radar") for v in vehicles if v.find("radar") is not None]
        long_tracks = g[g.n >= args.min_len]
        covered, multi = 0, 0
        for r in radar:
            fs, fe = int(r.get("frame_start")), int(r.get("frame_end"))
            k = ((long_tracks.ilk <= fe) & (long_tracks.son >= fs)).sum()
            covered += k > 0
        print(f"Yer gerçeği: {len(vehicles)} araç, {len(radar)} tanesinin hızı var")
        print(f"  Hız ölçüm aralığında en az bir uzun iz olan araç: {covered}/{len(radar)}")
        print("  (Kaba ölçüt: şerit ve konum eşleştirmesi Oturum 3'te evaluate.py ile yapılacak.)")


if __name__ == "__main__":
    main()
