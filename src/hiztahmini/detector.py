"""Modül 2 — Araç tespiti (FG-3).

COCO üzerinde önceden eğitilmiş YOLO11 ile karedeki araçları bulur. Sadece taşıt sınıfları
(car, motorcycle, bus, truck) tutulur. İsteğe bağlı yol maskesi (çokgen) dışındaki tespitler atılır.
Çıktı yalnızca NumPy dizileridir; takip modülü YOLO'ya bağlı değildir (KG-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# COCO sınıf numaraları
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detections:
    xyxy: np.ndarray   # (N, 4) piksel: x1, y1, x2, y2
    conf: np.ndarray   # (N,)
    cls: np.ndarray    # (N,) COCO sınıf numarası

    def __len__(self) -> int:
        return len(self.conf)

    @staticmethod
    def empty() -> "Detections":
        return Detections(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int32))

    def bottom_center(self) -> np.ndarray:
        """Kutunun alt-orta noktası (aracın yola en yakın noktası), (N, 2)."""
        return np.stack([(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]], axis=1)


class Detector:
    def __init__(self, weights: str = "yolo11s.pt", imgsz: int = 960, conf: float = 0.1,
                 iou: float = 0.7, device: str | int | None = None, half: bool = True,
                 road_polygon: list[list[float]] | None = None, edge_margin: int | None = None,
                 agnostic_nms: bool = False):
        from ultralytics import YOLO  # içe aktarma yavaş olduğu için burada

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf  # ByteTrack düşük skorlu kutuları da kullandığı için eşik düşük tutulur
        self.iou = iou
        self.device = device
        self.half = half
        self.edge_margin = edge_margin
        # Sınıftan bağımsız NMS: aynı araç hem "car" hem "truck" olarak bulunursa tek kutu kalır
        self.agnostic_nms = agnostic_nms
        self.road_polygon = (np.asarray(road_polygon, np.float32).reshape(-1, 1, 2)
                             if road_polygon else None)

    def __call__(self, frame: np.ndarray) -> Detections:
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               classes=list(VEHICLE_CLASSES), device=self.device,
                               agnostic_nms=self.agnostic_nms, quantize=16 if self.half else 32,
                               verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return Detections.empty()
        det = Detections(
            xyxy=r.boxes.xyxy.cpu().numpy().astype(np.float32),
            conf=r.boxes.conf.cpu().numpy().astype(np.float32),
            cls=r.boxes.cls.cpu().numpy().astype(np.int32),
        )
        det = self._drop_edge_boxes(det, frame.shape[0])
        return self._apply_mask(det)

    def _drop_edge_boxes(self, det: Detections, height: int) -> Detections:
        """Görüntünün alt kenarına değen (kadraja yeni giren, yarım görünen) kutuları atar.

        Kadraja giren araçların kutusu kesik olduğu için şekli kareden kareye çok değişir;
        bu da takipte kimlik değişimine (ID switch) yol açar. Kutunun alt-orta noktası da bu durumda
        aracın gerçek konumunu göstermez. Üst kenar filtrelenmez: uzun araçların (otobüs, minibüs)
        üstü kadraj dışına taşarken alt kısmı hâlâ ölçüm bölgesindedir.
        """
        if self.edge_margin is None or len(det) == 0:
            return det
        keep = det.xyxy[:, 3] < height - self.edge_margin
        return Detections(det.xyxy[keep], det.conf[keep], det.cls[keep])

    def _apply_mask(self, det: Detections) -> Detections:
        if self.road_polygon is None or len(det) == 0:
            return det
        pts = det.bottom_center()
        keep = np.array([cv2.pointPolygonTest(self.road_polygon, (float(x), float(y)), False) >= 0
                         for x, y in pts], dtype=bool)
        return Detections(det.xyxy[keep], det.conf[keep], det.cls[keep])
