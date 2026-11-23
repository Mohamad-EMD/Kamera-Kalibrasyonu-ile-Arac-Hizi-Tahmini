# Kamera Kalibrasyonu ile Video Görüntülerinden Araç Hızı Tahmini

**Fırat Üniversitesi – Mühendislik Fakültesi – Bilgisayar Mühendisliği**
Bilgisayar Mühendisliği Tasarım Dersi – Takım Çalışması Projesi

**Ekip:** PikselRadar

## Proje Hakkında

Bu proje, sabit açılı bir kameradan elde edilen trafik video kayıtları üzerinden araçların anlık hızlarını, radar veya LiDAR gibi ek bir fiziksel sensöre ihtiyaç duymadan, bilgisayarlı görü ve kamera kalibrasyonu yöntemleriyle tahmin eden bir yazılım sistemi geliştirmeyi amaçlamaktadır.

İki boyutlu görüntülerdeki perspektif bozulması, karayolu üzerindeki bilinen referans geometriler (şerit çizgileri, şerit genişlikleri) kullanılarak hesaplanan homografi matrisi ile giderilir ve piksel koordinatları gerçek dünya metrik koordinatlarına dönüştürülür.

## Hedefler

- YOLO ile araçların (otomobil, otobüs, kamyon) en az %90 doğrulukla tespiti
- ByteTrack ile araç kimliklerinin (ID) kareler boyunca kararlı şekilde takibi
- Homografi ile kuş bakışı (Bird's Eye View) dönüşümü ve piksel → metre haritalama
- Hız tahmininde ortalama mutlak hatanın (MAE) ±5 km/sa altında tutulması
- Standart donanımda en az 20–25 FPS çalışma performansı
- Araç ID, hız ve sınır kutusu bilgilerinin video üzerine yazdırılması, hız ihlallerinin vurgulanması

## Yöntem (Pipeline)

1. **Araç Tespiti:** YOLO derin öğrenme modeli
2. **Çoklu Nesne Takibi:** ByteTrack
3. **Kamera Kalibrasyonu ve Homografi:** OpenCV ile 3×3 homografi matrisi (H), araçların alt-orta temas noktalarının gerçek dünya koordinatlarına (X, Y metre) dönüştürülmesi
4. **Hız Kestirimi ve Filtreleme:**
   - `Δt = Δframe / FPS`
   - `v = (Δd / Δt) × 3.6` (km/sa)
   - Gürültü giderme için Kalman filtresi / hareketli ortalama

**Değerlendirme metrikleri:** MAE, RMSE (hız), yakalama oranı ve kimlik değişimi (takip), FPS

## Veri

- **UTFPR araç hızı veri seti** (Luvizon, Nassu, Minetto, IEEE T-ITS 2016): sabit kamera, 1920×1080, 30,15 FPS,
  her araç için endüktif döngüyle ölçülmüş gerçek hız. Örnek paket (Set01–Set05, ~7 GB) herkese açık Google Drive'da.
- **Demo videosu** (gerçek hız yok, sadece görsel): Roboflow `vehicles.mp4`.
- Veriler depoya konmaz, varsayılan konum `C:\veri\` (config dosyalarından değiştirilebilir).

## Kurulum (Windows, Python 3.11, NVIDIA GPU)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/download_data.py --sets 1 --demo
```

## Kullanım

```powershell
# Tespit + takip: outputs/<video>/izler.csv ve takip.mp4
python scripts/track_video.py --config configs/utfpr_set01.yaml --save-video

# Takip istatistikleri (yer gerçeğiyle kaba karşılaştırma)
python scripts/track_stats.py outputs/Set01_video01/izler.csv --gt C:/veri/UTFPR/Set01_video01.xml

# İşleme hızı ölçümü (YOLO11n/s, 640/960)
python scripts/benchmark_fps.py --config configs/utfpr_set01.yaml --frames 600

# Testler
python -m pytest -q tests
```

## Proje Durumu

| Modül | Durum |
|---|---|
| 1. Video okuma (`video_io.py`) | ✅ |
| 2. Araç tespiti, YOLO11 (`detector.py`) | ✅ |
| 3. Araç takibi, ByteTrack (`tracker.py`) | ✅ |
| 4. Kalibrasyon / homografi (`calibration.py`) | ⏳ Hafta 9 |
| 5. Hız kestirimi + Kalman (`speed.py`) | ⏳ Hafta 10 |
| 6. Görselleştirme + CSV (`visualize.py`) | 🟡 takip görüntüsü hazır, hız Hafta 10 |
| 7. Değerlendirme MAE/RMSE (`evaluate.py`) | ⏳ Hafta 10 |

## Takım Üyeleri ve Görevler

| Adı Soyadı | Görevi |
|---|---|
| Emre DURU (Takım Kaptanı) | Kamera kalibrasyonu, homografi matrisi hesaplama ve piksel-metre koordinat dönüşümü |
| Muhammed El MUHAMMED | Video veri setlerinin hazırlanması, YOLO ile araç tespiti ve ByteTrack ile nesne takibi |
| Luay HAMU | FPS ve yer değiştirmeye bağlı hız hesaplama, filtreleme algoritmaları ve video üzeri görselleştirme |

## Proje Dosyaları

- **GitHub:** https://github.com/Mohamad-EMD/Kamera-Kalibrasyonu-ile-Arac-Hizi-Tahmini
- **Google Drive** (videolar, veri setleri ve büyük dosyalar): https://drive.google.com/drive/folders/1B-E84FmoX5287Plz8nCQy2arP01OrQF8

## Haftalık Raporlar

| Hafta | Dosya |
|---|---|
| 1 – Proje Tanımı | [Hafta1_Proje raporu.doc](raporlar/Hafta1_Proje%20raporu.doc) |
| 2 – Gereksinim Analizi | [Hafta2_Proje raporu.doc](raporlar/Hafta2_Proje%20raporu.doc) |
| 3 – Veri Seti Analizi ve Seçimi | [Hafta3_Proje_raporu.docx](raporlar/Hafta3_Proje_raporu.docx) |
| 4 – Model Kurulumu ve Kalibrasyon | [Hafta4_Proje raporu.doc](raporlar/Hafta4_Proje%20raporu.doc) |
| 7 – Takip ve Hız Kestirimi Modülünün Tasarımı | [Hafta7_Proje raporu.doc](raporlar/Hafta7_Proje%20raporu.doc) |

## Kaynaklar

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You only look once: Unified, real-time object detection.* CVPR, 779–788.
- Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.
