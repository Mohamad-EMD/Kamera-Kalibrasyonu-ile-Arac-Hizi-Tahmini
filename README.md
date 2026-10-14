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

**Değerlendirme metrikleri:** MAE, RMSE (hız), mAP (tespit), MOTA (takip)

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
| 1 | [Hafta1_Proje raporu.doc](Hafta1_Proje%20raporu.doc) |
| 2 – Gereksinim Analizi | [Hafta2_Proje raporu.doc](Hafta2_Proje%20raporu.doc) |

## Kaynaklar

- Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You only look once: Unified, real-time object detection.* CVPR, 779–788.
- Hartley, R., & Zisserman, A. (2003). *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- Bradski, G. (2000). *The OpenCV Library.* Dr. Dobb's Journal of Software Tools.
