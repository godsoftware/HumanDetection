# HumanDetection - İnsan Algılama ve Takip Sistemi

## 📋 Proje Hakkında

HumanDetection, YOLOv8 ve OpenCV kullanarak gerçek zamanlı insan algılama, sayma ve takip yapabilen kapsamlı bir bilgisayarlı görü projesidir. Bu proje, güvenlik sistemleri, insan sayımı, trafik analizi ve benzeri uygulamalar için geliştirilmiştir.

## ✨ Özellikler

- **Gerçek Zamanlı İnsan Algılama**: YOLOv8 modeli ile yüksek doğrulukta insan tespiti
- **İnsan Sayımı**: Tespit edilen insanların gerçek zamanlı sayımı
- **Koordinat Takibi**: Tespit edilen insanların X-Y koordinatlarının kaydedilmesi
- **Alan Bazlı Sayım**: Belirli alanlardaki insan sayısının takibi
- **Çoklu Kamera Desteği**: Webcam, IP kamera ve Basler kamera desteği
- **Video İşleme**: Kayıtlı video dosyalarından insan tespiti
- **Model Eğitimi**: Özel veri setleri ile YOLOv8 model eğitimi
- **SORT Takip Algoritması**: İnsan takibi ve sayımı için gelişmiş algoritma

## 🚀 Kurulum

### Gereksinimler

- Python 3.8 veya üzeri
- CUDA destekli GPU (önerilen)
- Windows 10/11

### Adım 1: Repository'yi Klonlayın

```bash
git clone https://github.com/yourusername/HumanDetection.git
cd HumanDetection
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### Adım 3: Gerekli Paketleri Yükleyin

```bash
pip install ultralytics opencv-python numpy cvzone pypylon torch torchvision
```

### Adım 4: YOLOv8 Modelini İndirin

```bash
# YOLOv8n (nano) modeli için
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Veya kendi eğitilmiş modelinizi kullanın
# best.pt dosyasını proje klasörüne kopyalayın
```

## 📁 Proje Yapısı

```
HumanDetection/
├── areaofpeople.py              # Alan bazlı insan sayımı
├── get_basler_cam_realtime.py  # Basler kamera entegrasyonu
├── live_cam_numofpeople.py     # Canlı IP kamera insan sayımı
├── people_counter.py            # Gelişmiş insan sayımı (SORT ile)
├── train4.py                    # YOLOv8 model eğitimi
├── webcam_xy_center_txtfile.py # Webcam koordinat kaydetme
├── best.pt                      # Eğitilmiş YOLO modeli
├── last.pt                      # Son eğitim checkpoint'i
├── data.yaml                    # Eğitim veri konfigürasyonu
└── README.md                    # Bu dosya
```

## 🎯 Kullanım Örnekleri

### 1. Webcam ile İnsan Algılama ve Koordinat Kaydetme

```bash
python webcam_xy_center_txtfile.py
```

**Özellikler:**
- Gerçek zamanlı webcam görüntüsü
- İnsan tespiti ve sayımı
- X-Y koordinatlarının dosyaya kaydedilmesi
- Görsel geri bildirim

### 2. Alan Bazlı İnsan Sayımı

```bash
python areaofpeople.py
```

**Özellikler:**
- Belirli alanlardaki insan sayımı
- Görsel alan işaretleme
- Koordinat bazlı filtreleme
- Gerçek zamanlı istatistikler

### 3. Canlı IP Kamera ile İnsan Sayımı

```bash
python live_cam_numofpeople.py
```

**Özellikler:**
- M3U8 stream desteği
- FFmpeg entegrasyonu
- Gerçek zamanlı işleme
- Yüksek çözünürlük desteği

### 4. Gelişmiş İnsan Sayımı (SORT Algoritması)

```bash
python people_counter.py
```

**Özellikler:**
- SORT takip algoritması
- Çizgi geçiş sayımı
- Yön bazlı sayım
- Görsel geri bildirim

### 5. Basler Kamera Entegrasyonu

```bash
python get_basler_cam_realtime.py
```

**Özellikler:**
- Basler kamera desteği
- USB3 bağlantı
- Gerçek zamanlı görüntü işleme
- Yüksek performans

### 6. Model Eğitimi

```bash
python train4.py
```

**Özellikler:**
- YOLOv8s model eğitimi
- Özelleştirilmiş parametreler
- Veri artırımı
- GPU desteği

## ⚙️ Konfigürasyon

### Model Parametreleri

```python
# YOLOv8 eğitim parametreleri
model.train(
    data="data.yaml",           # Veri konfigürasyon dosyası
    epochs=50,                  # Epoch sayısı
    batch=16,                   # Batch boyutu
    imgsz=640,                  # Görüntü boyutu
    lr0=1e-4,                   # Öğrenme oranı
    optimizer='AdamW',          # Optimizer
    patience=7,                 # Erken durdurma
    device=0,                   # GPU cihazı
    augment=True                # Veri artırımı
)
```

### Güven Eşiği Ayarları

```python
# Farklı güven eşiği değerleri
conf_threshold = 0.2    # Düşük eşik - daha fazla tespit
conf_threshold = 0.5    # Orta eşik - dengeli tespit
conf_threshold = 0.8    # Yüksek eşik - yüksek güvenilirlik
```

## 🔧 Teknik Detaylar

### Ana Paketler

```
ultralytics>=8.0.0      # YOLOv8 modeli
opencv-python>=4.8.0   # Görüntü işleme
numpy>=1.21.0          # Sayısal işlemler
cvzone>=1.5.6          # Görsel arayüz
pypylon>=2.0.0         # Basler kamera
torch>=1.9.0           # PyTorch
torchvision>=0.10.0    # TorchVision
```

### Sistem Gereksinimleri

- **Minimum**: CPU i5, 8GB RAM
- **Önerilen**: GPU RTX 3060+, 16GB RAM
- **Optimal**: GPU RTX 4080+, 32GB RAM

### Model Performansı

- **YOLOv8n**: Hızlı, düşük doğruluk
- **YOLOv8s**: Dengeli hız ve doğruluk
- **YOLOv8m**: Yüksek doğruluk, orta hız
- **YOLOv8l**: Çok yüksek doğruluk, düşük hız

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **CUDA Hatası**
   ```bash
   # CPU kullanımı için
   device='cpu'
   ```

2. **Model Yükleme Hatası**
   ```bash
   # Model dosyasının varlığını kontrol edin
   ls -la *.pt
   ```

3. **Kamera Bağlantı Hatası**
   ```bash
   # Kamera indeksini kontrol edin
   cap = cv2.VideoCapture(0)  # 0, 1, 2... deneyin
   ```

4. **FFmpeg Hatası**
   ```bash
   # FFmpeg'in yüklü olduğundan emin olun
   ffmpeg -version
   ```

## 📈 Geliştirme

### Yeni Özellik Ekleme

1. Yeni Python dosyası oluşturun
2. Gerekli import'ları ekleyin
3. Ana fonksiyonu yazın
4. README'yi güncelleyin

### Model Geliştirme

1. `data.yaml` dosyasını düzenleyin
2. `train4.py` parametrelerini ayarlayın
3. Eğitimi başlatın
4. Sonuçları değerlendirin
