# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 00:17:58 2024

@author: ozkal
"""

# @author: ozkal
from ultralytics import YOLO

try:
    # Modelin yüklenmesi (YOLOv8s)
    model = YOLO("yolov8s.pt")

    # Eğitim parametrelerinin ayarlanması ve eğitimin başlatılması
    model.train(data="C:/YoloV8/openimagesV7/scripts/data.yaml",  # Verilerinize göre YAML dosyasının yolunu belirleyin
                epochs=50,                       # Epoch sayısını artırarak daha iyi genel performans elde edebilirsiniz
                batch=16,                        # Uygun bir batch boyutu seçin
                imgsz=640,                       # Görüntü boyutu
                lr0=1e-4,                        # Öğrenme oranını düşük tutarak daha stabil bir eğitim sağlarız
                optimizer='AdamW',               # AdamW optimizer, ağırlık çürümesi ile daha iyi sonuçlar verebilir
                patience=7,                      # Erken durdurma için sabır süresi
                weight_decay=0.0001,             # Ağırlık çürümesini biraz düşürdüm, daha az aşırı öğrenme için
                augment=True,                    # Veri artırımı ile modelin genelleme yeteneğini artırır
                device=0,                        # GPU'da eğitmek için (CPU kullanıyorsanız, 'cpu' olarak ayarlayın)
                workers=8,                       # Veri yükleme hızını artırmak için işçi sayısını artırın
                mosaic=True,                     # Mozaik veri artırımı etkinleştirildi
                hsv_h=0.015,                     # Veri artırımı için HSV renk aralığı ayarlamaları
                hsv_s=0.7,                       
                hsv_v=0.4,                       
                translate=0.1,                   # Çeviri veri artırımı
                scale=0.5,                       # Ölçeklendirme veri artırımı
                shear=0.01,                      # Kesme veri artırımı
                flipud=0.0,                      # Dikey çevirme
                fliplr=0.5)                      # Yatay çevirme veri artırımı

    # Eğitim sonrası değerlendirme
    results = model.val()

    # Eğitim sonrası sonuçları yazdırma
    print("Eğitim başarıyla tamamlandı!")
    print(results)

except Exception as e:
    print(f"Bir hata oluştu: {e}")
