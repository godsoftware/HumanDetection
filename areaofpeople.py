# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 01:14:04 2024

@author: ozkal
"""

#%% Libraries
import cv2  # OpenCV kütüphanesi, bilgisayarla görüntü işleme için kullanılır.
from ultralytics import YOLO  # YOLO kütüphanesi, nesne algılama için kullanılır.
import os  # Dosya işlemleri için kullanılır.

#%% Image processing function
def process_frame(frame, model, conf_threshold=0.2, area_coords=None):
    """
    Bu fonksiyon, verilen video karesi üzerinde YOLO modeli ile nesne algılama işlemi yapar 
    ve tespit edilen nesneler üzerinde çeşitli işlemler gerçekleştirir.
    
    Parametreler:
    frame: Üzerinde nesne algılama yapılacak video karesi. (Kamera görüntüsünden alınan tek bir kare)
    model: Yüklü YOLO modeli. (Kullandığınız YOLO modelini temsil eder)
    conf_threshold: Güvenlik eşiği; sadece bu eşiğin üzerindeki tespitler dikkate alınır. (Modelin güvenilirliğine göre filtreleme yapar)
    area_coords: Belirli bir alanın koordinatları (sol üst ve sağ alt köşe olarak).
    
    Döndürür:
    annotated_frame: Üzerinde nesne tespitleri çizilmiş video karesi.
    num_people: Tespit edilen insan sayısı.
    area_people_count: Belirli alandaki insan sayısı.
    tespit_koordinatlari: Tespit edilen her insanın koordinatları listesi [(x_center, y_center), ...]
    """
    
    # YOLO modelini kullanarak nesne algılama
    results = model(frame)  
    
    # Eşik değerinin altındaki tespitleri filtreleme
    filtered_results = results[0]  
    filtered_results.boxes = filtered_results.boxes[filtered_results.boxes.conf >= conf_threshold]  

    # İnsan sayısını ve belirli alandaki insan sayısını başlat
    num_people = 0
    area_people_count = 0
    tespit_koordinatlari = []

    # Algılanan nesnelerin her biri için döngüye gir
    for box in filtered_results.boxes:
        if int(box.cls) == 0:  # İnsan sınıfı için
            num_people += 1  # Her tespit edilen insan için sayaç artırılır
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)  # Nesnenin merkez X koordinatını hesapla
            y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)  # Nesnenin merkez Y koordinatını hesapla
            tespit_koordinatlari.append((x_center, y_center))  # Hesaplanan koordinatları listeye ekle
            
            # Tespit edilen insanın merkezine kırmızı bir daire çiz
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
            
            # X ve Y koordinatlarını video karesi üzerine yaz
            x_text = f"x: {x_center}"
            y_text = f"y: {y_center}"
            text_position = (int(box.xyxy[0][2]) - 50, int(box.xyxy[0][1]) - 20)  # Yazının pozisyonunu belirle
            cv2.putText(frame, x_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            cv2.putText(frame, y_text, (text_position[0], text_position[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            
            # Eğer insan belirli bir alanın içindeyse, o alan içindeki insan sayısını artır
            if area_coords:
                if area_coords[0][0] <= x_center <= area_coords[1][0] and area_coords[0][1] <= y_center <= area_coords[1][1]:
                    area_people_count += 1

    # Belirlenen alanı pembe renkte bir dikdörtgen ile işaretle
    if area_coords:
        cv2.rectangle(frame, area_coords[0], area_coords[1], (255, 0, 255), 2)
        cv2.putText(frame, f"Area People Count: {area_people_count}", (area_coords[0][0], area_coords[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Tespit edilen insan sayısını ekrana yaz
    cv2.putText(frame, f"People Count: {num_people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Sonuçları çizimlerle ekrana göster
    annotated_frame = filtered_results.plot()  
    return annotated_frame, num_people, tespit_koordinatlari, area_people_count  

#%% Save coordinates to a file
def save_to_file(coordinates, file_path):
    """
    Bu fonksiyon, tespit edilen koordinatları verilen dosya yoluna kaydeder.
    
    Parametreler:
    coordinates: Tespit edilen koordinatların listesi [(x_center, y_center), ...]
    file_path: Kaydedilecek dosya yolu
    """
    file_path = "C:/YoloV8/realtime/coordinate_files/coordinates.txt"
    
    # Eğer klasör yoksa, klasörü oluştur
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))  # Belirtilen klasör yoksa oluştur
        
    # Koordinatları dosyaya yaz
    with open(file_path, "a") as file:
        for x, y in coordinates:
            file.write(f"x: {x}, y: {y}, center: ({x},{y})\n")  # Koordinatları dosyaya kaydet

#%% Main processing function
def main():
    """
    Bu fonksiyon, bilgisayardaki bir video dosyasını alır, her kare üzerinde YOLO ile nesne tespiti yapar 
    ve sonuçları ekranda gösterir. Program kapanırken son karedeki tespitleri dosyaya kaydeder.
    """
    
    # YOLO modelini yükleyin
    model = YOLO("best.pt")

    # Video dosyasının yolunu belirtin
    video_path = 'C:/YoloV8/realtime/Videos/peoplewalking.mp4'  # Buraya video dosyanızın yolunu girin

    # Video dosyasını aç
    cap = cv2.VideoCapture(video_path)

    # Eğer video dosyası açılamazsa hata mesajı ver ve çık
    if not cap.isOpened():
        print(f"Video dosyası açılamadı: {video_path}")
        return

    file_path = "coordinates/coordinates.txt"  # Koordinatların kaydedileceği klasör ve dosya yolu
    last_coordinates = []  # Son karedeki koordinatları saklamak için bir liste

    # Belirli alanın koordinatlarını tanımlayın (sol üst ve sağ alt köşe)
    area_coords = ((100, 300), (1000, 700))  # Örnek koordinatlar

    # Video akışını oku ve her kareyi işle
    while True:
        ret, frame = cap.read()  # Videodan bir kare al
        if not ret:
            print("Video sonuna ulaşıldı veya video akışı alınamadı.")
            break

        # Kareyi işleyin ve üzerinde tespitleri gösterin
        annotated_frame, num_people, last_coordinates, area_people_count = process_frame(frame, model, conf_threshold=0.5, area_coords=area_coords)
        cv2.imshow('Video File - YOLO Detection', annotated_frame)  # İşlenmiş kareyi göster

        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Son karedeki koordinatları dosyaya kaydet
    if last_coordinates:
        save_to_file(last_coordinates, file_path)

    # Video ve pencereleri serbest bırak
    cap.release()
    cv2.destroyAllWindows()

#%% Eğer bu dosya doğrudan çalıştırılıyorsa, main() fonksiyonunu çağır
if __name__ == "__main__":
    main()
