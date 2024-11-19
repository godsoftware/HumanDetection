# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:39:20 2024

@author: ozkal
"""


#%% Libraries
import cv2  # OpenCV kütüphanesi, bilgisayarla görüntü işleme için kullanılır.
from ultralytics import YOLO  # YOLO kütüphanesi, nesne algılama için kullanılır.
import os  # Dosya işlemleri için kullanılır.

#%% Image processing function
def process_frame(frame, model, conf_threshold=0.5):
    """
    Bu fonksiyon, verilen video karesi üzerinde YOLO modeli ile nesne algılama işlemi yapar 
    ve tespit edilen nesneler üzerinde çeşitli işlemler gerçekleştirir.
    
    Parametreler:
    frame: Üzerinde nesne algılama yapılacak video karesi. (Kamera görüntüsünden alınan tek bir kare)
    model: Yüklü YOLO modeli. (Kullandığınız YOLO modelini temsil eder)
    conf_threshold: Güvenlik eşiği; sadece bu eşiğin üzerindeki tespitler dikkate alınır. (Modelin güvenilirliğine göre filtreleme yapar)
    
    Döndürür:
    annotated_frame: Üzerinde nesne tespitleri çizilmiş video karesi.
    num_people: Tespit edilen insan sayısı.
    tespit_koordinatlari: Tespit edilen her insanın koordinatları listesi [(x_center, y_center), ...]
    """
    
    # Detect objects using YOLO model
    results = model(frame)  # YOLO modelini kullanarak karedeki nesneleri algılar

    # Filter detections based on confidence threshold
    filtered_results = results[0]  # Algılama sonuçlarının ilk elemanını alır
    filtered_results.boxes = filtered_results.boxes[filtered_results.boxes.conf >= conf_threshold]  # Eşik değerinin altındaki tespitleri filtreler

    # Initialize person count and coordinate storage
    num_people = 0  # Başlangıçta tespit edilen insan sayısını 0 olarak ayarla
    tespit_koordinatlari = []  # Tespit edilen koordinatların listesi

    for box in filtered_results.boxes:
        if int(box.cls) == 0:  # '0' sınıfının insan olduğunu varsayıyoruz
            num_people += 1
            x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)  # X eksenindeki merkezi hesapla
            y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)  # Y eksenindeki merkezi hesapla
            tespit_koordinatlari.append((x_center, y_center))  # Koordinatları listeye ekle
            cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)  # Merkeze kırmızı bir daire çiz
            x_text = f"x: {x_center}"
            y_text = f"y: {y_center}"
            text_position = (int(box.xyxy[0][2]) - 50, int(box.xyxy[0][1]) - 20)
            cv2.putText(frame, x_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            cv2.putText(frame, y_text, (text_position[0], text_position[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

    annotated_frame = filtered_results.plot()  # Algılanan nesneleri çizin
    return annotated_frame, num_people, tespit_koordinatlari  # İşlenmiş kare, insan sayısı ve koordinatlar döndürülür.

#%% Save coordinates to a file
def save_to_file(coordinates, file_path):
    """
    Bu fonksiyon, tespit edilen koordinatları verilen dosya yoluna kaydeder.
    
    Parametreler:
    coordinates: Tespit edilen koordinatların listesi [(x_center, y_center), ...]
    file_path: Kaydedilecek dosya yolu
    """
    file_path = "C:/YoloV8/realtime/coordinate_files/coordinates.txt"
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))  # Belirtilen klasör yoksa oluştur
    with open(file_path, "a") as file:
        for x, y in coordinates:
            file.write(f"x: {x}, y: {y}, center: ({x},{y})\n")  # Koordinatları dosyaya yaz

#%% Main processing function
def main():
    """
    Bu fonksiyon, kameradan gerçek zamanlı video akışı alır, her kare üzerinde YOLO ile nesne tespiti yapar 
    ve sonuçları ekranda gösterir. Program kapanırken son karedeki tespitleri dosyaya kaydeder.
    """
    
    # Load your YOLO model (best.pt file)
    model = YOLO("bestgithub.pt")

    # Capture video from the webcam
    cap = cv2.VideoCapture(0)

    # Set video dimensions
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    file_path = "coordinates/coordinates.txt"  # Koordinatların kaydedileceği klasör ve dosya yolu
    last_coordinates = []  # Son karedeki koordinatları saklamak için bir liste

    # Read the video stream from the camera
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera verisi alınamadı. Lütfen kameranın düzgün bağlandığından emin olun.")
            break

        # Process the frame and display detections
        annotated_frame, num_people, last_coordinates = process_frame(frame, model, conf_threshold=0.5)
        cv2.imshow('Phone Camera - YOLO Detection', annotated_frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Save the last frame's coordinates to the file when the program closes
    if last_coordinates:
        save_to_file(last_coordinates, file_path)

    cap.release()
    cv2.destroyAllWindows()

#%%
if __name__ == "__main__":
    main()

