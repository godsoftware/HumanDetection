import cv2
import subprocess
import numpy as np
from ultralytics import YOLO

def process_frame(frame, model):
    # YOLO modelini kullanarak nesne algılama
    results = model(frame)
    
    # Tespit edilen insanların sayısını belirleme
    num_people = 0
    for r in results:
        for cls in r.boxes.cls:
            if int(cls) == 0:  # YOLOv8'de "0" genellikle 'person' sınıfını temsil eder
                num_people += 1

    # Algılanan nesneleri çizin
    annotated_frame = results[0].plot()

    # Ekrana insan sayısını yazdır
    cv2.putText(annotated_frame, f"People Count: {num_people}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return annotated_frame, num_people

def main():
    # Kendi YOLO modelinizi yükleyin (best.pt)
    model = YOLO("best.pt")

    # FFmpeg komutunu çalıştır
    ffmpeg_command = [
        'C:/YoloV8/realtime/ffmpeg/bin/ffmpeg.exe',
        '-i', 'https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/1725421336/ei/uILXZtOSJ9WCi9oPouyhiAE/ip/109.228.222.146/id/VR-x3HdhKLQ.28/itag/96/source/yt_live_broadcast/requiressl/yes/ratebypass/yes/live/1/sgoap/gir%3Dyes%3Bitag%3D140/sgovp/gir%3Dyes%3Bitag%3D137/rqh/1/hdlc/1/hls_chunk_host/rr4---sn-u0g3uxax3-5q5e.googlevideo.com/xpc/EgVo2aDSNQ%3D%3D/spc/Mv1m9hl3CwxkJ0elmlECFFQHpQHKhNLdMdASaPEKTG5UcER_NIn7vX4/vprv/1/playlist_type/DVR/initcwndbps/1081250/mh/v8/mm/44/mn/sn-u0g3uxax3-5q5e/ms/lva/mv/m/mvi/4/pl/24/dover/11/pacing/0/keepalive/yes/mt/1725399159/sparams/expire,ei,ip,id,itag,source,requiressl,ratebypass,live,sgoap,sgovp,rqh,hdlc,xpc,spc,vprv,playlist_type/sig/AJfQdSswRQIhAJCSRNKErXlxyRrf_Wdcce2DD3GU-bbMfcheZDN_YhIkAiAOnGigp1gBftMuYjrFn20mHMMOC6XF5WDPmA3qgNkz6Q%3D%3D/lsparams/hls_chunk_host,initcwndbps,mh,mm,mn,ms,mv,mvi,pl/lsig/ABPmVW0wRQIgCAQt_qdBv5sHHpm35y4Q13S1vgGD6CzqdCd_dp-vHBQCIQDOuVSscWjqtrf3KLSqSwdQ4Yhgg9HHwMLRe-WTj7OyhQ%3D%3D/playlist/index.m3u8',  # Buraya m3u8 URL'sini girin
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-'
    ]

    pipe = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, bufsize=10**8)

    # Video boyutlarını belirleyin (örneğin 640x360)
    width, height = 1920,1080

    # OpenCV ile FFmpeg'den gelen video akışını okuma
    while True:
        raw_image = pipe.stdout.read(width * height * 3)  # Her bir kare için veri al
        if len(raw_image) != (width * height * 3):
            break

        # FFmpeg'den gelen veriyi görüntü olarak dönüştür
        frame = np.frombuffer(raw_image, dtype='uint8').reshape((height, width, 3))

        # Kareyi işleyin ve üzerinde tespitleri gösterin
        annotated_frame, num_people = process_frame(frame, model)

        # Sonuçları göster
        cv2.imshow('MOBESE - YOLO Algılama', annotated_frame)

        # Çıkmak için 'q' tuşuna basın
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipe.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
