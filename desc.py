
import cv2 as cv  # OpenCV, görüntü işleme için kullanılıyor
import tkinter as tk  # Grafiksel kullanıcı arayüzü (GUI) oluşturmak için
from tkinter import filedialog  # Kullanıcının dosya seçmesine olanak tanır
from PIL import Image, ImageTk  # Tkinter ile görüntüleri GUI'ye entegre etmek için

# Yüz tespiti yapan fonksiyon. Bir sinir ağı kullanarak tespit yapar.
def getFaceBox(net, frame, conf_threshold=0.7):
    
    # Orijinal çerçevenin bir kopyasını oluşturuyoruz
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]  # Çerçevenin yüksekliği
    frameWidth = frameOpencvDnn.shape[1]  # Çerçevenin genişliği

    # Sinir ağına giriş için bir "blob" oluşturuyoruz
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Blob ağına giriş olarak veriliyor
    net.setInput(blob)
    detections = net.forward()  # Yüz tespiti sonuçlarını alıyoruz

    bboxes = []  # Tespit edilen yüzlerin koordinatlarını saklayacak liste
    confidences = []  # Her yüzün güven skorlarını saklar

    # Algılanan yüzler arasında gezinerek güven skoru eşik değerini geçenleri seçiyoruz

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Tespit edilen nesnenin güven skoru
        if confidence > conf_threshold:  # Güven skoru yeterince yüksekse
            # Koordinatları belirliyoruz
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])  # Koordinatları listeye ekliyoruz
            confidences.append(float(confidence))

    # Eğer herhangi bir yüz bulunmuşsa, NMS (Non-Maximum Suppression) uyguluyoruz
    if len(bboxes) > 0:
        indices = cv.dnn.NMSBoxes(bboxes, confidences, conf_threshold, 0.3)
        filtered_bboxes = []
        for idx in indices:
            if isinstance(idx, tuple):
                idx = idx[0]
            filtered_bboxes.append(bboxes[idx])
        return frameOpencvDnn, filtered_bboxes

    return frameOpencvDnn, []  # Eğer yüz bulunmazsa boş liste döner

# Model dosyalarının yolları tanımlanıyor
faceProto = "modelNweight/opencv_face_detector.pbtxt"  # Yüz tespiti için model yapısı
faceModel = "modelNweight/opencv_face_detector_uint8.pb"  # Yüz tespiti ağırlıkları
ageProto = "modelNweight/age_deploy.prototxt"  # Yaş tahmini için model yapısı
ageModel = "modelNweight/age_net.caffemodel"  # Yaş tahmini ağırlıkları

# Model için giriş görüntüsü ortalama değerleri tanımlanıyor
# Görüntülerimizi standartlaştırmak için her piksel değerinden bu ortalama değerler çıkarılarak normalize edilir.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Yaş aralıkları listesi
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Modeller yükleniyor
ageNet = cv.dnn.readNet(ageModel, ageProto)  # Yaş tahmini modeli
faceNet = cv.dnn.readNet(faceModel, faceProto)  # Yüz tespiti modeli

# Çerçeveler için ek bir boşluk tanımlanıyor
padding = 20

# Yaş tahmini yapan ana fonksiyon
def age_detector(frame):
    # Önce yüzler tespit ediliyor
    frameFace, bboxes = getFaceBox(faceNet, frame)

    # Eğer yüz bulunamazsa, ekrana "Yüz Bulunamadı" yazdırılıyor
    if not bboxes:
        cv.putText(frameFace, "Yuz Bulunamadi", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
        return frameFace
        
    # Tespit edilen her yüz için işlemler
    for bbox in bboxes:
        # Yüzün etrafına dikdörtgen çiziyoruz
        cv.rectangle(frameFace, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                    (0, 255, 0), int(round(frameFace.shape[0]/150)), 8)
        
        # Yüz bölgesini kırpıyoruz
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                    max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        # Eğer yüz bölgesi geçersizse işlemi atlıyoruz
        if face.size == 0:
            continue

        # Yüz bölgesinden blob oluşturup yaş tahmini yapıyoruz
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]  # En yüksek tahmin edilen yaş aralığı

        # Yaş tahmini sonuçlarını ekrana yazdırıyoruz
        cv.putText(frameFace, f"Yas: {age}", (bbox[0], bbox[1]-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
    return frameFace

# Ana GUI uygulaması

def main():
    root = tk.Tk()
    root.title("Yüz Tanıma Sistemi")
    root.geometry("400x300")

    global cap  # Kamera değişkenini global olarak tanımlıyoruz
    cap = None

    # Fotoğraf yükleme işlemi
    def fotograf_yukle():
        dosya_yolu = filedialog.askopenfilename()  # Kullanıcıdan dosya seçmesini istiyoruz
        if dosya_yolu:
            img = cv.imread(dosya_yolu)  # Görüntüyü yüklüyoruz
            output = age_detector(img)  # Yaş tahmini yapıyoruz
            cv.imshow('Sonuc', output)  # Sonucu ekrana gösteriyoruz
            cv.waitKey(0)
            cv.destroyAllWindows()

    # Kamera kapatma işlemi
    def kamera_kapat():
        global cap
        if cap is not None:
            cap.release()  # Kamerayı serbest bırak
            cv.destroyAllWindows()
            btn_kamera.config(state='normal')  # Kamera Aç butonunu aktif et
            btn_kapat.config(state='disabled')  # Kamera Kapat butonunu devre dışı bırak

    # Kamera açma işlemi
    def kamera_ac():
        global cap
        cap = cv.VideoCapture(0)  # Kamerayı başlatıyoruz
        btn_kamera.config(state='disabled')  # Kamera Aç butonunu devre dışı bırak
        btn_kapat.config(state='normal')  # Kamera Kapat butonunu aktif et
        
        # Canlı görüntü işleme fonksiyonu
        def update_frame():
            global cap
            if cap is not None:
                ret, frame = cap.read()  # Kameradan görüntü alınıyor
                if ret:
                    output = age_detector(frame)  # Yaş tahmini yapılıyor
                    cv.imshow('Kamera', output)
                    
                    # ESC veya 'q' tuşuna basılırsa kamerayı kapatıyoruz
                    key = cv.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):
                        kamera_kapat()
                        return
                    
                    # Pencere kapatıldıysa
                    if cv.getWindowProperty('Kamera', cv.WND_PROP_VISIBLE) < 1:
                        kamera_kapat()
                        return
                    
                    root.after(10, update_frame)  # 10ms sonra tekrar çağır

        update_frame()

    # GUI butonları
    btn_foto = tk.Button(root, text="Fotoğraf Yükle", command=fotograf_yukle)
    btn_foto.pack(pady=20)

    btn_kamera = tk.Button(root, text="Kamera Aç", command=kamera_ac)
    btn_kamera.pack(pady=20)

    btn_kapat = tk.Button(root, text="Kamera Kapat", command=kamera_kapat, state='disabled')
    btn_kapat.pack(pady=20)

    # Pencere kapatıldığında kamerayı da kapatıyoruz
    def on_closing():
        kamera_kapat()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# Ana program başlatılıyor
if __name__ == "__main__":
    main()
