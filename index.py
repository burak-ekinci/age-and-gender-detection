# Import required modules
import cv2 as cv
import tkinter as tk
from tkinter import filedialog


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            confidences.append(float(confidence))

    if len(bboxes) > 0:
        indices = cv.dnn.NMSBoxes(bboxes, confidences, conf_threshold, 0.3)
        filtered_bboxes = []
        for idx in indices:
            if isinstance(idx, tuple):
                idx = idx[0]
            filtered_bboxes.append(bboxes[idx])
        return frameOpencvDnn, filtered_bboxes
    
    return frameOpencvDnn, []

faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"
ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Sadece gerekli ağları yükle
ageNet = cv.dnn.readNet(ageModel, ageProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 20

def age_detector(frame):
    frameFace, bboxes = getFaceBox(faceNet, frame)
    
    if not bboxes:
        cv.putText(frameFace, "Yuz Bulunamadi", (10, 30),
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
        return frameFace
        
    for bbox in bboxes:
        # Yüz için dikdörtgen çiz
        cv.rectangle(frameFace, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                    (0, 255, 0), int(round(frameFace.shape[0]/150)), 8)
        
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                    max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        
        if face.size == 0:
            continue

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        cv.putText(frameFace, f"Yas: {age}", (bbox[0], bbox[1]-10), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
    return frameFace

def main():
    root = tk.Tk()
    root.title("Yüz Tanıma Sistemi")
    root.geometry("400x300")

    # Kamera değişkenini global olarak tanımlayalım
    global cap
    cap = None

    def fotograf_yukle():
        dosya_yolu = filedialog.askopenfilename()
        if dosya_yolu:
            img = cv.imread(dosya_yolu)
            output = age_detector(img)
            cv.imshow('Sonuc', output)
            cv.waitKey(0)
            cv.destroyAllWindows()

    def kamera_kapat():
        global cap
        if cap is not None:
            cap.release()
            cv.destroyAllWindows()
            btn_kamera.config(state='normal')  # Kamera Aç butonunu aktif et
            btn_kapat.config(state='disabled')  # Kamera Kapat butonunu devre dışı bırak

    def kamera_ac():
        global cap
        cap = cv.VideoCapture(0)
        btn_kamera.config(state='disabled')  # Kamera Aç butonunu devre dışı bırak
        btn_kapat.config(state='normal')  # Kamera Kapat butonunu aktif et
        
        def update_frame():
            global cap
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    output = age_detector(frame)
                    cv.imshow('Kamera', output)
                    
                    # ESC veya 'q' tuşuna basılırsa kamerayı kapat
                    key = cv.waitKey(1) & 0xFF
                    if key == 27 or key == ord('q'):  # ESC=27
                        kamera_kapat()
                        return
                    
                    # Pencere kapatıldıysa
                    if cv.getWindowProperty('Kamera', cv.WND_PROP_VISIBLE) < 1:
                        kamera_kapat()
                        return
                    
                    root.after(10, update_frame)  # 10ms sonra tekrar çağır

        update_frame()

    btn_foto = tk.Button(root, text="Fotoğraf Yükle", command=fotograf_yukle)
    btn_foto.pack(pady=20)

    btn_kamera = tk.Button(root, text="Kamera Aç", command=kamera_ac)
    btn_kamera.pack(pady=20)

    btn_kapat = tk.Button(root, text="Kamera Kapat", command=kamera_kapat, state='disabled')
    btn_kapat.pack(pady=20)

    # Pencere kapatıldığında kamerayı da kapat
    def on_closing():
        kamera_kapat()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

# Ana programı başlat
if __name__ == "__main__":
    main()
