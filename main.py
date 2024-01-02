import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model1.h5", "Model/labels1.txt")

offset = 20
imgSize = 300
# Untuk pengujian
labels = ["A", "B", "C",
          "D", "E", "F", "G", "H", "I", "K",
          "L", "M", "N", "O", "P", "Q", "R",
          "S", "T", "U", "V", "W", "X", "Y",
          ]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    # Looping tanpa batasan pembacaan pola tangan, menentukan tindakan ketika tangan terbaca
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspecRatio = h / w

        # Menghitung ukuran citra (imgResize) berdasarkan rasio
        # untuk dinormalkan ukurannya
        if aspecRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Menampilkan kotak pembatas tangan dan huruf prediksi
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50),
                      (255, 0, 255), (cv2.FILLED))
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset),
                      (255, 0, 255), 4)
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7,
                    (255, 255, 255), 2)

        # Menampilkan tangkapan kamera dgn menandai tangan yg terbaca oleh sistem
        # cv2.imshow("Image", img)
        cv2.imshow("Image", imgOutput)
        cv2.waitKey(1)