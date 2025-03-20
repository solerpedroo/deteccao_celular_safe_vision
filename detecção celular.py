import cv2
import cvzone
from ultralytics import YOLO
import winsound
import threading
import pygame
import time

video = cv2.VideoCapture(0)
video.set(3,1280)
video.set(4,720)

modelo = YOLO('yolov8n.pt')

controleAlarme = False

def alarme():
    global controleAlarme
    pygame.init()
    pygame.mixer.music.load('audio_para_guardar_celular.mp3')
    pygame.mixer.music.play()
    while time.sleep(3.5):
        # Espera até o som atual terminar de tocar
        pass

    controleAlarme = False

while True:
    check,img = video.read()
    # img = cv2.resize(img,(1280,720))

    resultado = modelo.predict(img,conf=0.5)

    for objetos in resultado:
        obj = objetos.boxes
        for dados in obj:
            #bbox
            x,y,w,h = dados.xyxy[0]
            x, y, w, h = int(x),int(y),int(w),int(h)
            cls = int(dados.cls[0])
            print(x,y,w,h,cls)
            if cls==67:
                cv2.rectangle(img,(x,y),(w,h),(255,0,255),5)
                cvzone.putTextRect(img,"CELULAR IDENTIFICADO",(105,65),colorR=(0,0,255))
                if not controleAlarme:
                    controleAlarme =True
                    threading.Thread(target=alarme).start()

    cv2.imshow('IMG',img)
    cv2.waitKey(1)