import cv2
import HandTrackingModule as Htm
import time
import numpy as np
import serial

###############################################
camera_width, camera_height = 640, 480
###############################################


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

detector = Htm.HandDetector(max_hands=1)

arduino = serial.Serial('/dev/ttyACM0', 9600)  

time.sleep(2)

clicked = False

while True:
    success, img = cap.read()

    img = detector.find_hands(img, draw=False)
    lmList, bbox = detector.find_position(img)

    if len(lmList) != 0:

        length, img, [x1, y1, x2, y2, cx, cy] = detector.find_distance(4, 8, img)

        if length < 30:
            clicked = True
            cv2.putText(img, "Clicked !", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        if length > 60:
            clicked = False

        if clicked:
            arduino.write(b'1')
            print("LED turned on")

        else:
            arduino.write(b'0')
            print("LED turned off")


