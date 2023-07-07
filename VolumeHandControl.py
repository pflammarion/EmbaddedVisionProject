import cv2
import HandTrackingModule as Htm
import numpy as np
#import autopy

###############################################
camera_width, camera_height = 640, 480
###############################################


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
cap.set(3, camera_width)
cap.set(4, camera_height)

detector = Htm.HandDetector(max_hands=1)

# to get the screen size


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

        #if clicked:
            #x3 = np.interp(x1, (0, camera_width), (0, screen_width))
            #y3 = np.interp(y1, (0, camera_height), (0, screen_height))



    cv2.imshow("Image", img)
    cv2.waitKey(1)