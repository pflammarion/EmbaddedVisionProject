import cv2
import time
import HandTrackingModule as Htm

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = Htm.HandDetector()
while True:
    success, img = cap.read()
    img = detector.find_hands(img)
    lmList = detector.find_position(img)
    if len(lmList) != 0:
        print("hand detected")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
