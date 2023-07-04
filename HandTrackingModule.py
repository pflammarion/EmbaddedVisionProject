import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_conf=0.5, track_conf=0.5):
        self.results = None
        self.mode = mode
        self.maxHands = max_hands
        self.modelComplexity = model_complexity
        self.detectionConf = detection_conf
        self.trackConf = track_conf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConf,
                                        self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results.multi_hand_landmarks:
            selected_hand = self.results.multi_hand_landmarks[hand_no]
            for lm_id, lm in enumerate(selected_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([lm_id, cx, cy])

        return lm_list
