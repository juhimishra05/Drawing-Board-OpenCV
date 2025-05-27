# handTrackingModule.py

import math
import mediapipe as mp
import cv2

class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []
        self.fingerTips = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPostition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            return [], 0

        # Thumb (adjusted for flipped webcam)
        if self.lmList[self.fingerTips[0]][1] > self.lmList[self.fingerTips[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for i in range(1, 5):
            if self.lmList[self.fingerTips[i]][2] < self.lmList[self.fingerTips[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers, fingers.count(1)
