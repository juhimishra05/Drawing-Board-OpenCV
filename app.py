# main.py

import cv2
import numpy as np
import time
import os
from handTrackingModule import HandDetector

# Setup
wCam, hCam = 1280, 720
capture = cv2.VideoCapture(0)
capture.set(3, wCam)
capture.set(4, hCam)

# Load header images
folderPath = "assets/drawing"
overlayList = [cv2.imread(f"{folderPath}/{imPath}") for imPath in sorted(os.listdir(folderPath))]
header = overlayList[0]

# Drawing variables
drawColor = (255, 0, 0)
brushThickness = 15
eraserThickness = 50
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# Hand detector
detector = HandDetector(maxHands=1)

# FPS
pTime = 0

while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPostition(img, draw=False)

    if lmList:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers, _ = detector.fingersUp()

        if fingers[1] == 1 and fingers[2] == 1:
            xp, yp = 0, 0
            # Selection Mode
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        elif fingers[1] == 1 and fingers[2] == 0:
            # Drawing Mode
            cv2.circle(img, (x1, y1), 12, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

        # Clear screen with all fingers up
        if fingers.count(1) == 5:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

    # Merge canvas and camera feed
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Add header
    img[0:header.shape[0], 0:header.shape[1]] = header

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (50, 680), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show windows
    cv2.imshow("Drawing", img)
    # cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
