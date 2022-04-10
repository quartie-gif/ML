#!/usr/bin/python

from re import L
import cv2
import time
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math
from collections import deque

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

pts = deque(maxlen=512)
blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
blackboard_copy = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    success, img = cap.read()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Using detector module to detect hands
    img = detector.findHands(img)
    # Geting position of hands
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) > 0:

        # Getting the position of thumbs and index finger
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Calculating the distance between the thumb and index finger
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Marking the position of the thumb and index finger using circles
        cv2.circle(img, (x1, y1), 15, (0, 150, 150), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (0, 150, 150), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 150, 150), 3)
        cv2.circle(img, (cx, cy), 15, (0, 150, 150), cv2.FILLED)

        # Calculating the distance between the thumb and index finger
        center = (int(x2-x1), int(y2/y1))
        # TODO 
        # pts.appendleft(center)
        # for i in range(1, len(pts)):
        #     if pts[i - 1] is None or pts[i] is None:
        #         continue
        #     cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)
        #     # cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 2)
        # cv2.circle(img, center 15, (0, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1.0 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('img', img)
    # cv2.imshow("black", blackboard_copy)
    cv2.waitKey(1)
