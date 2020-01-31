import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

video = cv2.VideoCapture("dataset/Assault/assault6.mp4");
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
    ret, oriImg = video.read()
    (rects, weights) = hog.detectMultiScale(oriImg, winStride=(4, 4),
                                            padding=(8, 8), scale=3.2)
    for (x, y, w, h) in rects:
        cv2.rectangle(oriImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
        print("Height is " + str(h))

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(oriImg, (xA, yA), (xB, yB), (0, 255, 0), 2)
        print("ya and yb is " + str(yA - yB))

    cv2.imshow('Video', oriImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break