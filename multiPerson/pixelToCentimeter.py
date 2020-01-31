from statistics import median

import cv2
from math import sqrt

import numpy as np
from imutils.object_detection import non_max_suppression


class pixelToCentimeter:
    def __init__(self, videoPath):
        self.targetVideo = videoPath

    def calculateCentimeterPerPixel(self):
        video = cv2.VideoCapture(self.targetVideo);
        frames = 0
        heightArray = []
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        while video.isOpened() and frames <= 120:
            frames += 1
            ret, oriImg = video.read()
            (rects, weights) = hog.detectMultiScale(oriImg, winStride=(4, 4),
                                                    padding=(8, 8), scale=3.2)
            for (x, y, w, h) in rects:
                cv2.rectangle(oriImg, (x, y), (x + w, y + h), (0, 0, 255), 2)
                heightArray.append(h)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(oriImg, (xA, yA), (xB, yB), (0, 255, 0), 2)

            cv2.imshow('Video', oriImg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

        averageHeight = 170

        clusterGroup = self.parse(heightArray, 9)
        meanData = []
        for cluster in clusterGroup:
            meanData.append(self.Average(cluster))
        meanData.sort()
        answer = averageHeight / median(meanData)
        return answer

    def Average(self, lst):
        a = round(sum(lst) / len(lst), 2)
        return a

    def stat(self, lst):
        """Calculate mean and std deviation from the input list."""
        n = float(len(lst))
        mean = sum(lst) / n
        stdev = sqrt((sum(x * x for x in lst) / n) - (mean * mean))
        return mean, stdev

    def parse(self, lst, n):
        cluster = []
        for i in lst:
            if len(cluster) <= 1:  # the first two values are going directly in
                cluster.append(i)
                continue
            mean, stdev = self.stat(cluster)
            if abs(mean - i) > n * stdev:  # check the "distance"
                yield cluster
                cluster[:] = []  # reset cluster to the empty list
            cluster.append(i)
        yield cluster