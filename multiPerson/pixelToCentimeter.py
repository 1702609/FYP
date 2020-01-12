from statistics import median

import cv2
from math import sqrt

class pixelToCentimeter:
    def __init__(self, videoPath):
        self.targetVideo = videoPath

    def calculateCentimeterPerPixel(self):
        fullBodyCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

        video = cv2.VideoCapture(self.targetVideo);
        frames = 0
        heightArray = []
        print("The width of video is "+ str(video.get(3)))  # float
        print("The height of video is "+str(video.get(4)))
        while video.isOpened() and frames <= 120:
            frames += 1
            _, img = video.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            body = fullBodyCascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in body:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                if frames >= 20:
                    heightArray.append(h)
                    print("The height is " + str(h))
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()

        averageHeight = 190  # Average height

        clusterGroup = self.parse(heightArray, 9)
        meanData = []
        for cluster in clusterGroup:
            meanData.append(self.Average(cluster))
        meanData.sort()
        answer = median(meanData)/averageHeight
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
