import threading

import cv2 as cv
import numpy as np
import argparse
from tkinter import *
import math

from pixelToCentimeter import pixelToCentimeter

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')


class PoseEstimationVideo:
    def __init__(self, videoPath, cmPerPixel):
        self.video = videoPath
        self.cmPerPixel = cmPerPixel
        self.window = Tk()
        self.args = parser.parse_args()
        self.inWidth = self.args.width
        self.inHeight = self.args.height
        self.fps = 0
        self.headChange = []
        self.leftHandChange = []
        self.rightHandChange = []
        self.leftLegChange = []
        self.rightLegChange = []
        self.gui()

    def gui(self):
        self.window.title("Movement Stats")
        self.window.minsize(500, 300)
        head = Label(self.window, text="head")
        head.grid(column=0, row=0)
        self.headSpeed = Label(self.window, text="Detecting")
        self.headSpeed.grid(column=1, row=0)

        leftHand = Label(self.window, text="left hand")
        leftHand.grid(column=0, row=1)
        self.leftHandSpeed = Label(self.window, text="Detecting")
        self.leftHandSpeed.grid(column=1, row=1)

        rightHand = Label(self.window, text="right hand")
        rightHand.grid(column=0, row=2)
        self.rightHandSpeed = Label(self.window, text="Detecting")
        self.rightHandSpeed.grid(column=1, row=2)

        leftFoot = Label(self.window, text="left foot")
        leftFoot.grid(column=0, row=3)
        self.leftFootSpeed = Label(self.window, text="Detecting")
        self.leftFootSpeed.grid(column=1, row=3)

        rightFoot = Label(self.window, text="right foot")
        rightFoot.grid(column=0, row=4)
        self.rightFootSpeed = Label(self.window, text="Detecting")
        self.rightFootSpeed.grid(column=1, row=4)

        self.window.update()

    def runAnalysis(self):
        self.initialiseBodyParts()
        self.loadData()
        self.analyzeVideo()
        self.window.mainloop()

    def initialiseBodyParts(self):
        self.BODY_PARTS = {"Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                           "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                           "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                           "LEye": 15, "REar": 16, "LEar": 17, "Background": 18}

        self.POSE_PAIRS = [["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
                           ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
                           ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
                           ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
                           ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

    def loadData(self):
        self.net = cv.dnn.readNetFromTensorflow("graph_freeze.pb")

        self.cap = cv.VideoCapture(self.video)

    def analyzeVideo(self):
        pos_frame = self.cap.get(cv.CAP_PROP_POS_FRAMES)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        while cv.waitKey(1) < 0:
            hasFrame, self.frame = self.cap.read()
            if not hasFrame:
                self.cap.set(cv.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                cv.waitKey(500)

            frameWidth = self.frame.shape[1]
            frameHeight = self.frame.shape[0]

            self.net.setInput(
                cv.dnn.blobFromImage(self.frame, 1.0, (self.inWidth, self.inHeight), (127.5, 127.5, 127.5), swapRB=True,
                                     crop=False))
            out = self.net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

            assert (len(self.BODY_PARTS) == out.shape[1])

            points = []
            for i in range(len(self.BODY_PARTS)):
                # Slice heatmap of corresponging body's part.
                heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums. To simplify a sample
                # we just find a global one. However only a single pose at the same time
                # could be detected this way.
                _, conf, _, point = cv.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                # Add a point if it's confidence is higher than threshold.
                points.append((int(x), int(y)) if conf > self.args.thr else None)

            for pair in self.POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert (partFrom in self.BODY_PARTS)
                assert (partTo in self.BODY_PARTS)
                idFrom = self.BODY_PARTS[partFrom]
                idTo = self.BODY_PARTS[partTo]
                if points[idFrom] and points[idTo]:
                    cv.line(self.frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv.ellipse(self.frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                    cv.ellipse(self.frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                self.analyzeBodyParts(idFrom, idTo, points)
            t, _ = self.net.getPerfProfile()
            freq = cv.getTickFrequency() / 1000
            cv.putText(self.frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            self.window.update()
            cv.imshow('OpenPose using OpenCV', self.frame)

    def analyzeHead(self, idFrom, idTo, points):
        if idFrom == 1 and idTo == 0:
            try:
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (points[idFrom][0], points[idFrom][1])
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv.putText(self.frame, 'Head', org, font,
                           fontScale, color, thickness, cv.LINE_AA)
                if points[idFrom] and points[idTo] != None:
                    cv.rectangle(self.frame, points[idFrom], points[idTo], color, 1)
                    if idFrom or idTo == 0:
                        self.calculateMovement(points, 0)
            except:
                pass

    def analyzeLeftHand(self, idFrom, idTo, points):
        if idFrom == 7 or idFrom == 6:
            try:
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (points[idFrom][0], points[idFrom][1])
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv.putText(self.frame, 'LeftHand', org, font,
                           fontScale, color, thickness, cv.LINE_AA)
                cv.rectangle(self.frame, points[idFrom][0], points[idFrom][1], color, 1)
                if points[idFrom] and points[idTo] is not None:
                    cv.rectangle(self.frame, points[idFrom], points[idTo], color, 1)
                    if idFrom or idTo == 7:
                        self.calculateMovement(points, 7)
            except:
                pass

    def analyzeRightHand(self, idFrom, idTo, points):
        if idFrom == 4 or idFrom == 3:
            try:
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (points[idFrom][0], points[idFrom][1])
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv.putText(self.frame, 'RightHand', org, font,
                           fontScale, color, thickness, cv.LINE_AA)
                if points[idFrom] and points[idTo] is not None:
                    cv.rectangle(self.frame, points[idFrom], points[idTo], color, 1)
                    self.calculateMovement(points, 4)
            except:
                pass

    def analyzeLeftFoot(self, idFrom, idTo, points):
        if idFrom == 13 or idFrom == 12:
            try:
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (points[idFrom][0], points[idFrom][1])
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv.putText(self.frame, 'LeftFoot', org, font,
                           fontScale, color, thickness, cv.LINE_AA)
                if points[idFrom] and points[idTo] != None:
                    cv.rectangle(self.frame, points[idFrom], points[idTo], color, 1)
                    self.calculateMovement(points, 13)
            except:
                pass

    def analyzeRightFoot(self, idFrom, idTo, points):
        if idFrom == 10 or idFrom == 9:
            try:
                font = cv.FONT_HERSHEY_SIMPLEX
                org = (points[idFrom][0], points[idFrom][1])
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv.putText(self.frame, 'RightFoot', org, font,
                           fontScale, color, thickness, cv.LINE_AA)
                if points[idFrom] and points[idTo] != None:
                    cv.rectangle(self.frame, points[idFrom], points[idTo], color, 1)
                    self.calculateMovement(points, 10)
            except:
                pass

    def analyzeBodyParts(self, idFrom, idTo, points):
        self.analyzeHead(idFrom, idTo, points)
        self.analyzeLeftFoot(idFrom, idTo, points)
        self.analyzeRightFoot(idFrom, idTo, points)
        self.analyzeLeftHand(idFrom, idTo, points)
        self.analyzeRightHand(idFrom, idTo, points)

    def calculateMovement(self, points, target):
        if target == 0:
            self.headChange.append(points[target])
        elif target == 7:
            self.leftHandChange.append(points[target])
        elif target == 4:
            self.rightHandChange.append(points[target])
        elif target == 13:
            self.leftLegChange.append(points[target])
        elif target == 10:
            self.rightLegChange.append(points[target])
        self.updateGUIStats()

    def updateGUIStats(self):
        perSecond = (1 / self.fps) * 2
        timeNeeded = 1 / perSecond
        if len(self.headChange) == 2:
            moveDistance = self.pixelChangeCalculator(self.headChange)
            self.headChange.clear()
            self.headSpeed.config(text="It moved " + str(moveDistance * timeNeeded) + "cm per second")
        if len(self.leftHandChange) == 2:
            moveDistance = self.pixelChangeCalculator(self.leftHandChange)
            self.leftHandChange.clear()
            self.leftHandSpeed.config(text="It moved " + str(moveDistance * timeNeeded) + "cm per second")
        if len(self.rightHandChange) == 2:
            moveDistance = self.pixelChangeCalculator(self.rightHandChange)
            self.rightHandChange.clear()
            self.rightHandSpeed.config(text="It moved " + str(moveDistance * timeNeeded) + "cm per second")
        if len(self.leftLegChange) == 2:
            moveDistance = self.pixelChangeCalculator(self.leftLegChange)
            self.leftLegChange.clear()
            self.leftLegSpeed.config(text="It moved " + str(moveDistance * timeNeeded) + "cm per second")
        if len(self.rightLegChange) == 2:
            moveDistance = self.pixelChangeCalculator(self.rightLegChange)
            self.rightLegChange.clear()
            self.rightLegSpeed.config(text="It moved " + str(moveDistance * timeNeeded) + "cm per second")

    def pixelChangeCalculator(self, fullArray):
        firstFrame = fullArray[0]
        secondFrame = fullArray[1]
        changeInX = abs(firstFrame[0] - secondFrame[0]) ** 2
        changeInY = abs(firstFrame[1] - secondFrame[1]) ** 2
        moveDistance = math.sqrt(changeInX + changeInY) * self.cmPerPixel
        moveDistance = round(moveDistance, 2)
        return moveDistance


print("Calculating how much pixel is needed to cover people's height")
videoPath = "dataset/Test/starJump.mp4"
preTest = pixelToCentimeter(videoPath)
cmPerPixel = preTest.calculateCentimeterPerPixel()

print("1 pixel is equal to " + str(cmPerPixel) + " cm")
p1 = PoseEstimationVideo(videoPath, cmPerPixel)
p1.runAnalysis()
