import csv
import math
import os.path
import os
from os import path
from csv import writer
import cv2
from SpeedPrediction import SpeedPrediction
from lib.utils.common import CocoPart, CocoColors, CocoPairsRender


class DrawHuman:
    def __init__(self, debugMode=False):
        self.debug = debugMode
        self.nose_pixel_change = []
        self.leftHand_pixel_change = []
        self.rightHand_pixel_change = []
        self.leftFoot_pixel_change = []
        self.rightFoot_pixel_change = []

        self.noseFrameOne = []
        self.noseFrameTwo = []
        self.tempNoseFrameTwo = []

        self.leftHandFrameOne = []
        self.leftHandFrameTwo = []
        self.tempLeftHandFrameTwo = []

        self.rightHandFrameOne = []
        self.rightHandFrameTwo = []
        self.tempRightHandFrameTwo = []

        self.leftFootFrameOne = []
        self.leftFootFrameTwo = []
        self.tempLeftFootFrameTwo = []

        self.rightFootFrameOne = []
        self.rightFootFrameTwo = []
        self.tempRightFootFrameTwo = []
        self.headPredict = SpeedPrediction()
        self.leftArmPredict = SpeedPrediction()
        self.rightArmPredict = SpeedPrediction()
        self.leftLegPredict = SpeedPrediction()
        self.rightLegPredict = SpeedPrediction()
        if (debugMode):
            if (path.exists("coordinateData.csv")):
                os.remove("coordinateData.csv")
            with open('coordinateData.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Body", "Frame1Coord", "Frame2Coord", "IsItFromPE"])

    def draw_humans(self, npimg, humans, frameEven):
        self.npimg = npimg
        self.humans = humans
        self.frameEven = frameEven
        image_h, image_w = self.npimg.shape[:2]
        centers = {}
        for human in self.humans:
            # draw point
            for i in range(CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(self.npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
            self.analyzeBodyParts(human.body_parts, centers)
            # draw line
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(self.npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
        return self.npimg

    def analyzeBodyParts(self, bodyParts, points):
        identifiedParts = {}
        for i in range(CocoPart.Background.value):
            if i not in bodyParts.keys():
                continue
            identifiedParts[bodyParts[i].part_idx] = points[i]
        point = self.getCoordinatePerBodypart(identifiedParts, [0,1,14,15,16,17]) #head
        self.analyzeLimbs(point, self.noseFrameOne, self.noseFrameTwo, "Head")
        point = self.getCoordinatePerBodypart(identifiedParts, [13]) #Lfoot
        self.analyzeLimbs(point, self.leftFootFrameOne, self.leftFootFrameTwo, "LFoot")
        point = self.getCoordinatePerBodypart(identifiedParts, [10]) #Rfoot
        self.analyzeLimbs(point, self.rightFootFrameOne, self.rightFootFrameTwo, "RFoot")
        point = self.getCoordinatePerBodypart(identifiedParts, [7]) #Lhand
        self.analyzeLimbs(point, self.leftHandFrameOne, self.leftHandFrameTwo, "LHand")
        point = self.getCoordinatePerBodypart(identifiedParts, [4]) #Rhand
        self.analyzeLimbs(point, self.rightHandFrameOne, self.rightHandFrameTwo, "RHand")

    def getCoordinatePerBodypart(self, identifiedParts, ids):
        for body in ids:
            if body in identifiedParts:
                return identifiedParts.get(body)
        return None


    def analyzeLimbs(self, points, frameOne, frameTwo, limbName):
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (points)
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(self.npimg, limbName, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
            if (self.frameEven):
                frameTwo.append(points)
            else:
                frameOne.append(points)

    def calculateSpeedForIndLimbs(self):
        self.pixelChangeCalculator(self.noseFrameOne, self.noseFrameTwo, self.nose_pixel_change)
        self.pixelChangeCalculator(self.leftHandFrameOne, self.leftHandFrameTwo, self.leftHand_pixel_change)
        self.pixelChangeCalculator(self.rightHandFrameOne, self.rightHandFrameTwo, self.rightHand_pixel_change)
        self.pixelChangeCalculator(self.leftFootFrameOne, self.leftFootFrameTwo, self.leftFoot_pixel_change)
        self.pixelChangeCalculator(self.rightFootFrameOne, self.rightFootFrameTwo, self.rightFoot_pixel_change)

    def transferFrameTwoToTemp(self):
        self.updateArrays(self.noseFrameOne, self.noseFrameTwo, self.tempNoseFrameTwo)
        self.updateArrays(self.leftHandFrameOne, self.leftHandFrameTwo, self.tempLeftHandFrameTwo)
        self.updateArrays(self.rightHandFrameOne, self.rightHandFrameTwo, self.tempRightHandFrameTwo)
        self.updateArrays(self.leftFootFrameOne, self.leftFootFrameTwo, self.tempLeftFootFrameTwo)
        self.updateArrays(self.rightFootFrameOne, self.rightFootFrameTwo, self.tempRightFootFrameTwo)

    def updateArrays(self, frameOne, frameTwo, frameTwoTemp):
        frameOne.clear()
        for i in range(len(frameTwo)):
            frameTwoTemp.append(frameTwo[i])
        frameTwo.clear()

    def pixelChangeCalculator(self, frameOne, frameTwo, pixelChange):
        for i in range(len(self.humans)):
            try:
                frame1 = frameOne[i]
                frame2 = frameTwo[i]
                pixelChange.append(self.pixelChangeAlgorithm(frame1, frame2))
            except:
                pixelChange.append(None)

    def pixelChangeAlgorithm(self, tupleFrame1, tupleFrame2):
        firstFrame = tupleFrame1
        secondFrame = tupleFrame2
        changeInX = abs(firstFrame[0] - secondFrame[0]) ** 2
        changeInY = abs(firstFrame[1] - secondFrame[1]) ** 2
        moveDistance = math.sqrt(changeInX + changeInY)
        moveDistance = round(moveDistance, 2)
        return moveDistance

    def getSpeed(self):
        list_of_speed = [self.nose_pixel_change, self.leftHand_pixel_change, self.rightHand_pixel_change,
                         self.leftFoot_pixel_change, self.rightFoot_pixel_change]
        return list_of_speed

    def clearSpeedData(self):
        self.nose_pixel_change = []
        self.leftHand_pixel_change = []
        self.rightHand_pixel_change = []
        self.leftFoot_pixel_change = []
        self.rightFoot_pixel_change = []

    def distance(self, x, coordinate):
        if (x != None and coordinate != None):
            x_value = abs(x[0] - coordinate[0])
            y_value = abs(x[1] - coordinate[1])
        else:
            return 10000  # large number because it cannot be compared to None
        return x_value + y_value

    def syncFrameTwoWithFrameOne(self):
        self.trackingAdjacentFrame(self.noseFrameOne, self.noseFrameTwo)
        self.removeWorst(self.noseFrameOne,self.noseFrameTwo)
        self.trackingAdjacentFrame(self.leftHandFrameOne, self.leftHandFrameTwo)
        self.removeWorst(self.leftHandFrameOne, self.leftHandFrameTwo)
        self.trackingAdjacentFrame(self.rightHandFrameOne, self.rightHandFrameTwo)
        self.removeWorst(self.rightHandFrameOne, self.rightHandFrameTwo)
        self.trackingAdjacentFrame(self.leftFootFrameOne, self.leftFootFrameTwo)
        self.removeWorst(self.leftFootFrameOne,self.leftHandFrameTwo)
        self.trackingAdjacentFrame(self.rightFootFrameOne, self.rightFootFrameTwo)
        self.removeWorst(self.rightFootFrameOne, self.rightFootFrameTwo)

    def removeWorst(self,f1, f2):
        record = []
        if (len(f2) != len(self.humans)):
            for i in range(len(f1)):
                try:
                    x = abs(f1[i][0] - f2[i][0])
                    y = abs(f1[i][1] - f2[i][1])
                    record.append(x + y)
                except:
                    record.append(10000)
        while (len(f2) != len(self.humans)):
            try:
                index = record.index(max(record))
                del record[index]
                del f1[index]
                del f2[index]
            except:
                pass

    def trackingAdjacentFrame(self, frameOne, frameTwo):
        try:
            newFrame = []
            for i in range(len(frameOne)):
                newFrame.append(min(frameTwo, key=lambda x: self.distance(x, frameOne[i])))
            if (len(frameOne) != len(frameTwo)):
                extras = self.checkForMissingCoordinate(frameTwo, newFrame)
                for i in range(len(extras)):
                    newFrame.append(extras[i])
            frameTwo.clear()
            for i in range(len(newFrame)):
                frameTwo.append(newFrame[i])
        except:
            pass

    def syncFrameOneWithFrameTwo(self):
        self.fixTracking(self.noseFrameOne, self.tempNoseFrameTwo)
        self.fixTracking(self.leftHandFrameOne, self.tempLeftHandFrameTwo)
        self.fixTracking(self.rightHandFrameOne, self.tempRightHandFrameTwo)
        self.fixTracking(self.leftFootFrameOne, self.tempLeftFootFrameTwo)
        self.fixTracking(self.rightFootFrameOne, self.tempRightFootFrameTwo)

    def checkForMissingCoordinate(self, one, two):
        return list(set(one) - set(two))

    def fixTracking(self, frameOne, tempFrame):
        try:
            if tempFrame:
                newFrame = []
                for i in range(len(tempFrame)):
                    newFrame.append(min(frameOne, key=lambda x: self.distance(x, tempFrame[i])))
                if (len(frameOne) != len(tempFrame)):
                    extras = self.checkForMissingCoordinate(frameOne, newFrame)
                    for i in range(len(extras)):
                        newFrame.append(extras[i])
                frameOne.clear()
                for i in range(len(newFrame)):
                    frameOne.append(newFrame[i])
                tempFrame.clear()
        except:
            pass

    def updatePredictEngine(self):
        PredictionEngine.setHumans(self.humans)
        self.headPredict.createSpeedMatrix(PredictionEngine.setSpeedForThePredictor(self.noseFrameOne, self.noseFrameTwo))
        self.headPredict.setLatestCoord(self.noseFrameTwo)

        self.leftArmPredict.createSpeedMatrix(PredictionEngine.setSpeedForThePredictor(self.leftHandFrameOne, self.leftHandFrameTwo))
        self.leftArmPredict.setLatestCoord(self.leftHandFrameTwo)

        self.rightArmPredict.createSpeedMatrix(PredictionEngine.setSpeedForThePredictor(self.rightHandFrameOne, self.rightHandFrameTwo))
        self.rightArmPredict.setLatestCoord(self.rightHandFrameTwo)

        self.leftLegPredict.createSpeedMatrix(PredictionEngine.setSpeedForThePredictor(self.leftFootFrameOne, self.leftFootFrameTwo))
        self.leftLegPredict.setLatestCoord(self.leftFootFrameTwo)

        self.rightLegPredict.createSpeedMatrix(PredictionEngine.setSpeedForThePredictor(self.rightFootFrameOne, self.rightFootFrameTwo))
        self.rightLegPredict.setLatestCoord(self.rightFootFrameTwo)

    def fixMissingCoordinate(self,evenFrame):
        dataF1 = [self.noseFrameOne, self.leftHandFrameOne, self.rightHandFrameOne, self.leftFootFrameOne,
                  self.rightFootFrameOne]
        dataF2 = [self.noseFrameTwo, self.leftHandFrameTwo, self.rightHandFrameTwo, self.leftHandFrameTwo,
                  self.rightHandFrameTwo]
        if (evenFrame):
            data = dataF2
        else:
            data = dataF1
        data[0] = self.headPredict.predictMissingCoordination(data[0])
        self.headPredict.setLatestCoord(data[0])
        data[1] = self.leftArmPredict.predictMissingCoordination(data[1])
        self.leftArmPredict.setLatestCoord(data[1])
        data[2] = self.rightArmPredict.predictMissingCoordination(data[2])
        self.rightArmPredict.setLatestCoord(data[2])
        data[3] = self.leftLegPredict.predictMissingCoordination(data[3])
        self.leftLegPredict.setLatestCoord(data[3])
        data[4] = self.rightLegPredict.predictMissingCoordination(data[4])
        self.rightLegPredict.setLatestCoord(data[4])

    def writeDataToCSV(self):
        dataF1 = [self.noseFrameOne, self.leftHandFrameOne, self.rightHandFrameOne, self.leftFootFrameOne,
                  self.rightFootFrameOne]
        dataF2 = [self.noseFrameTwo, self.leftHandFrameTwo, self.rightHandFrameTwo, self.leftHandFrameTwo,
                  self.rightHandFrameTwo]
        dataName = ["Head", "LeftArm", "RightArm", "LeftLeg", "RightLeg"]
        for i in range(len(self.humans)):
            for j in range(len(dataF1)):
                try:
                    subDataF1 = str(dataF1[j][i])
                    subDataF2 = str(dataF2[j][i])
                    subDataF1.strip("()")
                    subDataF2.strip("()")
                except:
                    subDataF1 = "None"
                    subDataF2 = "None"
                self.append_list_as_row(dataName[j]+str(i), subDataF1, subDataF2, "True")

    def append_list_as_row(self, a1, a2, a3, a4):
        # Open file in append mode
        with open("coordinateData.csv", 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj)
            # Add contents of list as last row in the csv file
            csv_writer.writerow([a1, a2, a3, a4])

class PredictionEngine:

    humans = None
    def setHumans(humans):
        PredictionEngine.humans = humans

    def setSpeedForThePredictor(frameOne, frameTwo):
        transition = []
        for i in range(len(PredictionEngine.humans)):
            try:
                a = frameOne[i]
                b = frameTwo[i]
                transition.append([])
                transition[i].append(a)
                transition[i].append(b)
            except:
                transition.append([])
                transition[i].append(None)
        return transition