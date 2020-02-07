class SpeedPrediction:
    def __init__(self):
        self.predictionUsed = 0
        self.transitionArray= []
        self.latestCoordinate = []

    def setLatestCoord(self, frame):
        self.latestCoordinate = frame

    def createSpeedMatrix(self, speedData):
        transitionArray = []
        for i in range(len(speedData)):
            try:
                changeX = speedData[i][1][0] - speedData[i][0][0]
                changeY = speedData[i][1][1] - speedData[i][0][1]
                transitionArray.append((changeX, changeY))
            except:
                transitionArray.append(None)
        self.compareItWithBackUp(transitionArray)

    def compareItWithBackUp(self, checkArray):
        if len(self.transitionArray) != 0:  # check if back up in NOT empty
            for i in range(len(self.transitionArray)):
                try:
                    if checkArray[i] is None and self.transitionArray[i] is not None:
                        checkArray[i] = self.transitionArray[i]
                except:
                    pass
        self.transitionArray = checkArray

    def getSpeedMatrix(self):
        return self.transitionArray

    def predictMissingCoordination(self, frame):
        count = 0
        for i in frame:
            if (type(i) != tuple and len(self.latestCoordinate)!= 0):
                try:
                    if (type(self.latestCoordinate[count]) == tuple):
                        x = self.latestCoordinate[count][0] + self.transitionArray[count][0]
                        y = self.latestCoordinate[count][1] + self.transitionArray[count][1]
                        frame[count] = (x, y)
                except:
                    pass
            count += 1
        return frame