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
            if (type(i) != tuple):
                if (type(self.latestCoordinate[count]) == tuple and type(self.transitionArray[count]) == tuple):
                    x = self.latestCoordinate[count][0] + self.transitionArray[count][0]
                    y = self.latestCoordinate[count][1] + self.transitionArray[count][1]
                    frame[count] = (x, y)
            count += 1
        return frame


test1 = SpeedPrediction()

def setSpeedForThePredictor(frameOne, frameTwo):
    transition = []
    for i in range(4):
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

f1 = [(646,456),(771,264),None,(100,100)]
f2 = [(644,462),(773,264),None,(105,106)]
f3 = [None,(776,268),None,None]



test1.createSpeedMatrix(setSpeedForThePredictor(f1, f2))
test1.setLatestCoord(f2)
#print(test1.getSpeedMatrix())
f3 = test1.predictMissingCoordination(f3)
test1.setLatestCoord(f3)
f4 = [(655,460),None,(231,23),None] #returned by pose estimation
test1.createSpeedMatrix(setSpeedForThePredictor(f2, f3))
print(test1.getSpeedMatrix())
f4=test1.predictMissingCoordination(f4)
print(f4)