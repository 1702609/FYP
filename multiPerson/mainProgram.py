from SpeedPrediction import SpeedPrediction

leftArmPredict = SpeedPrediction()
rightArmPredict = SpeedPrediction()

leftArmF1 = [(646, 456),(771, 264)]
leftArmF2 = [(644, 462),(773, 264)]

rightArmF1 = [(400, 479),None]
rightArmF2 = [(402, 481),None]

def organiseData(f1,f2,predict):
    data = [f1,f2]
    data = [[row[i] for row in data] for i in range(len(data[0]))] #transpose
    predict.createSpeedMatrix(data)

def predictNextCoordinate(ta, f2):
    currentFrame = f2
    for person in range(len(ta)):
        try:
            tempX = int(currentFrame[person][0])
            tempY = int(currentFrame[person][1])
            tempX += int(ta[person][0])
            tempY += int(ta[person][1])
            currentFrame[person] = (tempX,tempY)
        except:
            currentFrame[person] = None
    print(currentFrame)

organiseData(leftArmF1, leftArmF2, leftArmPredict)
organiseData(rightArmF1, rightArmF2, rightArmPredict)
_,rta = rightArmPredict.getSpeedMatrix()
_,lta = leftArmPredict.getSpeedMatrix()

predictNextCoordinate(lta, leftArmF2)
predictNextCoordinate(rta, rightArmF2)

state = True
while (state):
    predictNextCoordinate(lta, leftArmF2)
    state, lta = leftArmPredict.getSpeedMatrix()

headPredict = SpeedPrediction()

headF1 = [(615, 258),(574, 231),None]
headF2 = [(613, 258),(574, 231),None]
organiseData(headF1, headF2, headPredict)
_,hta = headPredict.getSpeedMatrix()
predictNextCoordinate(hta, headF2)

