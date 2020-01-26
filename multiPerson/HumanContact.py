
class HumanContact:

    def __init__(self, everyHumanCoordinate, offendingLimb):
        self.everyHumanCoordinate = everyHumanCoordinate
        self.offendingLimb = offendingLimb

    def distance(self,x, coordinate):
        try:
            x_value = abs(x[0] - coordinate[0])
            y_value = abs(x[1] - coordinate[1])
            return x_value + y_value
        except:
            return None

    def getCoord(self,bodyCode, humanID):
        if bodyCode == 0:
            return self.everyHumanCoordinate[humanID][0]
        elif bodyCode == 1:
            return self.everyHumanCoordinate[humanID][3]
        elif bodyCode == 2:
            return self.everyHumanCoordinate[humanID][5]
        elif bodyCode == 3:
            return self.everyHumanCoordinate[humanID][7]
        elif bodyCode == 4:
            return self.everyHumanCoordinate[humanID][9]
        elif bodyCode == 5:
            return self.everyHumanCoordinate[humanID][11]
        elif bodyCode == 6:
            return self.everyHumanCoordinate[humanID][13]

    def isThereACollision(self):
        distanceValue = []
        noOfHumans = len(self.everyHumanCoordinate)
        for offendingPerson in range(noOfHumans):
            victimID = self.targetPerson(noOfHumans, offendingPerson)
            for id in victimID:
                sel = self.getCoord(self.offendingLimb, offendingPerson)
                if (id != offendingPerson):
                    for indCoor in self.everyHumanCoordinate[id]:
                        distanceValue.append(self.distance(sel, indCoor))
                #print(distanceValue)
                try:
                    minValue = min(i for i in distanceValue if i is not None)
                    minIndex = distanceValue.index(minValue)
                    if minValue < 90:
                        return True, minIndex
                except:
                    pass
                distanceValue.clear()
        return False, None

    def targetPerson(self,n, rem):
        x = list(range(n))
        x.remove(rem)
        return x

    def getBodyPart(self,bodyId):
        if bodyId == 0 or bodyId == 1:
            return "head"
        elif bodyId == 2 or bodyId == 3:
            return "left torso"
        elif bodyId == 4 or bodyId == 5:
            return "right torso"
        elif bodyId == 6 or bodyId == 7:
            return "left arm"
        elif bodyId == 8 or bodyId == 9:
            return "right arm"
        elif bodyId == 10 or bodyId == 11:
            return "left foot"
        elif bodyId == 12 or bodyId == 13:
            return "right foot"



