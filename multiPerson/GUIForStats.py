from tkinter import *


class GUIForStats:

    def __init__(self, cmPerPixel, fps):
        self.cmPerPixel = cmPerPixel
        self.root = Tk()
        self.root.geometry("400x500")
        perSecond = (1 / fps) * 2
        self.timeNeeded = 1 / perSecond
        self.speedThreshold = False
        self.offendingLimb = 0

    def drawGUI(self, speedList, numberOfPeople):
        self.clearGUI()
        self.speedThreshold = False
        fcol = 0
        frow = 0
        for i in range(numberOfPeople):
            op_frame = Frame(self.root, bg='white', width=150, height=150).grid(row=frow, rowspan=6, columnspan=2,
                                                                                column=fcol)
            PersonID = Label(op_frame, text="Person" + str(i))
            PersonID.grid(column=0 + fcol, row=0 + frow)

            head = Label(op_frame, text="head")
            head.grid(column=0 + fcol, row=1 + frow)
            headSpeed = Label(op_frame, text="Detecting")
            headSpeed.grid(column=1 + fcol, row=1 + frow)
            self.updateGUI(speedList[0][i], headSpeed, 0)

            leftHand = Label(op_frame, text="left hand")
            leftHand.grid(column=0 + fcol, row=2 + frow)
            leftHandSpeed = Label(op_frame, text="Detecting")
            leftHandSpeed.grid(column=1 + fcol, row=2 + frow)
            self.updateGUI(speedList[1][i], leftHandSpeed, 3)

            rightHand = Label(op_frame, text="right hand")
            rightHand.grid(column=0 + fcol, row=3 + frow)
            rightHandSpeed = Label(op_frame, text="Detecting")
            rightHandSpeed.grid(column=1 + fcol, row=3 + frow)
            self.updateGUI(speedList[2][i], rightHandSpeed, 4)

            leftFoot = Label(op_frame, text="left foot")
            leftFoot.grid(column=0 + fcol, row=4 + frow)
            leftFootSpeed = Label(op_frame, text="Detecting")
            leftFootSpeed.grid(column=1 + fcol, row=4 + frow)
            self.updateGUI(speedList[3][i], leftFootSpeed, 5)

            rightFoot = Label(op_frame, text="right foot")
            rightFoot.grid(column=0 + fcol, row=5 + frow)
            rightFootSpeed = Label(op_frame, text="Detecting")
            rightFootSpeed.grid(column=1 + fcol, row=5 + frow)
            self.updateGUI(speedList[4][i], rightFootSpeed, 6)

            if (fcol == 2):
                frow += 6
                fcol = 0
            else:
                fcol += 2
        self.root.update()

    def updateGUI(self, data, target, limb):
        if data is not None:
            speed = data * self.cmPerPixel * self.timeNeeded
            if (speed >= 200):
                target.config(text="Calculating.....")
            else:
                target.config(text=str(round(speed, 2)) + " cm/s")
                if (speed >= 100):
                    target.config(bg="red")
                    self.speedThreshold = True
                    self.offendingLimb = limb
        else:
            target.config(text=str(data))

    def clearGUI(self):
        _list = self.root.winfo_children()

        for item in _list:
            if item.winfo_children():
                _list.extend(item.winfo_children())
        for item in _list:
            item.destroy()

    def dangerousSpeed(self):
        return self.speedThreshold, self.offendingLimb