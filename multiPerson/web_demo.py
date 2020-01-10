import sys
from tkinter import Label

sys.path.append('.')
import cv2
import argparse
import numpy as np
import torch
from lib.network.rtpose_vgg import get_model
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs
from lib.utils.common import DrawHuman
from lib.utils.paf_to_pose import paf_to_pose_cpp

class GUIStats:
    def __init__(self, numberOfPeople):
        self.numberOfPeople = numberOfPeople

    def gui(self):
        self.window.title("Movement Stats")
        self.window.minsize(700, 700)
        for x in range(0,self.numberOfPeople):
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

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)   

model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model.cuda()
model.float()
model.eval()

if __name__ == "__main__":
    evenFrame = True;
    video_capture = cv2.VideoCapture('assault1.mp4')
    hd = DrawHuman()
    while True:
        evenFrame = not evenFrame
        ret, oriImg = video_capture.read()
        
        shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')
                  
        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        #print("There are "+ str(len(humans)) + " humans in the video")
        out = hd.draw_humans(oriImg, humans, evenFrame)
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if (evenFrame):
            hd.calculateSpeedForIndLimbs()
            listOfSpeed = hd.getSpeed()
            print("This is how much pixel has changed "+str(listOfSpeed))
            hd.clearSpeedData()
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
