import sys
from tkinter import Label

from GUIForStats import GUIForStats
from pixelToCentimeter import pixelToCentimeter

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


def maximumNumberOfHuman():
    humanLength = []
    for i in range(5): #it will sample 5 frames to find maximum number of people in the video
        _, oriImg = video_capture.read()
        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')

        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        humanLength.append(len(humans))
    humanLength.sort()
    return humanLength[-1]

def determineHumanSizeInVideo():
    print("Calculating the pixels required to cover a person's height...")
    cmPerPixelObject = pixelToCentimeter(videoPath)
    return cmPerPixelObject.calculateCentimeterPerPixel()

if __name__ == "__main__":
    videoPath = 'dataset/Assault/assault4.mp4'

    evenFrame = True;
    video_capture = cv2.VideoCapture(videoPath)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    numberOfHumans = maximumNumberOfHuman()
    cmPerPixel = determineHumanSizeInVideo()
    print("1 pixel represents "+str(cmPerPixel))
    hd = DrawHuman(numberOfHumans)
    gui = GUIForStats(numberOfHumans,cmPerPixel,fps)
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
            gui.drawGUI(listOfSpeed)
            hd.clearSpeedData()
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
