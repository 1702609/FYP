import sys

from MotionAnalysis import DrawHuman
from GUIForStats import GUIForStats
from HumanContact import HumanContact
from pixelToCentimeter import pixelToCentimeter

sys.path.append('.')
import cv2
import argparse
import numpy as np
import torch
from lib.network.rtpose_vgg import get_model
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs
from lib.utils.common import HumanCoordinate
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

def determineHumanSizeInVideo():
    print("Calculating the pixels required to cover a person's height...")
    cmPerPixelObject = pixelToCentimeter(videoPath,model)
    return cmPerPixelObject.getHeight()


def checkForContact(img, offendingLimb, cmPerPixel):
    huCoord = HumanCoordinate(img, humans)
    result = huCoord.collectCoordinate()
    #print(result)
    listParts = ["head","LT","RT","left arm","right arm","left foot","right foot"]
    #0 Head, 1 Left Torso, 2 Right Torso, 3 left arm, 4 right arm, 5 left foot, 6 right foot

    # will only execute if there is a high speed movement,
    # detected high speed right foot movement
    detectCol = HumanContact(result, offendingLimb, cmPerPixel)
    collision, minIndex = detectCol.isThereACollision()
    if (collision):
        bodyPartVictim = detectCol.getBodyPart(minIndex)
        print("The offender has used the "+listParts[offendingLimb]+" to attack")
        print("Victim has been attacked in the " + bodyPartVictim+'\n')

if __name__ == "__main__":
    videoPath = 'dataset/Assault/assault10.mp4'

    evenFrame = True;
    video_capture = cv2.VideoCapture(videoPath)
    #fps = video_capture.get(cv2.CAP_PROP_FPS)
    fps = 8
    try:
        cmPerPixel = determineHumanSizeInVideo()
    except:
        cmPerPixel = 0.816
    print("1 pixel represents " + str(cmPerPixel) + " cm")
    debugMode = False
    hd = DrawHuman(debugMode)
    gui = GUIForStats(cmPerPixel, fps)
    while video_capture.isOpened():
        evenFrame = not evenFrame
        ret, oriImg = video_capture.read()

        shape_dst = np.min(oriImg.shape[0:2])

        with torch.no_grad():
            paf, heatmap, imscale = get_outputs(
                oriImg, model, 'rtpose')

        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        # print("There are "+ str(len(humans)) + " humans in the video")
        out = hd.draw_humans(oriImg, humans, evenFrame)
        cv2.imshow('Video', out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (not evenFrame):
            hd.syncFrameOneWithFrameTwo()
            hd.fixMissingCoordinate(evenFrame)
        if (evenFrame):
            hd.syncFrameTwoWithFrameOne()
            hd.updatePredictEngine()
            hd.fixMissingCoordinate(evenFrame)
            if debugMode:
                hd.writeDataToCSV()
            hd.calculateSpeedForIndLimbs()
            listOfSpeed = hd.getSpeed()
            gui.drawGUI(listOfSpeed,len(humans))
            hd.clearSpeedData()
            hd.transferFrameTwoToTemp()
            danger,offendingLimb = gui.dangerousSpeed()
            if(danger):
                checkForContact(oriImg,offendingLimb,cmPerPixel)
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
