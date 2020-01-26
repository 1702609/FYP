import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
from HumanContact import HumanContact

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from lib.config import update_config, cfg
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans, HumanCoordinate
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

if __name__ == "__main__":
    
    image_capture = cv2.imread("contact8.jpg")

    shape_dst = np.min(image_capture.shape[0:2])

    with torch.no_grad():
        paf, heatmap, imscale = get_outputs(
            image_capture, model, 'rtpose')

    humans = paf_to_pose_cpp(heatmap, paf, cfg)

    out = draw_humans(image_capture, humans)
    huCoord = HumanCoordinate(image_capture, humans)
    result = huCoord.collectCoordinate()
    #print(result)
    #0 Head, 1 Left Torso, 2 Right Torso, 3 left arm, 4 right arm, 5 left foot, 6 right foot

    # will only execute if there is a high speed movement,
    # detected high speed right foot movement
    detectCol = HumanContact(result, 6)
    #0 Head, 1 Left Torso, 2 Right Torso, 3 left arm, 4 right arm, 5 left foot, 6 right foot

    collision, minIndex = detectCol.isThereACollision()
    if (collision):
        bodyPartVictim = detectCol.getBodyPart(minIndex)
        print("Victim has been attacked in the " + bodyPartVictim)
    cv2.imshow('Image', out)
    cv2.waitKey(0)

    #cv2.destroyAllWindows()
