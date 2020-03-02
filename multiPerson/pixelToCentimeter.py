import sys

sys.path.append('.')
import cv2
import numpy as np
import torch

from lib.config import cfg
from evaluate.coco_eval import get_outputs
from lib.utils.common import CocoPart, CocoColors, CocoPairsRender
from lib.utils.paf_to_pose import paf_to_pose_cpp



class pixelToCentimeter:

    def __init__(self, videoPath, model):
        self.videoPath = videoPath
        self.model = model

    def getHeight(self):
        frames = 0
        video = cv2.VideoCapture(self.videoPath)
        self.height = []
        while video.isOpened() and frames <= 50:
            frames += 1
            ret, oriImg = video.read()
            oriImg = cv2.resize(oriImg, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            with torch.no_grad():
                paf, heatmap, imscale = get_outputs(
                    oriImg, self.model, 'rtpose')

            humans = paf_to_pose_cpp(heatmap, paf, cfg)

            out = self.draw_humans(oriImg, humans)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 0, 200)
            thickness = 2
            out = cv2.putText(out, 'Calculating height', org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('Video', out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
        averageHeight = 170
        self.height.sort()
        a = np.array(self.height)
        lq = np.percentile(a, 70)
        answer = averageHeight / lq
        return answer

    def draw_humans(self, npimg, humans):
        image_h, image_w = npimg.shape[:2]
        centers = {}
        tempHeight = []
        for human in humans:
            # draw point
            for i in range(CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
            tempHeight.append(self.meanHeightForSingleFrame(human.body_parts, centers))
            # draw line
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
        tempHeight = [i for i in tempHeight if i]
        tempHeight = np.mean(tempHeight)
        self.height.append(tempHeight)
        return npimg

    def meanHeightForSingleFrame(self, bodyParts, points):
        identifiedParts = {}
        for i in range(CocoPart.Background.value):
            if i not in bodyParts.keys():
                continue
            identifiedParts[bodyParts[i].part_idx] = points[i]
        head = self.getCoordinatePerBodypart(identifiedParts, [0, 1, 14, 15, 16, 17])  # head
        foot = self.getCoordinatePerBodypart(identifiedParts, [13,10])  # Lfoot
        if (head != None and foot != None):
            return abs(head[1] - foot[1])
        else:
            return None

    def getCoordinatePerBodypart(self, identifiedParts, ids):
        for body in ids:
            if body in identifiedParts:
                return identifiedParts.get(body)
        return None