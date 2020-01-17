import math
import cv2
from enum import Enum

class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    @staticmethod
    def _get_uidx(part_idx, idx):
        return '%d-%d' % (part_idx, idx)

    def add_pair(self, pair):
        self.pairs.append(pair)
        self.body_parts[pair.part_idx1] = BodyPart(Human._get_uidx(pair.part_idx1, pair.idx1),
                                                   pair.part_idx1,
                                                   pair.coord1[0], pair.coord1[1], pair.score)
        self.body_parts[pair.part_idx2] = BodyPart(Human._get_uidx(pair.part_idx2, pair.idx2),
                                                   pair.part_idx2,
                                                   pair.coord2[0], pair.coord2[1], pair.score)
        self.uidx_list.add(Human._get_uidx(pair.part_idx1, pair.idx1))
        self.uidx_list.add(Human._get_uidx(pair.part_idx2, pair.idx2))

    def is_connected(self, other):
        return len(self.uidx_list & other.uidx_list) > 0

    def merge(self, other):
        for pair in other.pairs:
            self.add_pair(pair)

    def part_count(self):
        return len(self.body_parts.keys())

    def get_max_score(self):
        return max([x.score for _, x in self.body_parts.items()])

    def get_face_box(self, img_w, img_h, mode=0):
        """
        Get Face box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :param mode:
        :return:
        """
        # SEE : https://github.com/ildoonet/tf-pose-estimation/blob/master/tf_pose/common.py#L13
        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _REye = CocoPart.REye.value
        _LEye = CocoPart.LEye.value
        _REar = CocoPart.REar.value
        _LEar = CocoPart.LEar.value

        _THRESHOLD_PART_CONFIDENCE = 0.2
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        is_nose, part_nose = _include_part(parts, _NOSE)
        if not is_nose:
            return None

        size = 0
        is_neck, part_neck = _include_part(parts, _NECK)
        if is_neck:
            size = max(size, img_h * (part_neck.y - part_nose.y) * 0.8)
        is_reye, part_reye = _include_part(parts, _REye)
        is_leye, part_leye = _include_part(parts, _LEye)
        if is_reye and is_leye:
            size = max(size, img_w * (part_reye.x - part_leye.x) * 2.0)
            size = max(size,
                       img_w * math.sqrt((part_reye.x - part_leye.x) ** 2 + (part_reye.y - part_leye.y) ** 2) * 2.0)

        if mode == 1:
            if not is_reye and not is_leye:
                return None

        is_rear, part_rear = _include_part(parts, _REar)
        is_lear, part_lear = _include_part(parts, _LEar)
        if is_rear and is_lear:
            size = max(size, img_w * (part_rear.x - part_lear.x) * 1.6)

        if size <= 0:
            return None

        if not is_reye and is_leye:
            x = part_nose.x * img_w - (size // 3 * 2)
        elif is_reye and not is_leye:
            x = part_nose.x * img_w - (size // 3)
        else:  # is_reye and is_leye:
            x = part_nose.x * img_w - size // 2

        x2 = x + size
        if mode == 0:
            y = part_nose.y * img_h - size // 3
        else:
            y = part_nose.y * img_h - _round(size / 2 * 1.2)
        y2 = y + size

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        if mode == 0:
            return {"x": _round((x + x2) / 2),
                    "y": _round((y + y2) / 2),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}
        else:
            return {"x": _round(x),
                    "y": _round(y),
                    "w": _round(x2 - x),
                    "h": _round(y2 - y)}

    def get_upper_body_box(self, img_w, img_h):
        """
        Get Upper body box compared to img size (w, h)
        :param img_w:
        :param img_h:
        :return:
        """
        if not (img_w > 0 and img_h > 0):
            raise Exception("img size should be positive")

        _NOSE = CocoPart.Nose.value
        _NECK = CocoPart.Neck.value
        _RSHOULDER = CocoPart.RShoulder.value
        _LSHOULDER = CocoPart.LShoulder.value
        _THRESHOLD_PART_CONFIDENCE = 0.3
        parts = [part for idx, part in self.body_parts.items() if part.score > _THRESHOLD_PART_CONFIDENCE]
        part_coords = [(img_w * part.x, img_h * part.y) for part in parts if
                       part.part_idx in [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]]

        if len(part_coords) < 5:
            return None

        # Initial Bounding Box
        x = min([part[0] for part in part_coords])
        y = min([part[1] for part in part_coords])
        x2 = max([part[0] for part in part_coords])
        y2 = max([part[1] for part in part_coords])

        # # ------ Adjust heuristically +
        # if face points are detcted, adjust y value

        is_nose, part_nose = _include_part(parts, _NOSE)
        is_neck, part_neck = _include_part(parts, _NECK)
        torso_height = 0
        if is_nose and is_neck:
            y -= (part_neck.y * img_h - y) * 0.8
            torso_height = max(0, (part_neck.y - part_nose.y) * img_h * 2.5)
        #
        # # by using shoulder position, adjust width
        is_rshoulder, part_rshoulder = _include_part(parts, _RSHOULDER)
        is_lshoulder, part_lshoulder = _include_part(parts, _LSHOULDER)
        if is_rshoulder and is_lshoulder:
            half_w = x2 - x
            dx = half_w * 0.15
            x -= dx
            x2 += dx
        elif is_neck:
            if is_lshoulder and not is_rshoulder:
                half_w = abs(part_lshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)
            elif not is_lshoulder and is_rshoulder:
                half_w = abs(part_rshoulder.x - part_neck.x) * img_w * 1.15
                x = min(part_neck.x * img_w - half_w, x)
                x2 = max(part_neck.x * img_w + half_w, x2)

        # ------ Adjust heuristically -

        # fit into the image frame
        x = max(0, x)
        y = max(0, y)
        x2 = min(img_w - x, x2 - x) + x
        y2 = min(img_h - y, y2 - y) + y

        if _round(x2 - x) == 0.0 or _round(y2 - y) == 0.0:
            return None
        return {"x": _round((x + x2) / 2),
                "y": _round((y + y2) / 2),
                "w": _round(x2 - x),
                "h": _round(y2 - y)}

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

class DrawHuman:
    def __init__(self, numberOfHumans):
        self.numberOfHumans = numberOfHumans
        self.nose_pixel_change = []
        self.leftHand_pixel_change = []
        self.rightHand_pixel_change = []
        self.leftFoot_pixel_change = []
        self.rightFoot_pixel_change = []

        self.noseFrameOne = []
        self.noseFrameTwo = []
        self.tempNoseFrameTwo = []

        self.leftHandFrameOne = []
        self.leftHandFrameTwo = []
        self.tempLeftHandFrameTwo = []

        self.rightHandFrameOne = []
        self.rightHandFrameTwo = []
        self.tempRightHandFrameTwo = []

        self.leftFootFrameOne = []
        self.leftFootFrameTwo = []
        self.tempLeftFootFrameTwo = []

        self.rightFootFrameOne = []
        self.rightFootFrameTwo = []
        self.tempRightFootFrameTwo = []

    def draw_humans(self, npimg, humans, frameEven):
        self.npimg = npimg
        self.humans = humans
        self.frameEven = frameEven
        image_h, image_w = self.npimg.shape[:2]
        centers = {}
        for human in self.humans:
            # draw point
            for i in range(CocoPart.Background.value):
                if i not in human.body_parts.keys():
                    continue
                body_part = human.body_parts[i]
                center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
                centers[i] = center
                cv2.circle(self.npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)
                self.analyzeBodyParts(body_part.part_idx, center)
            # draw line
            for pair_order, pair in enumerate(CocoPairsRender):
                if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                    continue
                cv2.line(self.npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)
        return self.npimg

    def analyzeBodyParts(self, bodyPart, points):
        if (bodyPart == 0):
            self.analyzeLimbs(points, self.noseFrameOne, self.noseFrameTwo, "Head")
        if (bodyPart == 13):
            self.analyzeLimbs(points, self.leftFootFrameOne, self.leftFootFrameTwo, "LFoot")
        if (bodyPart == 10):
            self.analyzeLimbs(points, self.rightFootFrameOne, self.rightFootFrameTwo, "RFoot")
        if (bodyPart == 7):
            self.analyzeLimbs(points, self.leftHandFrameOne, self.leftHandFrameTwo, "LHand")
        if (bodyPart == 4):
            self.analyzeLimbs(points, self.rightHandFrameOne, self.rightHandFrameTwo, "RHand")

    def analyzeLimbs(self, points, frameOne, frameTwo, limbName):
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (points)
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1
            cv2.putText(self.npimg, limbName, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
            if (self.frameEven):
                frameTwo.append(points)
            else:
                frameOne.append(points)
        except:
            if (self.frameEven):
                frameTwo.append(None)
            else:
                frameOne.append(None)

    def calculateSpeedForIndLimbs(self):
        self.pixelChangeCalculator(self.noseFrameOne, self.noseFrameTwo, self.nose_pixel_change)
        self.pixelChangeCalculator(self.leftHandFrameOne, self.leftHandFrameTwo, self.leftHand_pixel_change)
        self.pixelChangeCalculator(self.rightHandFrameOne, self.rightHandFrameTwo, self.rightHand_pixel_change)
        self.pixelChangeCalculator(self.leftFootFrameOne, self.leftFootFrameTwo, self.leftFoot_pixel_change)
        self.pixelChangeCalculator(self.rightFootFrameOne, self.rightFootFrameTwo, self.rightFoot_pixel_change)

    def transferFrameTwoToTemp(self):
        self.updateArrays(self.noseFrameOne, self.noseFrameTwo, self.tempNoseFrameTwo)
        self.updateArrays(self.leftHandFrameOne, self.leftHandFrameTwo, self.tempLeftHandFrameTwo)
        self.updateArrays(self.rightHandFrameOne, self.rightHandFrameTwo, self.tempRightHandFrameTwo)
        self.updateArrays(self.leftFootFrameOne, self.leftFootFrameTwo, self.tempLeftFootFrameTwo)
        self.updateArrays(self.rightFootFrameOne, self.rightFootFrameTwo, self.tempRightFootFrameTwo)


    def updateArrays(self, frameOne, frameTwo, frameTwoTemp):
        frameOne.clear()
        for i in range(len(frameTwo)):
            frameTwoTemp.append(frameTwo[i])
        frameTwo.clear()

    def pixelChangeCalculator(self, frameOne, frameTwo, pixelChange):
        for i in range(self.numberOfHumans):
            try:
                frame1 = frameOne[i]
                frame2 = frameTwo[i]
                pixelChange.append(self.pixelChangeAlgorithm(frame1, frame2))
            except:
                pixelChange.append(None)

    def pixelChangeAlgorithm(self, tupleFrame1, tupleFrame2):
        firstFrame = tupleFrame1
        secondFrame = tupleFrame2
        changeInX = abs(firstFrame[0] - secondFrame[0]) ** 2
        changeInY = abs(firstFrame[1] - secondFrame[1]) ** 2
        moveDistance = math.sqrt(changeInX + changeInY)
        moveDistance = round(moveDistance, 2)
        return moveDistance

    def getSpeed(self):
        list_of_speed = [self.nose_pixel_change, self.leftHand_pixel_change, self.rightHand_pixel_change,
                         self.leftFoot_pixel_change, self.rightFoot_pixel_change]
        return list_of_speed

    def clearSpeedData(self):
        self.nose_pixel_change = []
        self.leftHand_pixel_change = []
        self.rightHand_pixel_change = []
        self.leftFoot_pixel_change = []
        self.rightFoot_pixel_change = []

    def distance(self, x, coordinate):
        x_value = abs(x[0] - coordinate[0])
        y_value = abs(x[1] - coordinate[1])
        return x_value + y_value

    def syncFrameOneWithFrameTwo(self):
        self.trackingAdjacentFrame(self.noseFrameOne, self.noseFrameTwo)
        self.trackingAdjacentFrame(self.leftHandFrameOne, self.leftHandFrameTwo)
        self.trackingAdjacentFrame(self.rightHandFrameOne, self.rightHandFrameTwo)
        self.trackingAdjacentFrame(self.leftFootFrameOne, self.leftFootFrameTwo)
        self.trackingAdjacentFrame(self.rightFootFrameOne, self.rightFootFrameTwo)

    def trackingAdjacentFrame(self, frameOne, frameTwo):
        newFrame = []
        for i in range(len(frameTwo)):
            try:
                newFrame.append(min(frameTwo, key=lambda x: self.distance(x, frameOne[i])))
            except:
                newFrame.append(None)
        frameTwo.clear()
        for i in range(len(newFrame)):
            frameTwo.append(newFrame[i])

    def syncHumanFromPreviousFrame(self):
        self.fixTracking(self.noseFrameOne, self.tempNoseFrameTwo)
        self.fixTracking(self.leftHandFrameOne, self.tempLeftHandFrameTwo)
        self.fixTracking(self.rightHandFrameOne, self.tempRightHandFrameTwo)
        self.fixTracking(self.leftFootFrameOne, self.tempLeftFootFrameTwo)
        self.fixTracking(self.rightFootFrameOne, self.tempRightFootFrameTwo)

    def fixTracking(self, frameOne, tempFrame):
        if tempFrame:
            newFrame = []
            for i in range(len(frameOne)):
                try:
                    newFrame.append(min(frameOne, key=lambda x: self.distance(x, tempFrame[i])))
                except:
                    newFrame.append(None)
            frameOne.clear()
            for i in range(len(newFrame)):
                frameOne.append(newFrame[i])
            tempFrame.clear()

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]  # = 19
CocoPairsRender = CocoPairs[:-2]
