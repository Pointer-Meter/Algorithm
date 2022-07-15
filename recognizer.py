from distutils.command.config import config
from cv2 import imread
from mmdet.apis import init_detector, inference_detector
from mmdet.core.mask.structures import bitmap_to_polygon
import mmcv
import os
import numpy as np
import cv2
from math import atan, sqrt, pi, sin, cos, degrees, radians
import json
import yaml
from myUtils import *

class CRecognizer:
    def __init__(self, config):
        self.m_config = config
        self.m_model = init_detector(
            config['MODEL_CONFIG_PATH'], 
            config['CHECKPOINT_PATH'], 
            device='cuda:0'
            )
        self.m_ScaleCorner = None

    def _detectCorner(self, vMask, vRawImg):
        voCornerImg = vRawImg.copy()
        Corners = cv2.goodFeaturesToTrack(
            vMask, 
            self.m_config['MAX_CORNERS'], 
            self.m_config['QUALITY_LEVEL'], 
            self.m_config['MIN_DISTANCE'], 
            blockSize = self.m_config['BLOCK_SIZE'])
        for p in Corners:
            voCornerImg = cv2.circle(
                voCornerImg, 
                (np.int32(p[0][0]), np.int32(p[0][1])), 
                10, (255, 255, 0), 2)
        return Corners, voCornerImg

    def _getRotateMatrix(self, vShape, vP1, vP2, vPr):
        k = (vP1[1] - vP2[1])*1.0 / (vP1[0] - vP2[0])
        b = vP1[1] - k*vP1[0]
        vP1 = np.array(vP1+[1]).reshape(-1, 1)
        vP2 = np.array(vP2+[1]).reshape(-1, 1)
        Angle = atan(k)*180/3.14
        (h, w) = vShape[:2]
        (cx, cy) = (w // 2, h // 2)

        if vPr[0]*k - vPr[1] + b <= 0:
            Angle += 180
        voMatrix = cv2.getRotationMatrix2D((cx, cy), Angle, 1.0)
        cos = np.abs(voMatrix[0, 0])
        sin = np.abs(voMatrix[0, 1])
        voW = int((h * sin) + (w * cos))
        voH = int((h * cos) + (w * sin))
        voMatrix[0, 2] += (voW / 2) - cx
        voMatrix[1, 2] += (voH / 2) - cy
        return voMatrix, voW, voH

    def _getPerspectiveMatrix(self, vImg, vMask):
        East = [0, 0]
        West = [0x7fffffff, 0x7fffffff]
        North = [0x7fffffff, 0x7fffffff]
        Contours, _ = bitmap_to_polygon(vMask)
        Contour = Contours[0]
        North = min(Contour, key = lambda i: i[1])
        East = max(Contour, key = lambda i: i[0])
        West = min(Contour, key = lambda i: i[0])
        MidH = (East[1]+West[1])/2
        East[1] = West[1] = MidH
        MidW = (self.m_ScaleCorner[2][0]+North[0]+East[0]+West[0])/4
        North[0] = MidW
        South = [MidW, East[1]+West[1]-North[1]]
        Points1 = np.float32([East, South, West, North])
        Points2 = np.float32([
            [800, 500], [500, 800], [200, 500], [500, 200]
            ])
        PerMatrix = cv2.getPerspectiveTransform(Points1, Points2)
        return PerMatrix


    def _rotate(self, vImg, vScaleCorner, vRefPoint):
        RotateMatrix, NewW, NewH = self._getRotateMatrix(
            vImg.shape,
            vScaleCorner[0],
            vScaleCorner[1],
            vRefPoint
            )
        voRotateImg = cv2.warpAffine(
            vImg, 
            RotateMatrix,
            (NewW, NewH),
            )
        voRotateMatrix = RotateMatrix
        return voRotateImg, voRotateMatrix

    def _perspective(self, vImg, vMask):
        PerMatrix = self._getPerspectiveMatrix(vImg, vMask)
        PerImg = cv2.warpPerspective(vImg, PerMatrix, (1500, 1500))
        voPerImg = PerImg
        return voPerImg

    def _adjust(self, vImgPath): # vImgPath 有待商榷
        InferenceResult = inference_detector(
            self.m_model, vImgPath)
        self.m_model.show_result(
            vImgPath,
            InferenceResult,
            out_file=config['INFERENCE_SAVE_PATH']+'/'+vImgPath
            )
        
        BboxResult, SegmResult = InferenceResult
        Segm = mmcv.concat_list(SegmResult)
        Segm = np.stack(Segm, axis=0)

        PointerMask, ScaleMask = Segm[0], Segm[1]
        PointerMask = PointerMask.astype(np.uint8)
        ScaleMask = ScaleMask.astype(np.uint8)

        ReferencePoint = [
            (BboxResult[0][0][0]+BboxResult[0][0][2])/2, 
            (BboxResult[0][0][1]+BboxResult[0][0][3])/2
            ]
        
        RawImg = cv2.imread(vImgPath)
        CornerLis, CornerImg = self._detectCorner(ScaleMask, RawImg) 
        ScaleCorner = [[CornerLis[0][0][0], CornerLis[0][0][1]],
            [CornerLis[1][0][0], CornerLis[1][0][1]]]
        RotateImg, RotateMatrix = self._rotate(CornerImg, ScaleCorner, ReferencePoint)
        RotateMask, _ = self._rotate(ScaleMask, ScaleCorner, ReferencePoint)
        ScaleCorner[0] = np.array(ScaleCorner[0]+[1]).reshape(-1,1)
        ScaleCorner[1] = np.array(ScaleCorner[1]+[1]).reshape(-1,1)
        ScaleCorner[0] = np.dot(RotateMatrix, ScaleCorner[0])
        ScaleCorner[1] = np.dot(RotateMatrix, ScaleCorner[1])
        self.m_ScaleCorner = [
            ScaleCorner[0], 
            ScaleCorner[1], 
            [(ScaleCorner[0][0]+ScaleCorner[1][0])/2, (ScaleCorner[0][1]+ScaleCorner[1][1])/2]
            ]
        PerImg = self._perspective(RotateImg, RotateMask)
        cv2.imwrite(self.m_config['ADJUST_PATH']+'/'+vImgPath, PerImg) 

    def _fitCircle(self):
        pass

    def process(self, vDataPath):
        self._adjust(vDataPath)

        
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    cr = CRecognizer(config)
    cr.process('HD213.jpg')