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

class CClockRecognizer:
    def __init__(self, vConfig):
        self.m_Config = vConfig
        self.m_model = init_detector(
            vConfig['MODEL_CONFIG_PATH'], 
            vConfig['CHECKPOINT_PATH'], 
            device='cuda:0'
            )
        self.m_ScaleCorner = None
        self.m_ScaleCircle = None
        self.m_RotateImg = None
        self.m_AdjustScaleMask = None
        self.m_AdjustPointerMask = None

    def _detectCorner(self, vMask, vRawImg):
        CornerImg = vRawImg.copy()
        Corners = cv2.goodFeaturesToTrack(
            vMask, 
            self.m_Config['MAX_CORNERS'], 
            self.m_Config['QUALITY_LEVEL'], 
            self.m_Config['MIN_DISTANCE'], 
            blockSize = self.m_Config['BLOCK_SIZE'])
        for p in Corners:
            CornerImg = cv2.circle(
                CornerImg, 
                (np.int32(p[0][0]), np.int32(p[0][1])), 
                10, (255, 255, 0), 2)
        return Corners, CornerImg

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
        Matrix = cv2.getRotationMatrix2D((cx, cy), Angle, 1.0)
        cos = np.abs(Matrix[0, 0])
        sin = np.abs(Matrix[0, 1])
        W = int((h * sin) + (w * cos))
        H = int((h * cos) + (w * sin))
        Matrix[0, 2] += (W / 2) - cx
        Matrix[1, 2] += (H / 2) - cy
        return Matrix, W, H

    def _getPerspectiveMatrix(self, vMask):
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
        MidW = (North[0]+East[0]+West[0])/3
        North[0] = MidW
        South = [MidW, East[1]+West[1]-North[1]]
        Points1 = np.float32([East, South, West, North])
        Points2 = np.float32([
            self.m_Config['REC']['EAST'],
            self.m_Config['REC']['SOUTH'], 
            self.m_Config['REC']['WEST'],
            self.m_Config['REC']['NORTH']
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
        RotateImg = cv2.warpAffine(
            vImg, 
            RotateMatrix,
            (NewW, NewH),
            )
        return RotateImg, RotateMatrix, NewW, NewH

    def _perspective(self, vImg, vScaleMask, vPointerMask, vScaleCorner):
        PerMatrix = self._getPerspectiveMatrix(vScaleMask)
        PerImg = cv2.warpPerspective(vImg, PerMatrix, tuple(self.m_Config['REC']['SIZE']))
        ScaleMask = cv2.warpPerspective(
            vScaleMask, PerMatrix,
            tuple(self.m_Config['REC']['SIZE'])
        )
        PointerMask = cv2.warpPerspective(
            vPointerMask, PerMatrix,
            tuple(self.m_Config['REC']['SIZE'])
        )
        ConcatMask = ScaleMask + PointerMask
        ConcatMask[ConcatMask>0] = 255
        cv2.imwrite(self.m_Config['MASK_SAVE_PATH']+'/'+ 'Per_' + "jj.jpg", ConcatMask)

        ScaleCorner = vScaleCorner
        ScaleCorner = np.dot(PerMatrix, ScaleCorner)
        return PerImg, ScaleMask, PointerMask, ScaleCorner

    def _adjust(self, vImgPath): # vImgPath 是路径而不是单纯文件名
        print(">> adjusting " + vImgPath + ' ...')
        InferenceResult = inference_detector(
            self.m_model, vImgPath)
        self.m_model.show_result(
            vImgPath,
            InferenceResult,
            out_file=config['INFERENCE_SAVE_PATH']+'/'+ getNameFromPath(vImgPath)
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
        RotateImg, RotateMatrix, NewW, NewH = self._rotate(CornerImg, ScaleCorner, ReferencePoint)
        RotateScaleMask = cv2.warpAffine(ScaleMask, RotateMatrix, (NewW, NewH))
        RotatePointerMask = cv2.warpAffine(PointerMask, RotateMatrix, (NewW, NewH))
        
        ConcatMask = RotateScaleMask + RotatePointerMask
        ConcatMask[ConcatMask>0] = 255
        cv2.imwrite(self.m_Config['MASK_SAVE_PATH']+'/'+ 'Rotate_' + getNameFromPath(vImgPath), ConcatMask)

        self.m_RotateImg = RotateImg

        print("ScaleCorner: ", ScaleCorner)
        print("RotateMatrix: ", RotateMatrix)
        onev, oneh = np.array([[0, 0, 1]]), np.array([[1], [1]])
        RotateMatrix = np.vstack((RotateMatrix, onev))
        ScaleCorner = np.hstack((ScaleCorner, oneh)).T
        ScaleCorner = np.dot(RotateMatrix, ScaleCorner)
        print("RotateMatrix: ", RotateMatrix)
        print("ScaleCorner: ", ScaleCorner)


        self.m_ScaleCorner = [
            (int(ScaleCorner[0][0]), int(ScaleCorner[1][0])), 
            (int(ScaleCorner[1][0]), int(ScaleCorner[1][1])), 
            [(ScaleCorner[0][0]+ScaleCorner[1][0])/2, (ScaleCorner[0][1]+ScaleCorner[1][1])/2]
            ]
        RotateImg = cv2.circle(
            RotateImg,
            self.m_ScaleCorner[0],
            10,
            self.m_Config['RGBCOLOR_BLUE'],
            5
        )
        RotateImg = cv2.circle(
            RotateImg,
            self.m_ScaleCorner[1],
            10,
            self.m_Config['RGBCOLOR_BLUE'],
            5
        )
        PerImg, self.m_AdjustScaleMask, self.m_AdjustPointerMask, ScaleCorner = self._perspective(
            RotateImg, RotateScaleMask, RotatePointerMask, ScaleCorner)
        self.m_ScaleCorner = [
            ScaleCorner[0], 
            ScaleCorner[1], 
            [(ScaleCorner[0][0]+ScaleCorner[1][0])/2, (ScaleCorner[0][1]+ScaleCorner[1][1])/2]
            ]
        print("self.Corner: ", self.m_ScaleCorner)
        cv2.imwrite(self.m_Config['ADJUST_SAVE_PATH']+'/'+getNameFromPath(vImgPath), PerImg) 

    def _fitCircle(self, vImgPath):
        print(">> fitting " + vImgPath + ' ...')
        Contours = bitmap_to_polygon(self.m_AdjustScaleMask)
        Contour = Contours[0]
        # showImg(self.m_AdjustScaleMask, vIfMask=True)
        ImgPath = self.m_Config['ADJUST_SAVE_PATH']+'/'+getNameFromPath(vImgPath)
        Img = cv2.imread(ImgPath)
        ColoredImg = Img.copy()
        # 轮廓分段法取点采样---有待改进
        CircleLis1 = []
        CircleLis2 = []
        i = 0
        cnt = 0
        Lock = False
        while True:
            # Contour: [array[[x,y],[x,y],[x,y]...]]
            [x, y] = Contour[0][i]
            i = (i + 19) % len(Contour[0]) # 轮廓点太多，19个点取1个
            add = False
            d1 = calPointDistance(
                x, y,
                self.m_ScaleCorner[0][0], self.m_ScaleCorner[0][1]
                )
            d2 = calPointDistance(
                x, y,
                self.m_ScaleCorner[1][0], self.m_ScaleCorner[1][1]
                )
            # print(i, " d1=", d1, " d2=", d2, " len1=", len(CircleLis1), " len2=", len(CircleLis2), " cnt=", cnt)
            if d1 <= self.m_Config['SAMPLE_DISTANCE_TO_CORNER'] \
                or d2 <= self.m_Config['SAMPLE_DISTANCE_TO_CORNER']:
                if not Lock:
                    Lock = True
                    cnt += 1
                else:
                    print("skip lock")
                    continue
            
            Lock = False
            if cnt % 2 == 0:
                CircleLis1.append([x, y])
            else:
                CircleLis2.append([x, y])
            ColoredImg = cv2.circle(
                ColoredImg, (np.int32(x), np.int32(y)), 10, 
                tuple(self.m_Config['RGBCOLOR_YELLOW']), 5)
            
            if len(CircleLis1) > 19 and len(CircleLis2) > 19:
                break
        A1, B1, R1 = calCircleCenter(CircleLis1)
        A2, B2, R2 = calCircleCenter(CircleLis2)
        self.m_ScaleCircle = [(A1+A2)/2,(B1+B2)/2,(0.8*R1+0.2*R2)]
        ColoredImg = cv2.circle(
            ColoredImg, 
            (np.int32(self.m_ScaleCircle[0]), np.int32(self.m_ScaleCircle[1])),
            np.int32(self.m_ScaleCircle[2]),
            self.m_Config['RGBCOLOR_BLUE'], 5
            )
        cv2.imwrite(self.m_Config['FIT_SAVE_PATH']+'/'+getNameFromPath(ImgPath), ColoredImg)
            


    def process(self, vDataPath):
        if os.path.isdir(vDataPath):
            NameLis = os.listdir(vDataPath)
            for i in NameLis:
                self._adjust(vDataPath+'/'+i)
                self._fitCircle(vDataPath+'/'+i)
                
        elif os.path.isfile(vDataPath):
            self._adjust(vDataPath)
            self._fitCircle(vDataPath)
        else:
            print("[ERROR] Check your vDataPath!")

        
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    cr = CClockRecognizer(config)
    cr.process(config['INPUT_IMG_PATH'])
    
