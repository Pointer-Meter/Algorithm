from distutils.command.config import config
from mmdet.apis import init_detector, inference_detector
from mmdet.core.mask.structures import bitmap_to_polygon
import mmcv
from math import atan, sqrt, pi, sin, cos, degrees, radians
import yaml, json
from myUtils import *

class CClockRecognizer:
    def __init__(self, vConfig):
        self.m_Config = vConfig
        self.m_model = init_detector(
            vConfig['MODEL_CONFIG'], 
            vConfig['CHECKPOINT'], 
            device='cuda:0'
            )
        self.m_AdjustScaleMask = None
        self.m_AdjustPointerMask = None

        self.m_AdjustScaleCorner = None
        self.m_ScaleCircle = None
        self.m_AdjustPointerBbox = None
        self.m_PointerPoint = None

    def _saveParam(self, vSavePath):
        with open(vSavePath, 'w', encoding='utf-8') as f:
            ParamDict = {
                "AdjustScaleCorner":self.m_AdjustScaleCorner,
                "ScaleCircle":self.m_ScaleCircle,
                "AdjustPointerBbox":self.m_AdjustPointerBbox,#
                "PointerPoint":self.m_PointerPoint
            }
            json.dump(ParamDict, f, indent=1)

    def loadParam(self, vSavePath):
        with open(vSavePath, 'r', encoding='utf-8') as f:
            ParamDict = json.load(f)
            self.m_AdjustScaleCorner = ParamDict["AdjustScaleCorner"]
            self.m_ScaleCircle = ParamDict["ScaleCircle"]
            self.m_AdjustPointerBbox = ParamDict["AdjustPointerBbox"]
            self.m_PointerPoint = ParamDict["PointerPoint"]

    def _detectCorner(self, vMask, vRawImg, vMinDist, vImgSize):
        CornerImg = vRawImg.copy()
        BlockSize=vImgSize*self.m_Config['BLOCK_SIZE_FACTOR']
        Corners = cv2.goodFeaturesToTrack(vMask, 
            self.m_Config['MAX_CORNERS'], 
            self.m_Config['QUALITY_LEVEL'], 
            self.m_Config['MIN_DISTANCE_FACTOR']*vMinDist,
            mask=None,
            blockSize=int(BlockSize)
            )
        for p in Corners:
            CornerImg = cv2.circle(CornerImg, 
                (int(p[0][0]), int(p[0][1])), 
                10, self.m_Config['RGBCOLOR_RED'], 5)
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
            vImg.shape, vScaleCorner[0], vScaleCorner[1], vRefPoint)
        RotateImg = cv2.warpAffine(vImg, RotateMatrix, (NewW, NewH))
        return RotateImg, RotateMatrix, NewW, NewH

    def _perspective(self, vImg, vScaleMask, vPointerMask, vScaleCorner):
        PerMatrix = self._getPerspectiveMatrix(vScaleMask)
        PerImg = cv2.warpPerspective(vImg, PerMatrix, 
            tuple(self.m_Config['REC']['SIZE']))
        ScaleMask = cv2.warpPerspective(vScaleMask, PerMatrix, 
            tuple(self.m_Config['REC']['SIZE']))
        PointerMask = cv2.warpPerspective(vPointerMask, PerMatrix, 
            tuple(self.m_Config['REC']['SIZE']))
        ConcatMask = ScaleMask + PointerMask
        ConcatMask[ConcatMask>0] = 255

        # ????????????
        t = np.dot(PerMatrix, vScaleCorner.T).T
        ScaleCorner = np.array([[t[0][0]/t[0][2], t[0][1]/t[0][2]],
            [t[1][0]/t[1][2], t[1][1]/t[1][2]]]).tolist()
        return PerImg, ScaleMask, PointerMask, ScaleCorner

    def _adjust(self, vImgPath):
        # ??????mask
        InferenceResult = inference_detector(self.m_model, vImgPath)
        self.m_model.show_result(vImgPath, InferenceResult,
            out_file=config['INFERENCE_SAVE_PATH']+'/'+ getNameFromPath(vImgPath))
        
        # -- ??????????????????
        Bbox, Segm = getInferenceResult(InferenceResult)
        PointerBbox, ScaleBbox = Bbox
        PointerMask, ScaleMask = Segm

        PointerMask = PointerMask.astype(np.uint8)
        ScaleMask = ScaleMask.astype(np.uint8)

        ReferencePoint = [(Bbox[0][0]+Bbox[0][2])/2, 
            (Bbox[0][1]+Bbox[0][3])/2]
        
        # ????????????????????????mask????????????????????????????????????
        # ?????????????????????mask?????????mask
        RawImg = cv2.imread(vImgPath)
        CornerLis, CornerImg = self._detectCorner(ScaleMask, RawImg, ScaleBbox[2]-ScaleBbox[0], max(RawImg.shape))
        ScaleCorner = [[CornerLis[0][0][0], CornerLis[0][0][1]],
            [CornerLis[1][0][0], CornerLis[1][0][1]]]
        RotateImg, RotateMatrix, NewW, NewH = self._rotate(CornerImg, ScaleCorner, ReferencePoint)
        RotateScaleMask = cv2.warpAffine(ScaleMask, RotateMatrix, (NewW, NewH))
        RotatePointerMask = cv2.warpAffine(PointerMask, RotateMatrix, (NewW, NewH))
        
        # ?????????????????????mask??????
        # ConcatMask = RotateScaleMask + RotatePointerMask
        # ConcatMask[ConcatMask>0] = 255
        # cv2.imwrite(self.m_Config['MASK_SAVE_PATH']+'/'+ 'Rotate_' + getNameFromPath(vImgPath), ConcatMask)
        
        # ????????????????????????
        onev, oneh = np.array([[0, 0, 1]]), np.array([[1], [1]])
        RotateMatrix = np.vstack((RotateMatrix, onev))
        RotateScaleCorner = np.hstack((ScaleCorner, oneh)).T
        RotateScaleCorner = np.dot(RotateMatrix, RotateScaleCorner).T

        # ?????????????????????mask?????????mask??????????????????????????????
        PerImg, self.m_AdjustScaleMask, self.m_AdjustPointerMask, self.m_AdjustScaleCorner = self._perspective(
            RotateImg, RotateScaleMask, RotatePointerMask, RotateScaleCorner)

        ConcatMask = self.m_AdjustScaleMask + self.m_AdjustPointerMask
        ConcatMask[ConcatMask>0] = 255
        cv2.imwrite(self.m_Config['MASK_SAVE_PATH']+'/'+ 'Adjust_' + getNameFromPath(vImgPath), ConcatMask)

        cv2.imwrite(self.m_Config['ADJUST_SAVE_PATH']+'/'+getNameFromPath(vImgPath), PerImg) 

    def _fitCircle(self, vImgPath):
        Contour, _ = bitmap_to_polygon(self.m_AdjustScaleMask)
        ImgPath = self.m_Config['ADJUST_SAVE_PATH']+'/'+getNameFromPath(vImgPath)
        Img = cv2.imread(ImgPath)
        ColoredImg = Img.copy()
        # ?????????
        # ???????????????????????????---????????????
        CircleLis1, CircleLis2 = [], []
        i, cnt = 0, 0
        Lock = False

        while True:
            # Contour: [array[[x,y],[x,y],[x,y]...]]
            [x, y] = Contour[0][i]
            i = (i + self.m_Config['SAMPLE_INDEX']) % len(Contour[0]) # ?????????????????????n?????????1???
            add = False
            d1 = calPointDistance(x, y, self.m_AdjustScaleCorner[0][0], self.m_AdjustScaleCorner[0][1])
            d2 = calPointDistance(x, y, self.m_AdjustScaleCorner[1][0], self.m_AdjustScaleCorner[1][1])

            if d1 <= self.m_Config['SAMPLE_DIST_TO_CORNER'] or d2 <= self.m_Config['SAMPLE_DIST_TO_CORNER']:
                if not Lock:
                    Lock = True
                    cnt += 1
                continue
            
            Lock = False
            if cnt % 2 == 0:
                CircleLis1.append([x, y])
            else:
                CircleLis2.append([x, y])
            # ??????????????????
            ColoredImg = cv2.circle(
                ColoredImg, (int(x), int(y)), 4, 
                tuple(self.m_Config['RGBCOLOR_CYAN']), 2)
            
            ColoredImg = cv2.putText(ColoredImg, "("+str(x)+","+str(y)+")", 
                (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, 
                self.m_Config['RGBCOLOR_BLUE'], 1)

            if len(CircleLis1) > 19 and len(CircleLis2) > 19:
                break
        A1, B1, R1 = calCircleCenter(CircleLis1)
        A2, B2, R2 = calCircleCenter(CircleLis2)
        self.m_ScaleCircle = [(A1+A2)/2,(B1+B2)/2,(0.8*R1+0.2*R2)]

        # --- ????????????
        # ???????????????????????????????????????????????????bbox
        # -- ?????????BBOX?????????????????????????????????????????????
        InferenceResult = inference_detector(self.m_model, ImgPath)
        
        Bbox, _ = getInferenceResult(InferenceResult)
        PointerBbox, ScaleBbox = Bbox
        # BBOX: [[w, h], [w, h]]
        # TypeError: Object of type float32 is not JSON serializable
        self.m_AdjustPointerBbox = [[float(PointerBbox[0]), float(PointerBbox[1])],
             [float(PointerBbox[2]), float(PointerBbox[3])]]
        # -- ?????????????????????
        PointerXLis, PointerYLis = [], []
        for h in range(int(PointerBbox[1]), int(PointerBbox[3])):
            for w in range(int(PointerBbox[0]), int(PointerBbox[2])):
                if self.m_AdjustPointerMask[h][w] > 0:
                    PointerXLis.append(w)
                    PointerYLis.append(h)
                    # ?????????
                    ColoredImg[h][w] = 255


        PointerFitParam = calPCA(PointerXLis, PointerYLis)
        PointerMaskIndexMat = np.vstack((np.array([PointerXLis]), np.array([PointerYLis]))).T

        # -- ?????????????????????
        NormalK = -1 / PointerFitParam[0]
        NormalB = self.m_ScaleCircle[1] - NormalK * self.m_ScaleCircle[0]
        AvgSum = [0, 0]
        cnt1, cnt2 = 1, 1
        for pt in PointerMaskIndexMat:
            w, h = pt
            DistToCenter = calPointDistance(w, h, 
                self.m_ScaleCircle[0], self.m_ScaleCircle[1])
            if NormalK*w+NormalB-h>0:
                AvgSum[0] = max(AvgSum[0], DistToCenter)
                # AvgSum[0] = AvgSum[0] / cnt1 * (cnt1-1) + DistToCenter / cnt1
                # cnt1 += 1
            elif NormalK*w+NormalB-h<0:
                AvgSum[1] = max(AvgSum[1], DistToCenter)
                # AvgSum[1] = AvgSum[1] / cnt2 * (cnt2-1) + DistToCenter / cnt2
                # cnt2 += 1
            else:
                continue
        # -- ????????????????????????
        # ax^2 + bx + c = 0
        a = PointerFitParam[0]*PointerFitParam[0]+1
        b = 2*(PointerFitParam[0]*PointerFitParam[1]-
            self.m_ScaleCircle[0]-
            PointerFitParam[0]*self.m_ScaleCircle[1])
        c = (self.m_ScaleCircle[0]*self.m_ScaleCircle[0]+
            PointerFitParam[1]*PointerFitParam[1]+
            self.m_ScaleCircle[1]*self.m_ScaleCircle[1]-
            self.m_ScaleCircle[2]*self.m_ScaleCircle[2]-
            2*self.m_ScaleCircle[1]*PointerFitParam[1])
        
        w1 = (-b-sqrt(b*b-4*a*c))/(2*a)
        h1 = PointerFitParam[0]*w1+PointerFitParam[1]
        w2 = (-b+sqrt(b*b-4*a*c))/(2*a)
        h2 = PointerFitParam[0]*w2+PointerFitParam[1]

        assert ((w1-self.m_ScaleCircle[0])*(w1-self.m_ScaleCircle[0])+
            (h1-self.m_ScaleCircle[1])*(h1-self.m_ScaleCircle[1])-
            self.m_ScaleCircle[2]*self.m_ScaleCircle[2]) < 1e-3, "The point is not on the circle!"

        if (NormalK*w1+NormalB-h1)*(AvgSum[0]-AvgSum[1])>0 and (NormalK*w2+NormalB-h2)*(AvgSum[0]-AvgSum[1])<0:
            self.m_PointerPoint = [w1, h1]
        elif (NormalK*w1+NormalB-h1)*(AvgSum[0]-AvgSum[1])<0 and (NormalK*w2+NormalB-h2)*(AvgSum[0]-AvgSum[1])>0:
            self.m_PointerPoint = [w2, h2]
        else:
            raise Exception("ERROR! Please Check Your Cal.")
        

        # --- ???????????????
        # -- ????????????????????????
        ColoredImg = cv2.circle(ColoredImg, 
            (int(self.m_ScaleCircle[0]), int(self.m_ScaleCircle[1])),
            int(self.m_ScaleCircle[2]), self.m_Config['RGBCOLOR_BLUE'], 2)
        # -- ???????????????????????????
        ColoredImg = cv2.circle(ColoredImg, 
            (int(self.m_ScaleCircle[0]), int(self.m_ScaleCircle[1])),
            5, self.m_Config['RGBCOLOR_BLUE'], 2)
        # -- ?????????????????????????????????
        ColoredImg = cv2.line(ColoredImg, 
            (int(self.m_ScaleCircle[0]), int(self.m_ScaleCircle[1])),
            (int(self.m_PointerPoint[0]), int(self.m_PointerPoint[1])),
            self.m_Config['RGBCOLOR_ORANGE'], 2)

        # -- ???????????????
        ColoredImg = cv2.circle(ColoredImg, (int(self.m_AdjustScaleCorner[0][0]), int(self.m_AdjustScaleCorner[0][1])),
            3, self.m_Config['RGBCOLOR_YELLOW'], 2)
        ColoredImg = cv2.circle(ColoredImg, (int(self.m_AdjustScaleCorner[1][0]), int(self.m_AdjustScaleCorner[1][1])),
            3, self.m_Config['RGBCOLOR_YELLOW'], 2)
        ColoredImg = cv2.putText(ColoredImg, 
            "("+str(int(self.m_AdjustScaleCorner[0][0]))+","+str(int(self.m_AdjustScaleCorner[0][1]))+")", 
            (int(self.m_AdjustScaleCorner[0][0]), int(self.m_AdjustScaleCorner[0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.4, 
            self.m_Config['RGBCOLOR_RED'], 1)
        ColoredImg = cv2.putText(ColoredImg, 
            "("+str(int(self.m_AdjustScaleCorner[1][0]))+","+str(int(self.m_AdjustScaleCorner[1][1]))+")", 
            (int(self.m_AdjustScaleCorner[1][0]), int(self.m_AdjustScaleCorner[1][1])), cv2.FONT_HERSHEY_COMPLEX, 0.4, 
            self.m_Config['RGBCOLOR_RED'], 1)
        # -- ???????????????BBOX
        ColoredImg = cv2.rectangle(ColoredImg, 
            (int(self.m_AdjustPointerBbox[0][0]), int(self.m_AdjustPointerBbox[0][1])), 
            (int(self.m_AdjustPointerBbox[1][0]), int(self.m_AdjustPointerBbox[1][1])),
            self.m_Config['RGBCOLOR_BLUE'], 5)
        # -- ????????????????????? (w, h)
        ColoredImg = cv2.line(ColoredImg, (500, 100), (600, 100), self.m_Config['RGBCOLOR_YELLOW'], 2) 
        cv2.imwrite(self.m_Config['FIT_SAVE_PATH']+'/'+getNameFromPath(ImgPath), ColoredImg)
            

    def process(self, vDataPath):
        if os.path.isdir(vDataPath):
            NameLis = os.listdir(vDataPath)
            for i in NameLis:
                try:
                    print('--------------------------------------------')
                    print(">> adjusting " + vDataPath+'/'+i + ' ...')
                    self._adjust(vDataPath+'/'+i)
                    print(">> fitting " + vDataPath+'/'+i + ' ...')
                    self._fitCircle(vDataPath+'/'+i)
                    self._saveParam(self.m_Config['PARAM_SAVE_PATH']+'/'+getNameFromPath(i,vWithSuffix=False)+".json")
                except Exception as ErrMsg:
                    print(ErrMsg)

        elif os.path.isfile(vDataPath):
            try:
                print('--------------------------------------------')
                print(">> adjusting " + vDataPath + ' ...')
                self._adjust(vDataPath)
                print(">> fitting " + vDataPath + ' ...')
                self._fitCircle(vDataPath)
            except Exception as ErrMsg:
                print(ErrMsg)
        else:
            print("[ERROR] Check your vDataPath!")

        
if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    cr = CClockRecognizer(config)
    cr.process(config['INPUT_IMG_PATH'])
    
