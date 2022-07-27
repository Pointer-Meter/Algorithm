import os
import cv2
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from math import sqrt
import numpy as np
import math
import random
import mmcv

def initDir(vConfig):
    for key in vConfig:
        lis = key.split('_')
        if lis[-1] == "PATH" and not os.path.exists(vConfig[key]):
            os.makedirs(vConfig[key])

# 等比例imshow
def showImg(vImg, vWindowName = "Img", vIfMask = False):
    # shape: HWC
    Factor = vImg.shape[0] / vImg.shape[1]
    # print(vImg.shape, Factor)
    # resize WH
    if Factor >= 1:
        Img = cv2.resize(vImg, (int(900/Factor), 900))
    else:
        Img = cv2.resize(vImg, (900, int(900*Factor)))
    if vIfMask:
        Img[Img > 0] = 255
    cv2.imshow(vWindowName, Img)
    cv2.waitKey(0)

def saveOcr(vImgPath, vResult, vImgSavePath):
    Img = Image.open(vImgPath).convert('RGB')
    boxes = [line[0] for line in vResult]
    txts = [line[1][0] for line in vResult]
    scores = [line[1][1] for line in vResult]
    ImShow = draw_ocr(Img, boxes, txts, scores, font_path='./fonts/DejaVuSansMono.ttf')
    ImShow = Image.fromarray(ImShow)
    ImShow.save(vImgSavePath)

def getNameFromPath(vPath, vWithSuffix=True):
    lis = vPath.split('/')
    Name = lis[-1]
    if not vWithSuffix:
        Name = Name.split(".")[0]
    return Name

def calPointDistance(x1, y1, x2, y2):
    return sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def calCircleCenter(vPointsList):  # 最小二乘法计算拟合圆
    X1 = Y1 = X2 = Y2 = X3 = Y3 = X1Y1 = X1Y2 = X2Y1 = 0
    for pt in vPointsList:
        X1 += pt[0]
        Y1 += pt[1]
        X2 += pt[0] * pt[0]
        Y2 += pt[1] * pt[1]
        X3 += pt[0] * pt[0] * pt[0]
        Y3 += pt[1] * pt[1] * pt[1]
        X1Y1 += pt[0] * pt[1]
        X1Y2 += pt[0] * pt[1] * pt[1]
        X2Y1 += pt[0] * pt[0] * pt[1]

    N = len(vPointsList)
    C = N * X2 - X1 * X1
    D = N * X1Y1 - X1 * Y1
    E = N * X3 + N * X1Y2 - (X2 + Y2) * X1
    G = N * Y2 - Y1 * Y1
    H = N * X2Y1 + N * Y3 - (X2 + Y2) * Y1
    a = (H * D - E * G) / (C * G - D * D)
    b = (H * C - E * D) / (D * D - G * C)
    c = -(a * X1 + b * Y1 + X2 + Y2) / N

    A = a / (-2)
    B = b / (-2)
    R = sqrt(a * a + b * b - 4 * c) / 2

    return A, B, R

def calOLS(vXLis, vYLis):
    """
    vXLis: [x1, x2, ...]
    vYLis: [y1, y2, ...]
    Ret: [k, b]
    """
    ones = np.ones(len(vXLis)).reshape(-1, 1)
    X = np.array(vXLis).reshape(-1, 1)
    X = np.concatenate((X, ones), axis=1)
    Y = np.array(vYLis)
    # X^T * X * B = X^T * Y
    # B = (X^T * X)^-1 * (X^T * Y)
    Ret = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    return Ret

def calPCA(vXLis, vYLis, vK = 1):
    """
    vXLis: [x1, x2, ...]
    vYLis: [y1, y2, ...]
    X: [[x1,y1], [x2, y2], ...]
    """
    X = np.array([np.array([vXLis[i], vYLis[i]]) for i in range(len(vXLis))])

    NSamples, NFeatures = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(NFeatures)])
    NormX = X-mean
    ScatterMat = np.dot(np.transpose(NormX), NormX)

    EigVal, EigVec = np.linalg.eig(ScatterMat)
    EigPairs = [(np.abs(EigVal[i]), EigVec[:,i]) for i in range(NFeatures)]
    EigPairs.sort(reverse=True)
    Features = np.array([i[1] for i in EigPairs[:vK]])
    # data = np.dot(NormX,np.transpose(Feature))
    # 求最大
    tx, ty = Features[0]
    k = ty / tx
    b = mean[1] - k*mean[0]
    return np.array([k, b])

def calAbsDegree(vCircleCenter, vCenterPt):
    """计算ocr数字中心的绝对角度
    我们定义0度和h方向同向

    o---------------------->
    |               -90                      180(-180)
    |                ^                           |
    |                |                           |
    |                |                           |
    |                |                           | 
    |180(-180) ------o------- 0    ==>  90 ------o------ -90
    |                |                           |
    |                |                           |
    |                |                           |
    |                V                           V
    V               90                           0
    """
    CenterVec = [vCenterPt[0] - vCircleCenter[0], vCenterPt[1]-vCircleCenter[1]]
    
    Degree = math.degrees(math.atan2(CenterVec[1], CenterVec[0]))
    return (Degree + 270)%360

def calRANSAC(vXLis, vYLis, vConfig):
    Size = len(vXLis)
    assert Size >= 2, "[ERROR] Dot < 2."
    PreInlier = 0
    Epoch = vConfig['RANSAC_EPOCH']
    P = vConfig['RANSAC_P']
    BestK, BestB = 0, 0
    for _ in range(Epoch):
        SampleIdx = random.sample(range(Size),2)
        x1, x2 = vXLis[SampleIdx[0]], vXLis[SampleIdx[1]]
        y1, y2 = vYLis[SampleIdx[0]], vYLis[SampleIdx[1]]
        k = (y2-y1)/(x2-x1)
        b = y1-k*x1

        # 计算内点数量
        NumInlier = 0
        for i in range(Size):
            EstimateY = k * vXLis[i] + b
            if abs(EstimateY - vYLis[i]) < vConfig['RANSAC_SIGMA']:
                NumInlier += 1
        
        # 判断当前模型是否比之前估算的模型好
        if NumInlier > PreInlier:
            PreInlier = NumInlier
            BestK, BestB = k, b

        if NumInlier*2 > Size:
            break
    return BestK, BestB

def getInferenceResult(vInferenceResult):
    BboxResult, SegmResult = vInferenceResult

    bbx = mmcv.concat_list(BboxResult)
    bbx = np.stack(bbx, axis=0)
    sgm = mmcv.concat_list(SegmResult)
    sgm = np.stack(sgm, axis=0)
    Bbox, Segm = [], []
    SortLis = []
    cnt = bbx.shape[0]
    if cnt < 2:
        raise Exception('[ERROR] BBOX < 2')
    for i in range(bbx.shape[0]):
        if bbx[i][4] < 0.5 and cnt > 2:
            cnt -= 1
            continue
        Bbox.append(bbx[i])
        Segm.append(sgm[i])
        # SortLis.append((bbx[i][4], i))
    # print("Not sort: ", SortLis)
    # SortLis = sorted(SortLis, reverse=True)
    # print("Sort: ", SortLis)

    # for i in range(2):
    #     index = i

    #     Bbox.append(bbx[index])
    #     Segm.append(sgm[index])

    Bbox = np.array(Bbox)
    Segm = np.array(Segm)
    return Bbox, Segm

if __name__ == '__main__':
    a = './configs/myConfigs/pointer_config.py'
    print(getNameFromPath(a))