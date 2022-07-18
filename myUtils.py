import os
import yaml
import cv2
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from math import sqrt

def initDir(vConfig):
    InitList = [
        'IMG_SAVE_PATH', 
        'INFERENCE_SAVE_PATH', 
        'OCR_SAVE_PATH',
        'ADJUST_SAVE_PATH',
        'INPUT_IMG_PATH',
        'FIT_SAVE_PATH',
        'MASK_SAVE_PATH'
        ]
    for key in InitList:
        if not os.path.exists(vConfig[key]):
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

def getNameFromPath(vPath):
    lis = vPath.split('/')
    Name = lis[-1]
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

if __name__ == '__main__':
    a = './configs/myConfigs/pointer_config.py'
    print(getNameFromPath(a))