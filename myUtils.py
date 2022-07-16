import os
import yaml
import cv2
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr

def initDir(vConfig):
    InitList = [
        'IMG_SAVE_PATH', 
        'INFERENCE_SAVE_PATH', 
        'OCR_SAVE_PATH',
        'ADJUST_PATH',
        'INPUT_IMG_PATH'
        ]
    for key in InitList:
        if not os.path.exists(vConfig[key]):
            os.makedirs(vConfig[key])

# 等比例imshow
def showImg(vImg, vWindowName = "Img"):
    # shape: HWC
    Factor = vImg.shape[0] / vImg.shape[1]
    print(vImg.shape, Factor)
    # resize WH
    if Factor >= 1:
        Img = cv2.resize(vImg, (int(900/Factor), 900))
    else:
        Img = cv2.resize(vImg, (900, int(900*Factor)))
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

if __name__ == '__main__':
    a = './configs/myConfigs/pointer_config.py'
    print(getNameFromPath(a))