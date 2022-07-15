import os
import yaml
import cv2

def initDir(vConfig):
    InitList = [
        'IMG_SAVE_PATH', 
        'INFERENCE_SAVE_PATH', 
        'OCR_SAVE_PATH',
        'ADJUST_PATH'
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
