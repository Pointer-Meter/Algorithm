from paddleocr import PaddleOCR, draw_ocr
import json
import os
import cv2
import numpy as np
from myUtils import *

class CTextRecognizer:
    def __init__(self, config):
        self.m_model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        self.m_config = config
    
    def _detect(self, vImgPath):
        Result = self.m_model.ocr(vImgPath, cls = True)
        saveOcr(
            vImgPath, 
            Result, 
            self.m_config['OCR_SAVE_PATH']+'/'+getNameFromPath(vImgPath)
            )

    def process(self, vDataPath):
        if os.path.isdir(vDataPath):
            NameLis = os.listdir(vDataPath)
            for i in NameLis:
                print(i)
                self._detect(vDataPath+'/'+i)
                
        elif os.path.isfile(vDataPath):
            self._adjust(vDataPath)
        
        else:
            print("[ERROR] Check your vDataPath!")


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    tr = CTextRecognizer(config)
    tr.process(config['ADJUST_PATH'])
