import os, yaml, json
from myUtils import *
from clock_recognizer import CClockRecognizer

class CResultCalculator:
    def __init__(self, vConfig):
        self.m_Config = vConfig
        self.m_ClockRecognizer = None

    def _preprocess(self, vRawDict):
        FineDic = {}
        XLis, YLis = [], []
        ScaleCenterPt = self.m_ClockRecognizer.m_ScaleCircle[:2]
        ScaleR = self.m_ClockRecognizer.m_ScaleCircle[2]
        for key in vRawDict:
            CenterPt = [
                sum([i[0] for i in vRawDict[key]['Corner']])/4,
                sum([i[1] for i in vRawDict[key]['Corner']])/4]

            # -- 根据角度筛
            Degree = calAbsDegree(ScaleCenterPt, CenterPt)
            if Degree < self.m_Config['MIN_DEGREE'] or Degree > self.m_Config['MAX_DEGREE']:
                continue
            # -- 根据距离筛
            d = calPointDistance(CenterPt[0],CenterPt[1], ScaleCenterPt[0],ScaleCenterPt[1])
            if d < self.m_Config['MIN_R_FACTOR']*ScaleR:
                continue
            Fkey = float(key)
            FineDic[Fkey] = vRawDict[key]
            FineDic[Fkey]['Degree'] = Degree


            # -- 启动RANSAC算法
            XLis.append(Degree)
            YLis.append(Fkey)
        # print(XLis)
        # print(YLis)
        k, b = calRANSAC(XLis, YLis, self.m_Config)
        # print("k: ", k, " b: ", b)
        # -- 获取指针的degree
        PointerDegree = calAbsDegree(ScaleCenterPt,
            self.m_ClockRecognizer.m_PointerPoint)
        # print("PointerDegree: ", PointerDegree)
        Ret = k * PointerDegree + b
        print("Result: ", Ret)
        return FineDic


    def calculate(self, vSavePath):
        """结合ClockRecognizer的参数和ocr结果来计算读数   
        """
        with open(vSavePath, 'r', encoding='utf-8') as f:
            dic = json.load(f)
            Dic = self._preprocess(dic)

    def process(self, vDataPath):
        if os.path.isdir(vDataPath):
            NameLis = os.listdir(vDataPath)
            for i in NameLis:
                try:
                    print(">> calculating..." + vDataPath+'/'+i + ' ...')
                    self.m_ClockRecognizer = CClockRecognizer(self.m_Config)
                    self.m_ClockRecognizer.loadParam(vDataPath+'/'+i)
                    self.calculate(self.m_Config['OCR_FILE_SAVE_PATH']+'/'+i)
                except Exception as ErrMsg:
                    print(ErrMsg) 

                
        elif os.path.isfile(vDataPath):
            try:
                print(">> calculating..." + vDataPath + ' ...')
                self.m_ClockRecognizer = CClockRecognizer(self.m_Config)
                self.m_ClockRecognizer.loadParam(vDataPath)
                self.calculate(self.m_Config['OCR_FILE_SAVE_PATH']+'/'+getNameFromPath(vDataPath))
            except Exception as ErrMsg:
                print(ErrMsg) 
        
        else:
            print("[ERROR] Check your vDataPath!")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    rr = CResultCalculator(config)
    rr.process(config['PARAM_SAVE_PATH'])