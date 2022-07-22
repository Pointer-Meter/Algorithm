import os, yaml
from myUtils import *
from clock_recognizer import CClockRecognizer

class CResultCalculator:
    def __init__(self, vConfig):
        self.m_Config = vConfig
        self.m_ClockRecognizer = None

    def calculate(self):
        pass

    def process(self, vDataPath):
        if os.path.isdir(vDataPath):
            NameLis = os.listdir(vDataPath)
            for i in NameLis:
                print(">> calculating..." + vDataPath+'/'+i + ' ...')
                self.m_ClockRecognizer = CClockRecognizer(self.m_Config)
                self.m_ClockRecognizer.loadParam(vDataPath+'/'+i)
                
        elif os.path.isfile(vDataPath):
            print(">> calculating..." + vDataPath + ' ...')
            self.m_ClockRecognizer = CClockRecognizer(self.m_Config)
            self.m_ClockRecognizer.loadParam(vDataPath)
        
        else:
            print("[ERROR] Check your vDataPath!")

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    rr = CResultCalculator(config)
    rr.process(config['PARAM_SAVE_PATH'])