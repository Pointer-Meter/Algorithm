from myUtils import *
import json, yaml
class CTextRecognizer:
    def __init__(self, vConfig):
        self.m_Model = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False)
        self.m_Config = vConfig
    
    def _preProcess(self, vImgPath):
        Img = cv2.imread(vImgPath, cv2.IMREAD_GRAYSCALE)
        
        # 增加对比度
        MdImg = cv2.equalizeHist(Img)
        # 二值化
        # _, ModifiedImg = cv2.threshold(Img, 90, 255, cv2.THRESH_BINARY)
        
        # --- 开闭运算比较影响图片质量，不使用
        # Kernel = cv2.getStructuringElement(
        #     cv2.MORPH_RECT, 
        #     (self.m_config['MORPHOLOGY_KERNEL_SIZE'], self.m_config['MORPHOLOGY_KERNEL_SIZE'])
        #     )
        # ModifiedImg = cv2.morphologyEx(
        #     ModifiedImg, cv2.MORPH_OPEN, Kernel, 
        #     iterations=self.m_config['MORPHOLOGY_KERNEL_ITERATION']
        #     )

        Result = self.m_Model.ocr(MdImg, cls = True)
        saveOcr(vImgPath, Result, 
            self.m_Config['OCR_IMG_SAVE_PATH']+'/'+getNameFromPath(vImgPath))
        return Result

    def _postProcess(self, vResult, vSavePath):
        """对ocr结果后处理

        参数
        ---
        arg1: vResult
        [
            [ [p1, p2, p3, p4], (ocr_result, acc)],

            [ [p1, p2, p3, p4], (ocr_result, acc)],

            ...
        ]

        其中在空间中p1(左上),p2(右上),p3(右下),p4(左下)
        
        p1 ----> p2
         ^        |
         |        |
         |        V
        p4 <---- p3

        """
        PostRetDict = {}
        for res in vResult:
            if res[1][1] >= self.m_Config['ACC_THRESH']:
                try:
                    resValue = float(res[1][0])
                except ValueError:
                    continue
                # 强转float，因为 TypeError: Object of type float32 is not JSON serializable
                PostRetDict[resValue] = {"Corner":res[0], "Acc":float(res[1][1])}
        with open(vSavePath, 'w', encoding='utf-8') as f:
            json.dump(PostRetDict, f, indent=1)
        


    def process(self, vDataPath):
        if os.path.isdir(vDataPath):
            NameLis = os.listdir(vDataPath)
            for i in NameLis:
                try:
                    print(">> ocr..." + vDataPath+'/'+i + ' ...')
                    Ret = self._preProcess(vDataPath+'/'+i)
                    self._postProcess(Ret, self.m_Config['OCR_FILE_SAVE_PATH']+'/'+getNameFromPath(i,False)+".json")
                except Exception as ErrMsg:
                    print(ErrMsg)    
            
        elif os.path.isfile(vDataPath):
            try:
                print(">> ocr..." + vDataPath + ' ...')
                Ret = self._preProcess(vDataPath)
                self._postProcess(Ret, self.m_Config['OCR_FILE_SAVE_PATH']+'/'+getNameFromPath(vDataPath))
            except Exception as ErrMsg:
                print(ErrMsg)
        
        else:
            print("[ERROR] Check your vDataPath!")


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    initDir(config)
    tr = CTextRecognizer(config)
    tr.process(config['ADJUST_SAVE_PATH'])
