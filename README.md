# 摆正和ocr分开的环境

+ 摆正的环境只需要有
    ```
    python3 -m pip install torch torchvision
    python3 -m pip install openmim
    mim install mmcv-full
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -r requirements/build.txt
    pip install -v -e .
    ```

+ ocr 环境只需要有

    ```
    python3 -m pip install paddlepaddle-gpu paddleocr
    ```



# 统一环境配置

+ conda环境
    ```sh
    conda create -n open-mmlab python=3.8 -y && conda activate open-mmlab
    ```

+ 先装paddlepaddle和paddleocr
    ```sh
    python3 -m pip install paddlepaddle-gpu==2.0.0 paddleocr==2.2.0
    ```

+ 再装pytorch， mmdetection
    ```sh
    # conda install pytorch torchvision -c pytorch
    python3 -m pip install torch torchvision
    ```

    ```sh
    python3 -m pip install openmim==0.1.5
    mim install mmcv-full
    git clone https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -r requirements/build.txt
    pip install -v -e .
    ```

# 如何修改paddleocr的模型
ocr的检测，识别，方向分类模型的下载链接都在本地目录 `/your_envs/.../lib/python3.8/site-packages/paddleocr/paddleocr.py`里

在[文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.5/doc/doc_ch/ppocr_introduction.md) 中找到对应语言的模型链接替换该文件中的链接，再运行推理即可


# log
+ 7.18 
    - [x] 角点旋转出错
+ 7.19
    - [x] 透视变换的公式写错
    - [ ] 指针方向有问题

+ 7.22
    - [x] 将指针拟合方法从OLS换成了PCA

+ 7.23
    - [x] 使用RANSAC纠错
