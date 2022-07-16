# 摆正和ocr分开的环境

+ 摆正的环境只需要有
    ```
    python3 -m pip install torch torchvision
    python3 -m pip install openmim
    mim install mmdet
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
    python3 -m pip install openmim==0.1.5 && mim install mmdet
    ```

<!-- 
+ 如果matplotlib冲突，降级labelme `python3 -m pip install labelme==4.2.0`

+ 如果缺wrapt `python3 -m pip install wrapt`

+ `Click`包可能有冲突但不影响 -->
