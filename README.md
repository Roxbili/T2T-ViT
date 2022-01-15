# T2T-VIT Search

## 环境
- timm
- jpeg4py(需要自行安装或编译`libjpeg-turbo`库)
- pytorch
- nni

### imagenet2npy
[imagenet2npy.py](https://github.com/Roxbili/imagenet_prepare/blob/main/imagenet2npy.py)

(无法将整个numpy导入内存，880+G太大了，之前有jpeg编码的时候100G+还行)

### jpeg4py(弃用)
1. 安装python库：
    ```bash
    pip install jpeg4py
    ```

2. 安装`libjpeg-turbo`

    - apt安装：
        ```bash
        sudo apt-get install libturbojpeg
        ```
    
    - 编译安装：  
        1. 下载`libjpeg-turbo`[源码](https://sourceforge.net/projects/libjpeg-turbo/files/)
        2. 进入源码目录，新建build文件夹：`mkdir build`
        3. `cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/home/LAB/leifd/include/libjpeg-turbo-2.1.2/build /home/LAB/leifd/include/libjpeg-turbo-2.1.2`
        4. `make -j8`
        5. `make install`
        6. 添加环境变量至.zshrc或者.bashrc：`export LD_LIBRARY_PATH=/home/LAB/leifd/include/libjpeg-turbo-2.1.2/build/lib:$LD_LIBRARY_PATH   # jpeg4py需要的依赖`


## 目录介绍

### 根目录
- main.py: 旧训练脚本，timm==0.3.4
- train.py: 适配最新timm的训练脚本

### models
- statistic.py: 用于统计推理内存的类
- t2t_vit.py: t2t模型基本结构
- t2t_vit_search.py: 为了适应搜索对t2t模型结构进行修改

## 运行方式
```bash
bash script/train.sh
```

[问题发现] 从磁盘导入数据集实在是太慢了，导致GPU利用率一直是0。

[解决方案] `list(loader)`强行提前将整个数据集加载进内存中(no-prefetcher需要打开)。
目前224大小的数据集是不需要的，改成144缩小一下吧，减小内存占用。

----------------------------
## 加载图像时间对比
1000次图像加载时间/s (act服务器):
| tool      | time/s             |
|-----------|--------------------|
| jpeg4py   | 4.920205919072032  |
| cv2       | 5.536927016917616  |
| PIL Image | 6.248460555914789  |
| numpy     | 0.5741092250682414 |