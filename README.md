# T2T-VIT Search

## 目录介绍

### models
- statistic.py: 用于统计推理内存的类
- t2t_vit.py: t2t模型基本结构
- t2t_vit_search.py: 为了适应搜索对t2t模型结构进行修改

## 运行方式
```bash
bash script/train.py
```

从磁盘导入数据集实在是太慢了，导致GPU利用率一直是0。
因此`list(loader)`强行提前将整个数据集加载进内存中(no-prefetcher需要打开)。
目前224大小的数据集是不需要的，改成144缩小一下吧，减小内存占用。