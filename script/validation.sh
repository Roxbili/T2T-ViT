#!/bin/bash

# python main.py /home/fanding/dataset/ILSVRC2012 --model t2t_vit_7 -b 128 --eval_checkpoint checkpoint/71.7_T2T_ViT_7.pth.tar
python main.py /home/fanding/dataset/ILSVRC2012 --model t2t_vit_t_14 -b 64 --eval_checkpoint checkpoint/81.7_T2T_ViTt_14.pth.tar