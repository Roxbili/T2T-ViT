#!/bin/bash

# python main.py /home/fanding/dataset/ILSVRC2012 --model t2t_vit_7 -b 128 --lr 1e-3 --weight-decay .03 --img-size 224

# my model for converting to tflite
# python main.py /home/fanding/dataset/ILSVRC2012 --model t2t_vit_t_1 -b 32 --lr 1e-3 --weight-decay .03 --img-size 224
# python main.py /home/LAB/leifd/dataset/ILSVRC2012 --model t2t_vit_t_1 -b 32 --lr 1e-3 --weight-decay .03 --img-size 224

# search model
python main.py /home/fanding/dataset/ILSVRC2012 --model search_model -b 32 --lr 1e-3 --weight-decay .03 --img-size 224