#!/bin/bash

# SCRIPT_NAME="main.py"
SCRIPT_NAME="train.py"

# python ${SCRIPT_NAME} /home/fanding/dataset/ILSVRC2012 --model t2t_vit_7 -b 128 --lr 1e-3 --weight-decay .03 --img-size 224

# my model for converting to tflite
## za pc/v100
# python ${SCRIPT_NAME} /home/fanding/dataset/ILSVRC2012 --model t2t_vit_t_1 -b 32 --lr 1e-3 --weight-decay .03 --img-size 224 --no-prefetcher
# python ${SCRIPT_NAME} /home/fanding/dataset/ILSVRC2012 --model test_model -b 32 --lr 1e-3 --weight-decay .03 --img-size 224
# python ${SCRIPT_NAME} /home/fanding/dataset/ILSVRC2012 --model search_model -b 1024 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 4 --pin-mem --search
# python ${SCRIPT_NAME} /home/fanding/dataset/ILSVRC2012_npy --model search_model -b 4096 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 4 --pin-mem --search --no-prefetcher

## act
# python ${SCRIPT_NAME} /home/LAB/leifd/dataset/ILSVRC2012 --model t2t_vit_t_1 -b 128 --lr 1e-3 --weight-decay .03 --img-size 224
# python ${SCRIPT_NAME} /home/LAB/leifd/dataset/ILSVRC2012 --model search_model -b 8192 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 4 --pin-mem --search --no-prefetcher

# # search model
nnictl create --config nas/test.yml -p 8877
# # nnictl experiment list
# nnictl stop exp_id
# nnictl view uKJofsHS -p 8877