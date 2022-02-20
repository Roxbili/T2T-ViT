#!/bin/bash

###### main script
# SCRIPT_NAME="main.py"
SCRIPT_NAME="train.py"

###### dataset
DATASET_DIR="cifar/cifar-10"
DATASET_SRC="/home/LAB/leifd/dataset/${DATASET_DIR}"

# DATASET_DST="/tmp/${DATASET_DIR}"
# DATASET_READY=false


###### run
# no forward evalution
python ${SCRIPT_NAME} ${DATASET_DIR} --model search_model -b 64 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 2 --pin-mem -d torch/cifar10 --dataset-download --img-size 32 --num-classes 10 --use-multi-epochs-loader

# forward evalution
# python ${SCRIPT_NAME} ${DATASET_DIR} --model search_model -b 64 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 2 --pin-mem --search -d torch/cifar10 --num-classes 10 --dataset-download --img-size 32 --use-multi-epochs-loader