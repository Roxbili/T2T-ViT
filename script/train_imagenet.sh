#!/bin/bash

###### main script
# SCRIPT_NAME="main.py"
SCRIPT_NAME="train.py"

###### dataset
# DATASET_DIR="ILSVRC2012"
# DATASET_DIR="ILSVRC2012_npy"
DATASET_DIR="ILSVRC2012_lmdb"
DATASET_SRC="/home/LAB/leifd/dataset/${DATASET_DIR}"
DATASET_DST="/tmp/${DATASET_DIR}"
DATASET_READY=false

# trap ctrl+c
trap 'onCtrlC' INT
function onCtrlC() {
    if [ ${DATASET_READY} = false ]; then
        rm -rf ${DATASET_DST}
        echo "copy dataset interrupted, remove ${DATASET_DST}"
    fi
}

# check if /tmp has dataset
df -h /tmp
if [ ! -d ${DATASET_DST} ]; then
    echo "copy dataset to ${DATASET_DST}"
    cp -r ${DATASET_SRC} ${DATASET_DST}
    if [ $? -ne 0 ]; then   # 命令执行不成功
        rm -rf ${DATASET_DST}
        exit $?
    fi
    DATASET_READY=true
else
    DATASET_READY=true
fi

# run
# python ${SCRIPT_NAME} /home/LAB/leifd/dataset/ILSVRC2012 --model t2t_vit_t_1 -b 128 --lr 1e-3 --weight-decay .03 --img-size 224
# python ${SCRIPT_NAME} /home/LAB/leifd/dataset/ILSVRC2012_npy --model search_model -b 8192 --lr 1e-3 --weight-decay .03 --epochs 1 --workers 2 --pin-mem --search --img-load PIL
# python ${SCRIPT_NAME} /home/LAB/leifd/dataset/ILSVRC2012 --model search_model -b 8192 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 2 --pin-mem --search --img-load PIL --no-prefetcher
python ${SCRIPT_NAME} ${DATASET_DST} --model search_model -b 64 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 2 --pin-mem --search -d lmdb --use-multi-epochs-loader

# distributed
# python -m torch.distributed.launch --nproc_per_node=4 ${SCRIPT_NAME} ${DATASET_SRC} --model search_model -b 8192 --lr 1e-3 --weight-decay .03 --epochs 300 --workers 2 --pin-mem --search -d lmdb --no-prefetcher

###### nni
# nnictl create --config nas/imagenet.yml -p 8877
# # nnictl experiment list
# nnictl stop exp_id
# nnictl view uKJofsHS -p 8877