#!/bin/bash

prepare(){
    echo "Preparing data $1"
    time /home/dylan/anaconda2/envs/projects/bin/python $1.py
    echo "$1 done"
}


train(){
    echo "Start training $1"
    time /home/dylan/anaconda2/envs/projects/bin/python train.py $1 | tee ../train_log/$1.log
}


prepare 1_pre_basic
#prepare 1_pre_encode

#train xgb-benchmark
#train lgb-benchmark
