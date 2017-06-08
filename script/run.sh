#!/bin/bash

train(){
    echo "Start training $1"
    time /home/dylan/anaconda2/envs/projects/bin/python train.py $1 | tee ../train_log/$1.log
}

train xgb-benchmark
