#!/bin/bash
CONFIGS=("ex")

config_length=${#CONFIGS[@]}


for (( i=0; i<${config_length}; i++ ))
do

    echo ${CONFIGS[$i]}
    python3 train_z.py \
        --config ${CONFIGS[$i]}
done