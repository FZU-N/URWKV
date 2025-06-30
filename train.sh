#!/bin/bash

# Print usage instructions
print_usage() {
    echo "Usage: $0 <gpu_id> <dataset> [<model_name>] [<block_num>] [<recursive_num>] [<batch_size>] [<epochs>] [<lr_init>] [<lr_min>]"
    echo "       dataset options:"
    echo "            LOL_v1      -> Use LOL_v1 dataset for training"
    echo "            LOL_v2_real -> Use LOL_v2_real dataset for training"
    echo "            LOL_v2_sync -> Use LOL_v2_sync dataset for training"
    echo "            MIT_5K      -> Use MIT_5K dataset for training"
    echo "            SID        -> Use SID dataset for training"  
    echo "            SMID        -> Use SMID dataset for training"    
    echo "            SDSD_indoor        -> Use SDSD_indoor dataset for training"    
    echo "            SDSD_outdoor        -> Use SDSD_outdoor dataset for training"    
    echo "            LOL_blur      -> Use LOL_blur dataset for training"
    echo "Optional arguments (default values are used if not provided):"
    echo "    model_name: Model name (default: URWKV)"
    echo "    block_num: Number of blocks (default: 2)"
    echo "    recursive_num: Number of recursive blocks (default: 3)"
    echo "    batch_size: Batch size (default: 8)"
    echo "    epochs: Number of epochs (default: 1000)"
    echo "    lr_init: Initial learning rate (default: 0.0002)"
    echo "    lr_min: Minimum learning rate (default: 1e-6)"
}

# Define function for training LOL-v1
train_LOL_v1() {
    python ./tools/train.py \
        --gpu_id $gpu_id \
        --model_name $model_name \
        --yml_path './configs/LOL_v1.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel

    # train_MIT_5K
}

# Define function for training LOL-v2-real
train_LOL_v2_real() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/LOL_v2_real.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
    # train_LOL_v2_sync
}

# Define function for training LOL-v2-sync
train_LOL_v2_sync() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/LOL_v2_synthetic.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
}

# Define function for training MIT_5K
train_MIT_5K() {
    python ./tools/train_MIT5K.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/FiveK.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
    # train_SMID
}

# Define function for training SID
train_SID() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/SID.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
    # train_SDSD_outdoor
}

# Define function for training SMID
train_SMID() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/SMID.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
}

# Define function for training SDSD_indoor
train_SDSD_indoor() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/SDSD_indoor.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
}

# Define function for training SDSD_outdoor
train_SDSD_outdoor() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/SDSD_outdoor.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
}

# Define function for training LOL-blur
train_LOL_blur() {
    python ./tools/train.py \
        --gpu_id $gpu_id  \
        --model_name $model_name \
        --yml_path './configs/LOL_blur.yaml' \
        --pretrain_weights '' \
        --batch_size $batch_size \
        --epochs $epochs \
        --lr_init $lr_init \
        --lr_min $lr_min \
        --channel $channel
}



# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    print_usage
    exit 1
fi

# Parse command line arguments
gpu_id=$1
dataset=$2
model_name=${3:-"URWKV"}  # default value: "URWKV"
batch_size=${4:-8}  # default value: 8
epochs=${5:-1000}  # default value: 1000
lr_init=${6:-0.0002}  # default value: 0.0002
lr_min=${7:-1e-6}  # default value: 1e-6
channel=${8:-32}  # default value: 32

# Execute corresponding function based on dataset selection
case $dataset in
    "LOL_v1")
        train_LOL_v1
        ;;
    "LOL_v2_real")
        train_LOL_v2_real
        ;;
    "LOL_v2_sync")
        train_LOL_v2_sync
        ;;
    "MIT_5K")
        train_MIT_5K
        ;;
    "SID")
        train_SID
        ;;
    "SMID")
        train_SMID
        ;;
    "SDSD_indoor")
        train_SDSD_indoor
        ;;
    "SDSD_outdoor")
        train_SDSD_outdoor
        ;;
    "LOL_blur")
        train_LOL_blur
        ;;
    *)
        echo "Invalid dataset selection. Please choose one of the following: LOL_v1, LOL_v2_real, LOL_v2_sync, MIT_5K."
        print_usage
        exit 1
        ;;
esac