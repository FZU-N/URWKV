#!/bin/bash

# Print usage instructions -> set parameter 0
print_usage() {
    echo "Usage: $0 <dataset_number>"
    echo "       dataset_number:"
    echo "           1 -> LOL_v1"
    echo "           2 -> LOL_v2_real"
    echo "           3 -> LOL_v2_sync"
    echo "           4 -> MIT_5K"
    echo "           5 -> SID"
    echo "           6 -> SMID"
    echo "           7 -> SDSD_indoor"
    echo "           8 -> SDSD_outdoor"
    echo "           9 -> LOL_blur"
}

# Define function for testing LOL-v1
test_LOL_v1() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/LOL_v1/eval15/' \
        --weight_path './checkpoints/LOL_v1/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/LOL_v1'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name LOL_v1
}

# Define function for testing LOL_v2_real
test_LOL_v2_real() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/LOL_v2/Real_captured/Test/' \
        --weight_path './checkpoints/LOL_v2_real/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/LOL_v2_real'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name LOL_v2_real
}

# Define function for testing LOL_v2_sync
test_LOL_v2_sync() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/LOL_v2/Synthetic/Test/' \
        --weight_path './checkpoints/LOL_v2_sync/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/LOL_v2_sync'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name LOL_v2_sync
}

# Define function for testing MIT_5K
test_MIT_5K() {
    python ./tools/test_MIT5K.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/MIT-Adobe-5K-512/test/' \
        --weight_path './checkpoints/MIT_5K/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/MIT_5K'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name MIT_5K
}

# Define function for testing SID
test_SID() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/SID_png/eval/' \
        --weight_path './checkpoints/SID/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/SID'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name SID
}

# Define function for testing SMID
test_SMID() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/SMID_png/eval/' \
        --weight_path './checkpoints/SMID/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/SMID'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name SMID
}

# Define function for testing SDSD_indoor
test_SDSD_indoor() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/SDSD_indoor_png/eval/' \
        --weight_path './checkpoints/SDSD_indoor/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/SDSD_indoor'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name SDSD_indoor
}

# Define function for testing SDSD_outdoor
test_SDSD_outdoor() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/SDSD_outdoor_png/eval/' \
        --weight_path './checkpoints/SDSD_outdoor/URWKV/models/model_bestPSNR.pth' \
        --save_path './results/SDSD_outdoor'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name SDSD_outdoor
}

# Define function for testing LOL-blur
test_LOL_blur() {
    python ./tools/test.py \
        --gpu_id 0 \
        --model_name URWKV \
        --testSet_path '/data/xr/Dataset/light_dataset/LOL_blur/eval' \
        --weight_path './checkpoints/LOL_blur/URWKV/models/model_bestSSIM.pth' \
        --save_path './results/LOL_blur'

    python ./tools/measure.py \
        --gpu_id 0 \
        --model_name URWKV \
        --dataset_name LOL_blur
}

# Check if argument is provided
if [ $# -ne 1 ]; then
    print_usage
    exit 1
fi

# Parse command line argument
case $1 in
    1)
        test_LOL_v1
        ;;
    2)
        test_LOL_v2_real
        ;;
    3)
        test_LOL_v2_sync
        ;;
    4)
        test_MIT_5K
        ;;
    5)
        test_SID
        ;;
    6)
        test_SMID
        ;;
    7)
        test_SDSD_indoor
        ;;
    8)
        test_SDSD_outdoor
        ;;
    9)
        test_LOL_blur
        ;;
    *)
        echo "Invalid dataset number. Please provide a number between 1 to 4."
        print_usage
        exit 1
        ;;
esac