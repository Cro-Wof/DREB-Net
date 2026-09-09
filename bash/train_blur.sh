#!/usr/bin/env bash
set -euo pipefail

ARCH=DREB_Net
EXP_ID=train_DREB_Net_VID_blur
DATASET=visdrone_vid
INP_SHARP_OR_BLUR=blur
SHARP_DATA_DIR=/home/zhuhongxiang/DataSet/VisDrone2019-VID
BLUR_DATA_DIR=/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB
# Keep these defaults aligned with train_sharp.sh for the input-quality comparison.
CUDA_TRAIN_DEVICE=${CUDA_TRAIN_DEVICE:-1}
MASTER_BATCH_SIZE=${MASTER_BATCH_SIZE:-4}
MAX_FRAMES_PER_SEQUENCE=${MAX_FRAMES_PER_SEQUENCE:-50}
NUM_EPOCHS=${NUM_EPOCHS:-200}
VAL_INTERVALS=${VAL_INTERVALS:-1}
PRINT_ITER=${PRINT_ITER:-100}
NUM_WORKERS=${NUM_WORKERS:-8}

CUDA_VISIBLE_DEVICES=$CUDA_TRAIN_DEVICE python -u main.py \
  --exp_id "$EXP_ID" \
  --arch "$ARCH" \
  --num_input_frames 1 \
  --dataset "$DATASET" \
  --inp_sharp_or_blur "$INP_SHARP_OR_BLUR" \
  --sharp_data_dir "$SHARP_DATA_DIR" \
  --blur_data_dir "$BLUR_DATA_DIR" \
  --input_res 1024 \
  --mode train \
  --batch_size "$MASTER_BATCH_SIZE" \
  --master_batch_size "$MASTER_BATCH_SIZE" \
  --lr 2e-4 \
  --num_epochs "$NUM_EPOCHS" \
  --val_intervals "$VAL_INTERVALS" \
  --max_frames_per_sequence "$MAX_FRAMES_PER_SEQUENCE" \
  --print_iter "$PRINT_ITER" \
  --num_workers "$NUM_WORKERS" \
  --gpus "$CUDA_TRAIN_DEVICE"
