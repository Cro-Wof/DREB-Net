#!/usr/bin/env bash

# DREB-VID training entry point.
#
# Default configuration: the completed single-frame baseline.
# 3F-Align configuration:
#   ARCH=DREB_Net_MF_RG NUM_INPUT_FRAMES=3 \
#   EXP_ID=train_DREB_Net_VID_3f_align bash bash/train.sh
#
# All variables below can be overridden from the command line. For example:
#   CUDA_TRAIN_DEVICE=1 MASTER_BATCH_SIZE=4 NUM_EPOCHS=200 \
#   bash bash/train.sh

# Model architecture. Keep the default as DREB_Net so running this script
# without overrides reproduces the existing 1F baseline.
# DREB_Net
# DREB_Net_MF (3F-Align)
# DREB_Net_MF_RG (3F-Align + temporal reliability gate)
ARCH=${ARCH:-DREB_Net_MF_RG}

# Number of ordered video frames provided to each sample:
#   1: original single-frame DREB
#   3: DREB_Net_MF / DREB_Net_MF_RG three-frame modes
NUM_INPUT_FRAMES=${NUM_INPUT_FRAMES:-3}


EXP_ID=${EXP_ID:-train_DREB_Net_VID_RG}


DATASET=visdrone_vid
INP_SHARP_OR_BLUR=SB_deblur

# Original sharp VisDrone-VID sequences and generated DREB blur data.
SHARP_DATA_DIR=${SHARP_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID}
BLUR_DATA_DIR=${BLUR_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB}

# Checkpoints are written by main.py under this experiment directory.
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth

# CUDA_TRAIN_DEVICE is the physical GPU index before CUDA_VISIBLE_DEVICES
# remaps it. The project convention is GPU 1.
CUDA_TRAIN_DEVICE=${CUDA_TRAIN_DEVICE:-1}

# 3F-Align uses more memory than 1F. Lower this value if necessary; keep the
# effective batch size and learning-rate protocol documented for comparisons.
MASTER_BATCH_SIZE=${MASTER_BATCH_SIZE:-4}

# Keep the first N consecutive frames from each sequence for the current
# debugging experiment. 0 means use all frames.
MAX_FRAMES_PER_SEQUENCE=${MAX_FRAMES_PER_SEQUENCE:-50}

INPUT_RES=${INPUT_RES:-1024}
LR=${LR:-2e-4}
NUM_EPOCHS=${NUM_EPOCHS:-200}
VAL_INTERVALS=${VAL_INTERVALS:-1}
PRINT_ITER=${PRINT_ITER:-100}
NUM_WORKERS=${NUM_WORKERS:-8}

case "$ARCH" in
  DREB_Net_MF|DREB_Net_MF_RG)
    if [[ "$NUM_INPUT_FRAMES" != "3" ]]; then
      echo "$ARCH requires NUM_INPUT_FRAMES=3" >&2
      exit 1
    fi
    ;;
  *)
    if [[ "$NUM_INPUT_FRAMES" != "1" ]]; then
      echo "Only DREB_Net_MF and DREB_Net_MF_RG support NUM_INPUT_FRAMES=3" >&2
      exit 1
    fi
    ;;
esac

echo "Starting training: arch=${ARCH}, frames=${NUM_INPUT_FRAMES}, exp_id=${EXP_ID}"
echo "GPU=${CUDA_TRAIN_DEVICE}, batch_size=${MASTER_BATCH_SIZE}, epochs=${NUM_EPOCHS}"
echo "max_frames_per_sequence=${MAX_FRAMES_PER_SEQUENCE}, input_res=${INPUT_RES}"

CUDA_VISIBLE_DEVICES="$CUDA_TRAIN_DEVICE" python -u main.py \
  --exp_id "$EXP_ID" \
  --arch "$ARCH" \
  --num_input_frames "$NUM_INPUT_FRAMES" \
  --dataset "$DATASET" \
  --inp_sharp_or_blur "$INP_SHARP_OR_BLUR" \
  --sharp_data_dir "$SHARP_DATA_DIR" \
  --blur_data_dir "$BLUR_DATA_DIR" \
  --input_res "$INPUT_RES" \
  --mode train \
  --batch_size "$MASTER_BATCH_SIZE" \
  --master_batch_size "$MASTER_BATCH_SIZE" \
  --lr "$LR" \
  --num_epochs "$NUM_EPOCHS" \
  --val_intervals "$VAL_INTERVALS" \
  --max_frames_per_sequence "$MAX_FRAMES_PER_SEQUENCE" \
  --print_iter "$PRINT_ITER" \
  --num_workers "$NUM_WORKERS" \
  --gpus "$CUDA_TRAIN_DEVICE"
