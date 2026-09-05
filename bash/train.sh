#!/usr/bin/env bash

# DREB-VID training entry point.
#
# Default configuration: restored original two-level three-frame alignment.
# Localization-quality branch experiment:
#   ARCH=DREB_Net_MF_LQ LOCALIZATION_QUALITY=1 \
#   EXP_ID=train_DREB_Net_VID_3f_LQ bash bash/train.sh
#
# All variables below can be overridden from the command line. For example:
#   CUDA_TRAIN_DEVICE=1 MASTER_BATCH_SIZE=4 NUM_EPOCHS=200 \
#   bash bash/train.sh

# Model architecture. The current baseline is the original two-level
# three-frame model.
# DREB_Net
# DREB_Net_MF (3F-Align)
# DREB_Net_MF_LQ (3F-Align + localization quality branch)
ARCH=${ARCH:-DREB_Net_MF_LQ}

# Number of ordered video frames provided to each sample:
#   1: original single-frame DREB
#   3: DREB_Net_MF / DREB_Net_MF_LQ three-frame modes
NUM_INPUT_FRAMES=${NUM_INPUT_FRAMES:-3}


EXP_ID=${EXP_ID:-train_DREB_Net_VID_3f_LQ}


DATASET=visdrone_vid
INP_SHARP_OR_BLUR=SB_deblur

# Original sharp VisDrone-VID sequences and generated DREB blur data.
SHARP_DATA_DIR=${SHARP_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID}
BLUR_DATA_DIR=${BLUR_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB}

# Checkpoints are written by main.py under this experiment directory.
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth
# Optional initialization checkpoint for legacy experiments. Leave empty for
# the A/Q comparisons, which must both use independent from-scratch training.
LOAD_MODEL=${LOAD_MODEL:-}

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

# The current default experiment is the independent localization-quality run.
LOCALIZATION_QUALITY=${LOCALIZATION_QUALITY:-1}
LOCALIZATION_QUALITY_WEIGHT=${LOCALIZATION_QUALITY_WEIGHT:-0.5}
LOCALIZATION_QUALITY_SCORE_POWER=${LOCALIZATION_QUALITY_SCORE_POWER:-1.0}

case "$ARCH" in
  DREB_Net_MF|DREB_Net_MF_LQ)
    if [[ "$NUM_INPUT_FRAMES" != "3" ]]; then
      echo "$ARCH requires NUM_INPUT_FRAMES=3" >&2
      exit 1
    fi
    ;;
  *)
    if [[ "$NUM_INPUT_FRAMES" != "1" ]]; then
      echo "Only DREB_Net_MF and DREB_Net_MF_LQ support NUM_INPUT_FRAMES=3" >&2
      exit 1
    fi
    ;;
esac

if [[ "$LOCALIZATION_QUALITY" == "1" && "$ARCH" != "DREB_Net_MF_LQ" ]]; then
  echo "LOCALIZATION_QUALITY requires ARCH=DREB_Net_MF_LQ" >&2
  exit 1
fi

if [[ "$LOCALIZATION_QUALITY" == "1" && -n "$LOAD_MODEL" ]]; then
  echo "Localization-quality experiments must train from scratch; leave LOAD_MODEL empty" >&2
  exit 1
fi

LOCALIZATION_QUALITY_FLAG=()
if [[ "$LOCALIZATION_QUALITY" == "1" ]]; then
  LOCALIZATION_QUALITY_FLAG+=(--localization_quality)
fi

LOAD_MODEL_FLAG=()
if [[ -n "$LOAD_MODEL" ]]; then
  LOAD_MODEL_FLAG+=(--load_model "$LOAD_MODEL")
fi

echo "Starting training: arch=${ARCH}, frames=${NUM_INPUT_FRAMES}, exp_id=${EXP_ID}"
echo "GPU=${CUDA_TRAIN_DEVICE}, batch_size=${MASTER_BATCH_SIZE}, epochs=${NUM_EPOCHS}"
echo "max_frames_per_sequence=${MAX_FRAMES_PER_SEQUENCE}, input_res=${INPUT_RES}"
echo "localization_quality=${LOCALIZATION_QUALITY} (weight=${LOCALIZATION_QUALITY_WEIGHT}, score_power=${LOCALIZATION_QUALITY_SCORE_POWER})"
if [[ -n "$LOAD_MODEL" ]]; then
  echo "initializing from checkpoint: ${LOAD_MODEL}"
fi

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
  --localization_quality_weight "$LOCALIZATION_QUALITY_WEIGHT" \
  --localization_quality_score_power "$LOCALIZATION_QUALITY_SCORE_POWER" \
  "${LOCALIZATION_QUALITY_FLAG[@]}" \
  "${LOAD_MODEL_FLAG[@]}" \
  --gpus "$CUDA_TRAIN_DEVICE"
