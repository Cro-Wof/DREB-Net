#!/usr/bin/env bash

# DREB-VID training entry point.
#
# Default configuration: two-level three-frame alignment.
# B (temporal detection supervision):
#   ARCH=DREB_Net_MF_TDS TEMPORAL_DET_SUPERVISION=1 \
#   EXP_ID=train_DREB_Net_VID_B bash bash/train.sh
# C (B + hard negative):
#   ARCH=DREB_Net_MF_TDS TEMPORAL_DET_SUPERVISION=1 HARD_NEGATIVE=1 \
#   EXP_ID=train_DREB_Net_VID_C bash bash/train.sh
#
# All variables below can be overridden from the command line. For example:
#   CUDA_TRAIN_DEVICE=1 MASTER_BATCH_SIZE=4 NUM_EPOCHS=200 \
#   bash bash/train.sh

# Model architecture. The current mainline is the two-level three-frame
# model. DREB_Net_MF_RG remains available only for later ablations.
# DREB_Net
# DREB_Net_MF (3F-Align)
# DREB_Net_MF_RG (3F-Align + temporal reliability gate)
# DREB_Net_MF_TDS (3F-Align + temporal detection supervision branch)
ARCH=${ARCH:-DREB_Net_MF_TDS}

# Number of ordered video frames provided to each sample:
#   1: original single-frame DREB
#   3: DREB_Net_MF / DREB_Net_MF_RG three-frame modes
NUM_INPUT_FRAMES=${NUM_INPUT_FRAMES:-3}


EXP_ID=${EXP_ID:-train_DREB_Net_VID_3f_B}


DATASET=visdrone_vid
INP_SHARP_OR_BLUR=SB_deblur

# Original sharp VisDrone-VID sequences and generated DREB blur data.
SHARP_DATA_DIR=${SHARP_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID}
BLUR_DATA_DIR=${BLUR_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB}

# Checkpoints are written by main.py under this experiment directory.
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth
# Optional initialization checkpoint. Leave empty for from-scratch training;
# B/C experiments should set this to the designated common checkpoint.
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

# B/C switches. All are disabled by default so this script retains the
# original two-level alignment behavior unless explicitly enabled.
TEMPORAL_DET_SUPERVISION=${TEMPORAL_DET_SUPERVISION:-1}
TEMPORAL_DET_WEIGHT=${TEMPORAL_DET_WEIGHT:-0.2}
HARD_NEGATIVE=${HARD_NEGATIVE:-0}
HARD_NEGATIVE_WEIGHT=${HARD_NEGATIVE_WEIGHT:-0.1}
HARD_NEGATIVE_WARMUP_EPOCHS=${HARD_NEGATIVE_WARMUP_EPOCHS:-10}
HARD_NEGATIVE_SCORE_THRESH=${HARD_NEGATIVE_SCORE_THRESH:-0.3}
HARD_NEGATIVE_TOPK=${HARD_NEGATIVE_TOPK:-32}
HARD_NEGATIVE_EXCLUSION_MARGIN=${HARD_NEGATIVE_EXCLUSION_MARGIN:-1}

case "$ARCH" in
  DREB_Net_MF|DREB_Net_MF_RG|DREB_Net_MF_TDS)
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

if [[ ( "$TEMPORAL_DET_SUPERVISION" == "1" || "$HARD_NEGATIVE" == "1" ) \
      && "$ARCH" != "DREB_Net_MF_TDS" ]]; then
  echo "TEMPORAL_DET_SUPERVISION and HARD_NEGATIVE require ARCH=DREB_Net_MF_TDS" >&2
  exit 1
fi

if [[ ( "$TEMPORAL_DET_SUPERVISION" == "1" || "$HARD_NEGATIVE" == "1" ) \
      && -n "$LOAD_MODEL" ]]; then
  echo "B/C experiments must train from scratch; leave LOAD_MODEL empty" >&2
  exit 1
fi

TEMPORAL_DET_FLAG=()
if [[ "$TEMPORAL_DET_SUPERVISION" == "1" ]]; then
  TEMPORAL_DET_FLAG+=(--temporal_det_supervision)
fi

HARD_NEGATIVE_FLAG=()
if [[ "$HARD_NEGATIVE" == "1" ]]; then
  HARD_NEGATIVE_FLAG+=(--hard_negative)
fi

LOAD_MODEL_FLAG=()
if [[ -n "$LOAD_MODEL" ]]; then
  LOAD_MODEL_FLAG+=(--load_model "$LOAD_MODEL")
fi

echo "Starting training: arch=${ARCH}, frames=${NUM_INPUT_FRAMES}, exp_id=${EXP_ID}"
echo "GPU=${CUDA_TRAIN_DEVICE}, batch_size=${MASTER_BATCH_SIZE}, epochs=${NUM_EPOCHS}"
echo "max_frames_per_sequence=${MAX_FRAMES_PER_SEQUENCE}, input_res=${INPUT_RES}"
echo "temporal_det_supervision=${TEMPORAL_DET_SUPERVISION} (weight=${TEMPORAL_DET_WEIGHT})"
echo "hard_negative=${HARD_NEGATIVE} (weight=${HARD_NEGATIVE_WEIGHT}, warmup=${HARD_NEGATIVE_WARMUP_EPOCHS}, topk=${HARD_NEGATIVE_TOPK})"
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
  --temporal_det_weight "$TEMPORAL_DET_WEIGHT" \
  --hard_negative_weight "$HARD_NEGATIVE_WEIGHT" \
  --hard_negative_warmup_epochs "$HARD_NEGATIVE_WARMUP_EPOCHS" \
  --hard_negative_score_thresh "$HARD_NEGATIVE_SCORE_THRESH" \
  --hard_negative_topk "$HARD_NEGATIVE_TOPK" \
  --hard_negative_exclusion_margin "$HARD_NEGATIVE_EXCLUSION_MARGIN" \
  "${TEMPORAL_DET_FLAG[@]}" \
  "${HARD_NEGATIVE_FLAG[@]}" \
  "${LOAD_MODEL_FLAG[@]}" \
  --gpus "$CUDA_TRAIN_DEVICE"
