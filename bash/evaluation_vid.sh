#!/usr/bin/env bash

# Evaluation entry point for the independently trained A/B/C experiments.
#
# Defaults to B, matching bash/train.sh:
#   DREB_Net_MF_TDS + 3 frames + train_DREB_Net_VID_3f_B/model_last.pth
#
# Examples:
#   # A: original two-level alignment baseline
#   ARCH=DREB_Net_MF TRAIN_EXP_ID=train_DREB_Net_VID_A \
#     EXP_ID=test_DREB_Net_VID_A_last bash bash/evaluation_vid.sh
#
#   # B: temporal detection supervision
#   ARCH=DREB_Net_MF_TDS TRAIN_EXP_ID=train_DREB_Net_VID_B \
#     EXP_ID=test_DREB_Net_VID_B_last bash bash/evaluation_vid.sh
#
#   # C: temporal detection supervision + hard-negative training
#   ARCH=DREB_Net_MF_TDS TRAIN_EXP_ID=train_DREB_Net_VID_C \
#     EXP_ID=test_DREB_Net_VID_C_last bash bash/evaluation_vid.sh
#
# If EXP_ID already exists, opts.py automatically appends _1, _2, ... so
# result.txt, results.json and visualizations from separate tests are not mixed.

ARCH=${ARCH:-DREB_Net_MF_TDS}
NUM_INPUT_FRAMES=${NUM_INPUT_FRAMES:-3}

case "$ARCH" in
  DREB_Net_MF)
    DEFAULT_TRAIN_EXP_ID=train_DREB_Net_VID_A
    DEFAULT_TEST_EXP_ID=test_DREB_Net_VID_A_last
    ;;
  DREB_Net_MF_TDS)
    DEFAULT_TRAIN_EXP_ID=train_DREB_Net_VID_3f_B
    DEFAULT_TEST_EXP_ID=test_DREB_Net_VID_3f_B_last
    ;;
  DREB_Net_MF_RG)
    DEFAULT_TRAIN_EXP_ID=train_DREB_Net_VID_RG
    DEFAULT_TEST_EXP_ID=test_DREB_Net_VID_RG_last
    ;;
  DREB_Net)
    DEFAULT_TRAIN_EXP_ID=train_DREB_Net_VID_single
    DEFAULT_TEST_EXP_ID=test_DREB_Net_VID_single_last
    ;;
  *)
    echo "Unsupported ARCH: ${ARCH}" >&2
    exit 1
    ;;
esac

TRAIN_EXP_ID=${TRAIN_EXP_ID:-$DEFAULT_TRAIN_EXP_ID}
EXP_ID=${EXP_ID:-$DEFAULT_TEST_EXP_ID}
DATASET=visdrone_vid
INP_SHARP_OR_BLUR=SB_deblur
SHARP_DATA_DIR=${SHARP_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID}
BLUR_DATA_DIR=${BLUR_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB}
MODEL_PATH=${MODEL_PATH:-./exp/detect/train/${TRAIN_EXP_ID}/model_last.pth}
CUDA_VAL_DEVICE=${CUDA_VAL_DEVICE:-1}
MAX_FRAMES_PER_SEQUENCE=${MAX_FRAMES_PER_SEQUENCE:-50}
INPUT_RES=${INPUT_RES:-1024}
VIS_THRESH=${VIS_THRESH:-0.3}
VIS_DIR=${VIS_DIR:-}
TEST_SPLIT=${TEST_SPLIT:-val}
FLIP_TEST=${FLIP_TEST:-0}
SAVE_VISUALIZATIONS=${SAVE_VISUALIZATIONS:-1}   #结果可视图

case "$ARCH" in
  DREB_Net_MF|DREB_Net_MF_RG|DREB_Net_MF_TDS)
    if [[ "$NUM_INPUT_FRAMES" != "3" ]]; then
      echo "$ARCH requires NUM_INPUT_FRAMES=3" >&2
      exit 1
    fi
    ;;
  DREB_Net)
    if [[ "$NUM_INPUT_FRAMES" != "1" ]]; then
      echo "DREB_Net requires NUM_INPUT_FRAMES=1" >&2
      exit 1
    fi
    ;;
esac

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Checkpoint not found: $MODEL_PATH" >&2
  echo "Set MODEL_PATH explicitly or set TRAIN_EXP_ID to the training experiment directory." >&2
  exit 1
fi

echo "Starting evaluation: arch=${ARCH}, frames=${NUM_INPUT_FRAMES}, exp_id=${EXP_ID}"
echo "checkpoint=${MODEL_PATH}"
echo "input_res=${INPUT_RES}, max_frames_per_sequence=${MAX_FRAMES_PER_SEQUENCE}"

TRAINVAL_FLAG=()
if [ "$TEST_SPLIT" = "test-dev" ]; then
  TRAINVAL_FLAG+=(--trainval)
fi

VIS_DIR_ARGS=()
if [ -n "$VIS_DIR" ]; then
  VIS_DIR_ARGS+=(--vis_dir "$VIS_DIR")
fi

FLIP_TEST_FLAG=()
if [ "$FLIP_TEST" = "1" ]; then
  FLIP_TEST_FLAG+=(--flip_test)
fi

VISUALIZATION_FLAG=()
if [ "$SAVE_VISUALIZATIONS" = "1" ]; then
  VISUALIZATION_FLAG+=(--save_visualizations)
fi

CUDA_VISIBLE_DEVICES=$CUDA_VAL_DEVICE python -u test.py \
  --exp_id "$EXP_ID" \
  --arch "$ARCH" \
  --num_input_frames "$NUM_INPUT_FRAMES" \
  --dataset "$DATASET" \
  --inp_sharp_or_blur "$INP_SHARP_OR_BLUR" \
  --sharp_data_dir "$SHARP_DATA_DIR" \
  --blur_data_dir "$BLUR_DATA_DIR" \
  --input_res "$INPUT_RES" \
  --gpus "$CUDA_VAL_DEVICE" \
  --mode test \
  --load_model "$MODEL_PATH" \
  --max_frames_per_sequence "$MAX_FRAMES_PER_SEQUENCE" \
  --vis_thresh "$VIS_THRESH" \
  "${VISUALIZATION_FLAG[@]}" \
  "${TRAINVAL_FLAG[@]}" \
  "${VIS_DIR_ARGS[@]}" \
  "${FLIP_TEST_FLAG[@]}"
