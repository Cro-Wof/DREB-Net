#!/usr/bin/env bash

# If EXP_ID already exists, opts.py automatically appends _1, _2, ... so
# result.txt, results.json and visualizations from separate tests are not mixed.

ARCH=${ARCH:-DREB_Net_MF_RG}
EXP_ID=${EXP_ID:-test_VID_3f_align_RG_last}
NUM_INPUT_FRAMES=${NUM_INPUT_FRAMES:-3}
DATASET=visdrone_vid
INP_SHARP_OR_BLUR=SB_deblur
SHARP_DATA_DIR=${SHARP_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID}
BLUR_DATA_DIR=${BLUR_DATA_DIR:-/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB}
MODEL_PATH=${MODEL_PATH:-./exp/detect/train/train_DREB_Net_VID_RG/model_last.pth}
CUDA_VAL_DEVICE=${CUDA_VAL_DEVICE:-1}
MAX_FRAMES_PER_SEQUENCE=${MAX_FRAMES_PER_SEQUENCE:-50}
VIS_THRESH=${VIS_THRESH:-0.3}
VIS_DIR=${VIS_DIR:-}
TEST_SPLIT=${TEST_SPLIT:-val}
FLIP_TEST=${FLIP_TEST:-0}
SAVE_VISUALIZATIONS=${SAVE_VISUALIZATIONS:-1}   #结果可视图

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
  --input_res 1024 \
  --gpus "$CUDA_VAL_DEVICE" \
  --mode test \
  --load_model "$MODEL_PATH" \
  --max_frames_per_sequence "$MAX_FRAMES_PER_SEQUENCE" \
  --vis_thresh "$VIS_THRESH" \
  "${VISUALIZATION_FLAG[@]}" \
  "${TRAINVAL_FLAG[@]}" \
  "${VIS_DIR_ARGS[@]}" \
  "${FLIP_TEST_FLAG[@]}"
