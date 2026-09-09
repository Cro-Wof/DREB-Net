#!/usr/bin/env bash
set -euo pipefail

export INP_SHARP_OR_BLUR=blur
export EXP_ID=${EXP_ID:-test_VID_blur}
export MODEL_PATH=${MODEL_PATH:-./exp/detect/train/train_DREB_Net_VID_blur/model_last.pth}
bash "$(dirname "$0")/evaluation_vid.sh"
