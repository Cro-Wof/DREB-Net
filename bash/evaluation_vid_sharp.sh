#!/usr/bin/env bash
set -euo pipefail

export INP_SHARP_OR_BLUR=sharp
export EXP_ID=${EXP_ID:-test_VID_sharp}
export MODEL_PATH=${MODEL_PATH:-./exp/detect/train/train_DREB_Net_VID_sharp/model_best.pth}
bash "$(dirname "$0")/evaluation_vid.sh"
