ARCH=DREB_Net
EXP_ID=train_DREB_Net_VID_original
DATASET=visdrone_vid
INP_SHARP_OR_BLUR=SB_deblur
SHARP_DATA_DIR=/home/zhuhongxiang/DataSet/VisDrone2019-VID
BLUR_DATA_DIR=/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB
BEST_MODEL=./exp/detect/train/${EXP_ID}/model_best.pth
LAST_MODEL=./exp/detect/train/${EXP_ID}/model_last.pth
# Override these when needed, e.g. CUDA_TRAIN_DEVICE=1 MASTER_BATCH_SIZE=16 bash bash/train.sh
CUDA_TRAIN_DEVICE=${CUDA_TRAIN_DEVICE:-1}
MASTER_BATCH_SIZE=${MASTER_BATCH_SIZE:-8}
MAX_FRAMES_PER_SEQUENCE=${MAX_FRAMES_PER_SEQUENCE:-50}  #Extract the first n frames from each sequence to enable mini-batch training.
NUM_EPOCHS=${NUM_EPOCHS:-200}
VAL_INTERVALS=${VAL_INTERVALS:-1}
PRINT_ITER=${PRINT_ITER:-100}
NUM_WORKERS=${NUM_WORKERS:-8}

CUDA_VISIBLE_DEVICES=$CUDA_TRAIN_DEVICE python -u main.py \
--exp_id $EXP_ID \
--arch $ARCH \
--dataset $DATASET \
--inp_sharp_or_blur $INP_SHARP_OR_BLUR \
--sharp_data_dir $SHARP_DATA_DIR \
--blur_data_dir $BLUR_DATA_DIR \
--input_res 1024 \
--mode train \
--batch_size $MASTER_BATCH_SIZE \
--master_batch_size $MASTER_BATCH_SIZE \
--lr 1e-3  \
--num_epochs $NUM_EPOCHS \
--val_intervals $VAL_INTERVALS \
--max_frames_per_sequence $MAX_FRAMES_PER_SEQUENCE \
--print_iter $PRINT_ITER \
--num_workers $NUM_WORKERS \
--gpus $CUDA_TRAIN_DEVICE
