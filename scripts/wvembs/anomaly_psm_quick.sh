#!/usr/bin/env bash
set -euo pipefail

# WVEmbs 在 Anomaly Detection 的最小对照（timeF vs wv_timeF）
# 数据：./dataset/PSM

ROOT=./dataset/PSM
DATASET=PSM
MODEL=TimesNet

SEQ_LEN=100

EPOCHS=1
BATCH=32
WORKERS=2
MAX_TRAIN=50
MAX_VAL=10
MAX_TEST=10

for embed in timeF wv_timeF; do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --model_id "WVEmbs_AD_${DATASET}_${MODEL}_${embed}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --pred_len 0 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 2 \
    --enc_in 25 \
    --c_out 25 \
    --anomaly_ratio 1.0 \
    --top_k 3 \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    --itr 1 \
    --des "WVEmbs"
done
