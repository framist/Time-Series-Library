#!/usr/bin/env bash
set -euo pipefail

# WVEmbs 在 Imputation 任务的最小对照（timeF vs wv_timeF）
# 数据：ETT-small/ETTh1.csv

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

MODEL=TimesNet

SEQ_LEN=96
MASK_RATE=0.125

EPOCHS=1
BATCH=16
WORKERS=2

for embed in timeF wv_timeF; do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_Impute_ETTh1_${MODEL}_${embed}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len 0 \
    --mask_rate "${MASK_RATE}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 32 \
    --d_ff 64 \
    --e_layers 2 \
    --d_layers 1 \
    --top_k 3 \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    --itr 1 \
    --des "WVEmbs"
done

