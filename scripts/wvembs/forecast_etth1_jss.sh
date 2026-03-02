#!/usr/bin/env bash
set -euo pipefail

# WVEmbs ISS vs JSS（最小预算）
# - ISS：逐变量独立谱采样（wv_sampling=iss）
# - JSS：联合谱采样（wv_sampling=jss）
#
# 数据：./dataset/ETT-small/ETTh1.csv

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

MODEL=Transformer
EMBED=wv_timeF

SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96

D_MODEL=64
D_FF=128
N_HEADS=4
E_LAYERS=2
D_LAYERS=1

EPOCHS=1
BATCH=8
WORKERS=2

for sampling in iss jss; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_${EMBED}_${sampling}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${PRED_LEN}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --n_heads "${N_HEADS}" \
    --e_layers "${E_LAYERS}" \
    --d_layers "${D_LAYERS}" \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --num_workers "${WORKERS}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${EMBED}" \
    --wv_sampling "${sampling}" \
    --wv_jss_std 1.0 \
    --itr 1 \
    --des "WVEmbsJSS"
done

