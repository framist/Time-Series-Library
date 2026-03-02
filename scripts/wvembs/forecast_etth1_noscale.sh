#!/usr/bin/env bash
set -euo pipefail

# WVEmbs 分布无关倾向实验：关闭数据集级 StandardScaler（--no_scale）
# 说明：
# - 需要提前准备数据：./dataset/ETT-small/ETTh1.csv
# - 关闭 scale 后，timeF 基线可能更难优化；WVEmbs 理论上更稳健

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

MODEL=Transformer

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

for embed in timeF wv_timeF; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_${embed}_no_scale" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --no_scale \
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
    --embed "${embed}" \
    --itr 1 \
    --des "WVEmbsNoScale"
done

