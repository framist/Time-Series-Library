#!/usr/bin/env bash
set -euo pipefail

# WVEmbs masking 消融（最小预算）
# 说明：
# - 需要提前准备数据：./dataset/ETT-small/ETTh1.csv
# - 仅在 WVEmbs（wv_*）下生效；timeF 基线不会读取 wv_mask_* 参数

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

MASK_PROB=0.75
PHI_MAX=0.39269908169872414 # pi/8

for mask_type in none zero arcsine phase_rotate; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_${EMBED}_mask_${mask_type}" \
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
    --wv_mask_prob "${MASK_PROB}" \
    --wv_mask_type "${mask_type}" \
    --wv_mask_phi_max "${PHI_MAX}" \
    --itr 1 \
    --des "WVEmbsMask"
done

# dlow_limited 示例：只掩码更低频尾部（dlow_min 越大，掩码覆盖越小）
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "${ROOT}/" \
  --data_path "${DATA}" \
  --model_id "WVEmbs_ETTh1_${MODEL}_${EMBED}_phase_rotate_dlow_limited" \
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
  --wv_mask_prob "${MASK_PROB}" \
  --wv_mask_type phase_rotate \
  --wv_mask_phi_max "${PHI_MAX}" \
  --wv_mask_dlow_min 24 \
  --itr 1 \
  --des "WVEmbsMask"

