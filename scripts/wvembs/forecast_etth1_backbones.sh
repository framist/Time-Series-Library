#!/usr/bin/env bash
set -euo pipefail

# WVEmbs backbone 对照（最小预算）
# 说明：
# - 需要提前准备数据：./dataset/ETT-small/ETTh1.csv
# - 不强制 GPU；如果无 GPU，会自动回退 CPU（但会慢）
# - 对照维度：Embed = timeF vs wv_timeF；Backbone = Transformer/Informer/TimesNet/Autoformer/FEDformer

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

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
MAX_TRAIN=50
MAX_VAL=10
MAX_TEST=10

for model in Transformer Informer TimesNet Autoformer FEDformer; do
  for embed in timeF wv_timeF; do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path "${ROOT}/" \
      --data_path "${DATA}" \
      --model_id "WVEmbs_ETTh1_${model}_${embed}" \
      --model "${model}" \
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
      --top_k 5 \
      --num_kernels 6 \
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
done
