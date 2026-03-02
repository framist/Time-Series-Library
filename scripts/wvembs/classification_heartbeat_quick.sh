#!/usr/bin/env bash
set -euo pipefail

# WVEmbs 在 Classification 的最小对照（timeF vs wv_timeF）
# 数据：./dataset/Heartbeat/
#
# 注意：classification 任务内部会根据数据动态设置 seq_len/enc_in/num_class。

ROOT=./dataset/Heartbeat
DATASET=UEA
MODEL=TimesNet

EPOCHS=1
BATCH=16
WORKERS=0
MAX_TRAIN=50
MAX_VAL=10
MAX_TEST=10

for embed in timeF wv_timeF; do
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --model_id Heartbeat \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --d_model 64 \
    --d_ff 128 \
    --top_k 3 \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --learning_rate 0.001 \
    --num_workers "${WORKERS}" \
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    --itr 1 \
    --des "WVEmbs"
done
