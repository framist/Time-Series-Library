#!/usr/bin/env bash
set -euo pipefail

# 阶段 0：ETTm1 + TimesNet 的 embed 对照（对齐上游 ETTh1 官方脚本的超参口径）
#
# 对照组：
# - timeF：原始基线（TokenEmbedding + TimeFeatureEmbedding）
# - wv_timeF：消融（值 WVEmbs + 时间仍用 TimeFeatureEmbedding）
# - wv：统一模式（值 + 时间通道一起进入 WVEmbs）
#
# 指标口径：不启用 `--inverse`（指标在缩放后的无量纲空间）。

ROOT=./dataset/ETT-small
DATA=ETTm1.csv
DATASET=ETTm1
MODEL=TimesNet

SEQ_LEN=96
MASK_RATE=0.125

E_LAYERS=2
TOP_K=3
FACTOR=3

D_MODEL="${D_MODEL:-16}"
D_FF="${D_FF:-32}"
LR="${LR:-0.001}"

EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VAL="${MAX_VAL:--1}"
MAX_TEST="${MAX_TEST:--1}"
DES="${DES:-WVEmbsFinal}"

USE_AMP="${USE_AMP:-1}"
WV_SAMPLING="${WV_SAMPLING:-jss}"
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi
WV_ARGS=(--wv_sampling "${WV_SAMPLING}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")

for embed in timeF wv_timeF wv; do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTm1_${MODEL}_stage0_${embed}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len 0 \
    --mask_rate "${MASK_RATE}" \
    --factor "${FACTOR}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model "${D_MODEL}" \
    --d_ff "${D_FF}" \
    --e_layers "${E_LAYERS}" \
    --d_layers 1 \
    --top_k "${TOP_K}" \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --learning_rate "${LR}" \
    --num_workers "${WORKERS}" \
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
done
