#!/usr/bin/env bash
set -euo pipefail

# Cycle 1 补齐：Anomaly Detection（PSM + TimesNet）补充 `--embed wv`（统一模式）
#
# 说明（重要）：
# - anomaly/classification 任务通常没有显式时间特征（x_mark=None）
# - 本仓库的 `DataEmbedding` 在 `embed=wv` 且 x_mark=None 时会自动补零时间特征，因此该实验可跑通；
#   但它更像是“值 WVEmbs + 额外零时间通道”的对照项，用于补齐统一模式在该任务上的表现。
#
# 对照维度：
# - scale_mode ∈ {standard, none}
# - embed 固定为 wv

ROOT=./dataset/PSM
DATASET=PSM
MODEL=TimesNet

SEQ_LEN=100

EPOCHS="${EPOCHS:-3}"
BATCH="${BATCH:-128}"
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

for scale_mode in standard none; do
  embed=wv
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --model_id "WVEmbs_AD_${DATASET}_${MODEL}_${embed}_${scale_mode}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --pred_len 0 \
    --d_model 64 \
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
    --scale_mode "${scale_mode}" \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
done

