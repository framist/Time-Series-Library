#!/usr/bin/env bash
set -euo pipefail

# Cycle 1 补齐：Classification（Heartbeat / UEA + TimesNet）补充 `--embed wv`（统一模式）
#
# 说明（重要）：
# - classification 任务通常没有显式时间特征（x_mark=None）
# - 本仓库 `DataEmbedding` 在 `embed=wv` 且 x_mark=None 时会自动补零时间特征，因此该实验可跑通；
#   但它更像是“值 WVEmbs + 额外零时间通道”的对照项，用于补齐统一模式在该任务上的表现。
#
# 注意：
# - UEA loader 使用 `--model_id` 作为数据集文件前缀（Heartbeat），不要在 model_id 里塞实验标签；
#   实验标签请放到 `--des`。

ROOT=./dataset/Heartbeat
DATASET=UEA
MODEL=TimesNet

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-0}"
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
    --task_name classification \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --model_id Heartbeat \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --e_layers 3 \
    --d_model 16 \
    --d_ff 32 \
    --top_k 1 \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH}" \
    --learning_rate 0.001 \
    --patience 10 \
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
    --des "${DES}_${scale_mode}_${embed}"
done

