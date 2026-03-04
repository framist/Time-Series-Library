#!/usr/bin/env bash
set -euo pipefail

# 阶段 1：Classification（Heartbeat / UEA + TimesNet）scale_mode 对照（对齐上游官方脚本超参）
#
# 对照维度：
# - scale_mode ∈ {standard, none}
# - embed ∈ {timeF, wv_timeF}
#
# 注意：
# - UEA loader 使用 `--model_id` 作为数据集文件前缀（Heartbeat），不要在 model_id 里塞实验标签；
#   实验标签请放到 `--des`（并在脚本中包含 scale_mode 信息避免覆盖）。
# - classification 任务内部会根据数据动态设置 seq_len/enc_in/num_class。

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
  for embed in timeF wv_timeF; do
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
      --des "${DES}_${scale_mode}"
  done
done
