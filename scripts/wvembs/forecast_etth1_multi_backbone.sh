#!/usr/bin/env bash
set -euo pipefail

# Cycle 2：多 Backbone × 多 Embed（ETTh1 Forecast）
#
# 目标：
# - 验证 WVEmbs 在不同 backbone 上是否仍能带来收益（或至少不退化）
#
# 对照维度：
# - Backbone：Transformer / TimesNet / Nonstationary_Transformer / Autoformer / TimeMixer
# - Embed：timeF / wv_timeF / wv
#
# 指标口径：
# - stage0 口径：不启用 --inverse（指标在缩放空间，方便对齐上游默认）
#
# 重要说明：
# - 为避免不同实验因 early stopping 提前终止导致预算不一致，这里默认将 --patience 设为较大值
#   （PATIENCE 默认 20，可通过环境变量覆盖）。

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

SEQ_LEN=96
LABEL_LEN_DEFAULT=48
PRED_LEN=96

FACTOR=3
E_LAYERS=2
D_LAYERS=1

EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-20}"
BATCH_DEFAULT="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VAL="${MAX_VAL:--1}"
MAX_TEST="${MAX_TEST:--1}"
DES="${DES:-WVEmbsCycle2_Backbone_Forecast}"

USE_AMP="${USE_AMP:-1}"
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

run_one () {
  local model="$1"
  local embed="$2"
  shift 2

  local label_len="${LABEL_LEN_DEFAULT}"
  local batch_size="${BATCH_DEFAULT}"
  local learning_rate=""

  local d_model="${D_MODEL:-512}"
  local d_ff="${D_FF:-2048}"
  local n_heads="${N_HEADS:-8}"
  local -a EXTRA_ARGS=()

  if [[ "${model}" == "TimesNet" ]]; then
    d_model="${D_MODEL_TIMESNET:-16}"
    d_ff="${D_FF_TIMESNET:-32}"
    EXTRA_ARGS+=(--top_k 5)
  elif [[ "${model}" == "Nonstationary_Transformer" ]]; then
    d_model="${D_MODEL_NONSTAT:-128}"
    EXTRA_ARGS+=(--p_hidden_dims 256 256 --p_hidden_layers 2)
  elif [[ "${model}" == "TimeMixer" ]]; then
    # TimeMixer 官方脚本要求显式指定 down_sampling_method，否则内部会把 tensor 当 list 迭代导致维度错误
    label_len=0
    batch_size="${BATCH_TIMEMIXER:-128}"
    d_model="${D_MODEL_TIMEMIXER:-16}"
    d_ff="${D_FF_TIMEMIXER:-32}"
    learning_rate="${LR_TIMEMIXER:-0.01}"
    EXTRA_ARGS+=(
      --down_sampling_layers "${DOWN_SAMPLING_LAYERS_TIMEMIXER:-3}"
      --down_sampling_method "${DOWN_SAMPLING_METHOD_TIMEMIXER:-avg}"
      --down_sampling_window "${DOWN_SAMPLING_WINDOW_TIMEMIXER:-2}"
    )
  fi

  # WV 参数：按 embed 模式给默认采样策略（可用环境变量覆盖）
  local -a WV_ARGS=()
  if [[ "${embed}" == "wv_timeF" ]]; then
    local sampling="${WV_SAMPLING_WV_TIMEF:-jss}"
    WV_ARGS=(--wv_sampling "${sampling}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  elif [[ "${embed}" == "wv" ]]; then
    local sampling="${WV_SAMPLING_WV:-iss}"
    WV_ARGS=(--wv_sampling "${sampling}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  fi

  local -a LR_ARGS=()
  if [[ -n "${learning_rate}" ]]; then
    LR_ARGS=(--learning_rate "${learning_rate}")
  fi

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
    --label_len "${label_len}" \
    --pred_len "${PRED_LEN}" \
    --factor "${FACTOR}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model "${d_model}" \
    --d_ff "${d_ff}" \
    --n_heads "${n_heads}" \
    --e_layers "${E_LAYERS}" \
    --d_layers "${D_LAYERS}" \
    --train_epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${batch_size}" \
    --num_workers "${WORKERS}" \
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    "${LR_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" \
    "$@"
}

for model in Transformer TimesNet Nonstationary_Transformer Autoformer TimeMixer; do
  for embed in timeF wv_timeF wv; do
    echo "[RUN] model=${model} embed=${embed}"
    run_one "${model}" "${embed}"
  done
done
