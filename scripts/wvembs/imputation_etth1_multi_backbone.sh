#!/usr/bin/env bash
set -euo pipefail

# Cycle 2：多 Backbone × 多 Embed（ETTh1 Imputation）
#
# 对照维度：
# - Backbone：Transformer / TimesNet / Nonstationary_Transformer / Autoformer / TimeMixer
# - Embed：timeF / wv_timeF / wv
#
# 指标口径：
# - stage0 口径：不启用 --inverse（指标在缩放空间，方便对齐上游默认）
#
# 说明：
# - imputation 任务会向模型 forward 传入 mask（见 exp/exp_imputation.py），本脚本选择的模型均支持该签名。
# - 同样默认将 --patience 设为较大值，避免不同实验提前 early stop 导致预算不一致。

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

SEQ_LEN=96
MASK_RATE=0.125

FACTOR=3
E_LAYERS=2
D_LAYERS=1
TOP_K_DEFAULT="${TOP_K:-3}"
NUM_KERNELS_DEFAULT="${NUM_KERNELS:-6}"

EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VAL="${MAX_VAL:--1}"
MAX_TEST="${MAX_TEST:--1}"
DES="${DES:-WVEmbsCycle2_Backbone_Imputation}"

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

  local batch_size="${BATCH}"
  local learning_rate="${LR:-0.001}"
  local d_model="${D_MODEL:-128}"
  local d_ff="${D_FF:-128}"
  local n_heads="${N_HEADS:-8}"
  local top_k="${TOP_K_DEFAULT}"
  local num_kernels="${NUM_KERNELS_DEFAULT}"
  local -a EXTRA_ARGS=()

  if [[ "${model}" == "TimesNet" ]]; then
    d_model="${D_MODEL_TIMESNET:-16}"
    d_ff="${D_FF_TIMESNET:-32}"
    top_k="${TOP_K_TIMESNET:-3}"
  elif [[ "${model}" == "Nonstationary_Transformer" ]]; then
    d_model="${D_MODEL_NONSTAT:-128}"
    d_ff="${D_FF_NONSTAT:-128}"
    EXTRA_ARGS+=(--p_hidden_dims 256 256 --p_hidden_layers 2)
  elif [[ "${model}" == "Autoformer" ]]; then
    d_model="${D_MODEL_AUTOFORMER:-128}"
    d_ff="${D_FF_AUTOFORMER:-128}"
  elif [[ "${model}" == "Transformer" ]]; then
    d_model="${D_MODEL_TRANSFORMER:-128}"
    d_ff="${D_FF_TRANSFORMER:-128}"
  elif [[ "${model}" == "TimeMixer" ]]; then
    # TimeMixer 内部同样依赖 down_sampling_method，否则会把 tensor 当 list 迭代导致维度错误
    d_model="${D_MODEL_TIMEMIXER:-16}"
    d_ff="${D_FF_TIMEMIXER:-32}"
    learning_rate="${LR_TIMEMIXER:-0.001}"
    EXTRA_ARGS+=(
      --down_sampling_layers "${DOWN_SAMPLING_LAYERS_TIMEMIXER:-3}"
      --down_sampling_method "${DOWN_SAMPLING_METHOD_TIMEMIXER:-avg}"
      --down_sampling_window "${DOWN_SAMPLING_WINDOW_TIMEMIXER:-2}"
    )
  fi

  local -a WV_ARGS=()
  if [[ "${embed}" == "wv_timeF" ]]; then
    local sampling="${WV_SAMPLING_WV_TIMEF:-iss}"
    WV_ARGS=(--wv_sampling "${sampling}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  elif [[ "${embed}" == "wv" ]]; then
    local sampling="${WV_SAMPLING_WV:-iss}"
    WV_ARGS=(--wv_sampling "${sampling}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  fi

  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${model}_${embed}" \
    --mask_rate "${MASK_RATE}" \
    --model "${model}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len 0 \
    --factor "${FACTOR}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model "${d_model}" \
    --d_ff "${d_ff}" \
    --n_heads "${n_heads}" \
    --e_layers "${E_LAYERS}" \
    --d_layers "${D_LAYERS}" \
    --top_k "${top_k}" \
    --num_kernels "${num_kernels}" \
    --train_epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${batch_size}" \
    --learning_rate "${learning_rate}" \
    --num_workers "${WORKERS}" \
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
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
