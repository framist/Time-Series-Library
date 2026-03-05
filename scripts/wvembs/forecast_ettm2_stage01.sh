#!/usr/bin/env bash
set -euo pipefail

ROOT=./dataset/ETT-small
DATA=ETTm2.csv
DATASET=ETTm2
MODEL=Transformer

SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96

E_LAYERS=2
D_LAYERS=1
FACTOR=3

D_MODEL="${D_MODEL:-512}"
D_FF="${D_FF:-2048}"
N_HEADS="${N_HEADS:-8}"

EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VAL="${MAX_VAL:--1}"
MAX_TEST="${MAX_TEST:--1}"
DES="${DES:-WVEmbsCycle3_ETTm2}"

USE_AMP="${USE_AMP:-1}"
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"

RUN_STAGE0="${RUN_STAGE0:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"

PRIOR_SCALE_DEFAULT="215.786 72.878 186.46 59.616 34.436 62.924 116.875"
PRIOR_SCALE="${PRIOR_SCALE:-${PRIOR_SCALE_DEFAULT}}"
PRIOR_OFFSET="${PRIOR_OFFSET:-0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

run_one() {
  local tag="$1"
  local embed="$2"
  local use_inverse="$3"
  shift 3

  local -a EXTRA_ARGS=("$@")
  local -a INV_ARGS=()
  if [[ "${use_inverse}" == "1" ]]; then
    INV_ARGS=(--inverse)
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTm2_${MODEL}_${tag}_${embed}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${PRED_LEN}" \
    --factor "${FACTOR}" \
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
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${INV_ARGS[@]}" \
    "${AMP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
}

if [[ "${RUN_STAGE0}" == "1" ]]; then
  run_one "S0" "timeF" "0"
  run_one "S0" "wv_timeF" "0" --wv_sampling jss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}"
  run_one "S0" "wv" "0" --wv_sampling iss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}"
fi

if [[ "${RUN_STAGE1}" == "1" ]]; then
  run_one "D_std" "timeF" "1"
  run_one "A_noscale" "timeF" "1" --no_scale
  run_one "B_noscale" "wv" "1" --no_scale --wv_sampling iss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}"

  read -r -a PRIOR_SCALE_ARR <<< "${PRIOR_SCALE}"
  read -r -a PRIOR_OFFSET_ARR <<< "${PRIOR_OFFSET}"
  run_one "C_prior" "wv" "1" \
    --scale_mode prior \
    --prior_scale "${PRIOR_SCALE_ARR[@]}" \
    --prior_offset "${PRIOR_OFFSET_ARR[@]}" \
    --wv_sampling jss \
    --wv_jss_std "${WV_JSS_STD}" \
    --wv_base "${WV_BASE}"
fi
