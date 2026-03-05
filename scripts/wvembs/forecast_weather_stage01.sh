#!/usr/bin/env bash
set -euo pipefail

ROOT=./dataset/weather
DATA=weather.csv
DATASET=custom
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
DES="${DES:-WVEmbsCycle3_Weather}"

USE_AMP="${USE_AMP:-1}"
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"

RUN_STAGE0="${RUN_STAGE0:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_TIMEMIXER="${RUN_TIMEMIXER:-1}"

PRIOR_SCALE_DEFAULT="2040.14 69.6 618.26 41 200 111.34 48.32 84.2 30.8 49.06 2637.04 19998 45.8 720 22.4 1200 2230.58 4263.52 19998 98.18 19998"
PRIOR_SCALE="${PRIOR_SCALE:-${PRIOR_SCALE_DEFAULT}}"
PRIOR_OFFSET="${PRIOR_OFFSET:-0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

run_transformer() {
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
    --model_id "WVEmbs_Weather_${MODEL}_${tag}_${embed}" \
    --model "${MODEL}" \
    --data "${DATASET}" \
    --features M \
    --freq t \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${PRED_LEN}" \
    --factor "${FACTOR}" \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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

run_timemixer_stage0() {
  local embed="$1"

  local d_model_tm="${D_MODEL_TIMEMIXER:-16}"
  local d_ff_tm="${D_FF_TIMEMIXER:-32}"
  local batch_tm="${BATCH_TIMEMIXER:-128}"
  local lr_tm="${LR_TIMEMIXER:-0.01}"
  local down_layers_tm="${DOWN_SAMPLING_LAYERS_TIMEMIXER:-3}"
  local down_method_tm="${DOWN_SAMPLING_METHOD_TIMEMIXER:-avg}"
  local down_window_tm="${DOWN_SAMPLING_WINDOW_TIMEMIXER:-2}"

  local -a WV_ARGS=()
  if [[ "${embed}" == "wv_timeF" ]]; then
    WV_ARGS=(--wv_sampling jss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  elif [[ "${embed}" == "wv" ]]; then
    WV_ARGS=(--wv_sampling iss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_Weather_TimeMixer_S0_${embed}" \
    --model TimeMixer \
    --data "${DATASET}" \
    --features M \
    --freq t \
    --seq_len "${SEQ_LEN}" \
    --label_len 0 \
    --pred_len "${PRED_LEN}" \
    --factor "${FACTOR}" \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model "${d_model_tm}" \
    --d_ff "${d_ff_tm}" \
    --n_heads "${N_HEADS}" \
    --e_layers "${E_LAYERS}" \
    --d_layers "${D_LAYERS}" \
    --down_sampling_layers "${down_layers_tm}" \
    --down_sampling_method "${down_method_tm}" \
    --down_sampling_window "${down_window_tm}" \
    --learning_rate "${lr_tm}" \
    --train_epochs "${EPOCHS}" \
    --batch_size "${batch_tm}" \
    --num_workers "${WORKERS}" \
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    --itr 1 \
    --des "${DES}_TimeMixer"
}

if [[ "${RUN_STAGE0}" == "1" ]]; then
  run_transformer "S0" "timeF" "0"
  run_transformer "S0" "wv_timeF" "0" --wv_sampling jss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}"
  run_transformer "S0" "wv" "0" --wv_sampling iss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}"
fi

if [[ "${RUN_STAGE1}" == "1" ]]; then
  run_transformer "D_std" "timeF" "1"
  run_transformer "A_noscale" "timeF" "1" --no_scale
  run_transformer "B_noscale" "wv" "1" --no_scale --wv_sampling iss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}"

  read -r -a PRIOR_SCALE_ARR <<< "${PRIOR_SCALE}"
  read -r -a PRIOR_OFFSET_ARR <<< "${PRIOR_OFFSET}"
  run_transformer "C_prior" "wv" "1" \
    --scale_mode prior \
    --prior_scale "${PRIOR_SCALE_ARR[@]}" \
    --prior_offset "${PRIOR_OFFSET_ARR[@]}" \
    --wv_sampling jss \
    --wv_jss_std "${WV_JSS_STD}" \
    --wv_base "${WV_BASE}"
fi

if [[ "${RUN_TIMEMIXER}" == "1" ]]; then
  run_timemixer_stage0 "timeF"
  run_timemixer_stage0 "wv_timeF"
  run_timemixer_stage0 "wv"
fi
