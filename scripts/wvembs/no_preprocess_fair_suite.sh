#!/usr/bin/env bash
set -euo pipefail

# 无预处理公平对照套件
#
# 目标：
# - 统一关闭数据侧预处理（scale_mode=none）
# - 固定同一 backbone / 训练协议 / 数据划分
# - 仅替换输入层：timeF（原始 TSLib）、linear（线性统一前端）、wv（WVEmbs 统一前端）
#
# 默认覆盖：
# - Forecast：Transformer + ETTh1 / ETTh2 / Weather
# - Imputation：TimesNet + ETTh1
# - Anomaly Detection：TimesNet + PSM
# - Classification：TimesNet + Heartbeat
#
# 用法示例：
#   bash scripts/wvembs/no_preprocess_fair_suite.sh
#   RUN_FORECAST=1 RUN_IMPUTATION=0 RUN_ANOMALY=0 RUN_CLASSIFICATION=0 bash scripts/wvembs/no_preprocess_fair_suite.sh
#   PRED_LENS="96 336" WV_EXTRAP_MODE=scale WV_EXTRAP_SCALE=5.0 bash scripts/wvembs/no_preprocess_fair_suite.sh

DES="${DES:-WVEmbsNoPreprocessFair}"
USE_AMP="${USE_AMP:-1}"

RUN_FORECAST="${RUN_FORECAST:-1}"
RUN_IMPUTATION="${RUN_IMPUTATION:-1}"
RUN_ANOMALY="${RUN_ANOMALY:-1}"
RUN_CLASSIFICATION="${RUN_CLASSIFICATION:-1}"

RUN_ETTH1="${RUN_ETTH1:-1}"
RUN_ETTH2="${RUN_ETTH2:-1}"
RUN_WEATHER="${RUN_WEATHER:-1}"

PRED_LENS="${PRED_LENS:-96 192 336 720}"
read -r -a PRED_LENS_ARR <<< "${PRED_LENS}"

FAIR_EMBEDS=(timeF linear wv)

WV_SAMPLING="${WV_SAMPLING:-iss}"
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"
WV_EXTRAP_MODE="${WV_EXTRAP_MODE:-direct}"
WV_EXTRAP_SCALE="${WV_EXTRAP_SCALE:-1.0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

build_wv_args() {
  local -n out_ref="$1"
  out_ref+=(--wv_sampling "${WV_SAMPLING}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  if [[ "${WV_EXTRAP_MODE}" != "direct" || "${WV_EXTRAP_SCALE}" != "1.0" ]]; then
    out_ref+=(--wv_extrap_mode "${WV_EXTRAP_MODE}" --wv_extrap_scale "${WV_EXTRAP_SCALE}")
  fi
}

run_forecast_one() {
  local dataset_name="$1"
  local root_path="$2"
  local data_path="$3"
  local data_flag="$4"
  local enc_in="$5"
  local freq="$6"
  local pred_len="$7"
  local embed="$8"

  local tag
  case "${embed}" in
    timeF) tag="raw_timeF" ;;
    linear) tag="linear" ;;
    wv) tag="wv" ;;
    *) tag="${embed}" ;;
  esac

  local -a extra_args=(--scale_mode none)
  if [[ "${embed}" == "wv" ]]; then
    build_wv_args extra_args
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${root_path}/" \
    --data_path "${data_path}" \
    --model_id "NoPrepFair_${dataset_name}_Transformer_${tag}_pl${pred_len}" \
    --model Transformer \
    --data "${data_flag}" \
    --features M \
    --freq "${freq}" \
    --seq_len "${FORECAST_SEQ_LEN:-96}" \
    --label_len "${FORECAST_LABEL_LEN:-48}" \
    --pred_len "${pred_len}" \
    --factor "${FORECAST_FACTOR:-3}" \
    --enc_in "${enc_in}" \
    --dec_in "${enc_in}" \
    --c_out "${enc_in}" \
    --d_model "${FORECAST_D_MODEL:-512}" \
    --d_ff "${FORECAST_D_FF:-2048}" \
    --n_heads "${FORECAST_N_HEADS:-8}" \
    --e_layers "${FORECAST_E_LAYERS:-2}" \
    --d_layers "${FORECAST_D_LAYERS:-1}" \
    --train_epochs "${FORECAST_EPOCHS:-10}" \
    --batch_size "${FORECAST_BATCH:-32}" \
    --num_workers "${FORECAST_WORKERS:-8}" \
    --max_train_steps "${FORECAST_MAX_TRAIN:--1}" \
    --max_val_steps "${FORECAST_MAX_VAL:--1}" \
    --max_test_steps "${FORECAST_MAX_TEST:--1}" \
    --checkpoints ./checkpoints_wvembs/ \
    --inverse \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${extra_args[@]}" \
    --itr 1 \
    --des "${DES}"
}

run_imputation_one() {
  local embed="$1"
  local tag
  case "${embed}" in
    timeF) tag="raw_timeF" ;;
    linear) tag="linear" ;;
    wv) tag="wv" ;;
    *) tag="${embed}" ;;
  esac

  local -a extra_args=(--scale_mode none)
  if [[ "${embed}" == "wv" ]]; then
    build_wv_args extra_args
  fi

  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id "NoPrepFair_ETTh1_TimesNet_${tag}" \
    --model TimesNet \
    --data ETTh1 \
    --features M \
    --seq_len "${IMPUTATION_SEQ_LEN:-96}" \
    --label_len 0 \
    --pred_len 0 \
    --mask_rate "${IMPUTATION_MASK_RATE:-0.125}" \
    --factor "${IMPUTATION_FACTOR:-3}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model "${IMPUTATION_D_MODEL:-16}" \
    --d_ff "${IMPUTATION_D_FF:-32}" \
    --e_layers "${IMPUTATION_E_LAYERS:-2}" \
    --d_layers 1 \
    --top_k "${IMPUTATION_TOP_K:-3}" \
    --learning_rate "${IMPUTATION_LR:-0.001}" \
    --train_epochs "${IMPUTATION_EPOCHS:-10}" \
    --batch_size "${IMPUTATION_BATCH:-16}" \
    --num_workers "${IMPUTATION_WORKERS:-8}" \
    --max_train_steps "${IMPUTATION_MAX_TRAIN:--1}" \
    --max_val_steps "${IMPUTATION_MAX_VAL:--1}" \
    --max_test_steps "${IMPUTATION_MAX_TEST:--1}" \
    --checkpoints ./checkpoints_wvembs/ \
    --inverse \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${extra_args[@]}" \
    --itr 1 \
    --des "${DES}"
}

run_anomaly_one() {
  local embed="$1"
  local tag
  case "${embed}" in
    timeF) tag="raw_timeF" ;;
    linear) tag="linear" ;;
    wv) tag="wv" ;;
    *) tag="${embed}" ;;
  esac

  local -a extra_args=(--scale_mode none)
  if [[ "${embed}" == "wv" ]]; then
    build_wv_args extra_args
  fi

  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/PSM/ \
    --model_id "NoPrepFair_PSM_TimesNet_${tag}" \
    --model TimesNet \
    --data PSM \
    --features M \
    --seq_len "${ANOMALY_SEQ_LEN:-100}" \
    --pred_len 0 \
    --d_model "${ANOMALY_D_MODEL:-64}" \
    --d_ff "${ANOMALY_D_FF:-64}" \
    --e_layers "${ANOMALY_E_LAYERS:-2}" \
    --enc_in 25 \
    --c_out 25 \
    --anomaly_ratio 1.0 \
    --top_k "${ANOMALY_TOP_K:-3}" \
    --learning_rate "${ANOMALY_LR:-0.0001}" \
    --train_epochs "${ANOMALY_EPOCHS:-3}" \
    --batch_size "${ANOMALY_BATCH:-128}" \
    --num_workers "${ANOMALY_WORKERS:-8}" \
    --max_train_steps "${ANOMALY_MAX_TRAIN:--1}" \
    --max_val_steps "${ANOMALY_MAX_VAL:--1}" \
    --max_test_steps "${ANOMALY_MAX_TEST:--1}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${extra_args[@]}" \
    --itr 1 \
    --des "${DES}"
}

run_classification_one() {
  local embed="$1"
  local tag
  case "${embed}" in
    timeF) tag="raw_timeF" ;;
    linear) tag="linear" ;;
    wv) tag="wv" ;;
    *) tag="${embed}" ;;
  esac

  local -a extra_args=(--scale_mode none)
  if [[ "${embed}" == "wv" ]]; then
    build_wv_args extra_args
  fi

  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/Heartbeat/ \
    --model_id Heartbeat \
    --model TimesNet \
    --data UEA \
    --e_layers "${CLASSIFICATION_E_LAYERS:-3}" \
    --d_model "${CLASSIFICATION_D_MODEL:-16}" \
    --d_ff "${CLASSIFICATION_D_FF:-32}" \
    --top_k "${CLASSIFICATION_TOP_K:-1}" \
    --learning_rate "${CLASSIFICATION_LR:-0.001}" \
    --patience "${CLASSIFICATION_PATIENCE:-10}" \
    --train_epochs "${CLASSIFICATION_EPOCHS:-30}" \
    --batch_size "${CLASSIFICATION_BATCH:-16}" \
    --num_workers "${CLASSIFICATION_WORKERS:-0}" \
    --max_train_steps "${CLASSIFICATION_MAX_TRAIN:--1}" \
    --max_val_steps "${CLASSIFICATION_MAX_VAL:--1}" \
    --max_test_steps "${CLASSIFICATION_MAX_TEST:--1}" \
    --checkpoints ./checkpoints_wvembs/ \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${extra_args[@]}" \
    --itr 1 \
    --des "${DES}_${tag}"
}

if [[ "${RUN_FORECAST}" == "1" ]]; then
  if [[ "${RUN_ETTH1}" == "1" ]]; then
    for pred_len in "${PRED_LENS_ARR[@]}"; do
      for embed in "${FAIR_EMBEDS[@]}"; do
        run_forecast_one ETTh1 ./dataset/ETT-small ETTh1.csv ETTh1 7 h "${pred_len}" "${embed}"
      done
    done
  fi

  if [[ "${RUN_ETTH2}" == "1" ]]; then
    for pred_len in "${PRED_LENS_ARR[@]}"; do
      for embed in "${FAIR_EMBEDS[@]}"; do
        run_forecast_one ETTh2 ./dataset/ETT-small ETTh2.csv ETTh2 7 h "${pred_len}" "${embed}"
      done
    done
  fi

  if [[ "${RUN_WEATHER}" == "1" ]]; then
    for pred_len in "${PRED_LENS_ARR[@]}"; do
      for embed in "${FAIR_EMBEDS[@]}"; do
        run_forecast_one Weather ./dataset/weather weather.csv custom 21 t "${pred_len}" "${embed}"
      done
    done
  fi
fi

if [[ "${RUN_IMPUTATION}" == "1" ]]; then
  for embed in "${FAIR_EMBEDS[@]}"; do
    run_imputation_one "${embed}"
  done
fi

if [[ "${RUN_ANOMALY}" == "1" ]]; then
  for embed in "${FAIR_EMBEDS[@]}"; do
    run_anomaly_one "${embed}"
  done
fi

if [[ "${RUN_CLASSIFICATION}" == "1" ]]; then
  for embed in "${FAIR_EMBEDS[@]}"; do
    run_classification_one "${embed}"
  done
fi

echo "[DONE] no_preprocess_fair_suite finished. DES=${DES}"
