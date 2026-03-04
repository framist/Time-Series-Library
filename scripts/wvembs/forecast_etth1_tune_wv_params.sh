#!/usr/bin/env bash
set -euo pipefail

# ETTh1 Forecast：WV 参数最小调优（针对 Cycle2 中退化明显的 backbone）
#
# 目标：
# - 只调整 WVEmbs 的采样策略/尺度相关参数（wv_sampling / wv_jss_std / wv_base）
# - 观察能否显著缓解（或消除）Cycle2 多-backbone 对照中的退化现象
#
# 说明：
# - 这里不追求“全网格搜索”，只做少量代表性组合
# - 其它超参尽量与 `scripts/wvembs/forecast_etth1_multi_backbone.sh` 对齐
#
# 输出：
# - 结果将写入 `result_long_term_forecast.txt`，可用 `--des` 过滤提取。

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
BATCH_DEFAULT="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
PATIENCE="${PATIENCE:-20}"

MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VAL="${MAX_VAL:--1}"
MAX_TEST="${MAX_TEST:--1}"

DES="${DES:-WVEmbsCycle2_ForecastTune_20260305}"

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

run_one () {
  local model="$1"
  local embed="$2"
  local tag="$3"
  local wv_sampling="$4"
  local wv_jss_std="$5"
  local wv_base="$6"
  shift 6

  local label_len="${LABEL_LEN_DEFAULT}"
  local batch_size="${BATCH_DEFAULT}"
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
  fi

  local -a WV_ARGS=()
  if [[ "${embed}" == "wv_timeF" || "${embed}" == "wv" ]]; then
    WV_ARGS=(--wv_sampling "${wv_sampling}" --wv_jss_std "${wv_jss_std}" --wv_base "${wv_base}")
  fi

  echo "[RUN] model=${model} embed=${embed} tag=${tag} wv_sampling=${wv_sampling} wv_jss_std=${wv_jss_std} wv_base=${wv_base}"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${model}_${embed}_${tag}" \
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
    "${EXTRA_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" \
    "$@"
}

# TimesNet：尝试替换 wv_timeF 的采样策略 + 调小 base
run_one "TimesNet" "wv_timeF" "iss_b1e4" "iss" "1.0" "10000.0"
run_one "TimesNet" "wv" "iss_b1e3" "iss" "1.0" "1000.0"
run_one "TimesNet" "wv" "jss025_b1e4" "jss" "0.25" "10000.0"

# Nonstationary_Transformer：这是 Cycle2 中退化最明显的模型，尝试更多组合
run_one "Nonstationary_Transformer" "wv_timeF" "jss025_b1e4" "jss" "0.25" "10000.0"
run_one "Nonstationary_Transformer" "wv_timeF" "iss_b1e4" "iss" "1.0" "10000.0"
run_one "Nonstationary_Transformer" "wv" "iss_b1e3" "iss" "1.0" "1000.0"
run_one "Nonstationary_Transformer" "wv" "jss025_b1e4" "jss" "0.25" "10000.0"

# Autoformer：优先尝试把 wv_timeF 改为 iss + 调小 base
run_one "Autoformer" "wv_timeF" "iss_b1e4" "iss" "1.0" "10000.0"
run_one "Autoformer" "wv" "iss_b1e3" "iss" "1.0" "1000.0"

echo "[DONE] forecast tune finished. DES=${DES}"

