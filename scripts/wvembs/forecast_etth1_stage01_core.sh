#!/usr/bin/env bash
set -euo pipefail

# 阶段 0/1 核心对照（ETTh1 + Transformer，对齐上游官方脚本超参）
#
# 四组：
# - A：--no_scale + timeF
# - B：--no_scale + wv（统一时间入通道）
# - C：--scale_mode prior + wv（需要先验尺度）
# - D：默认 StandardScaler + timeF
#
# 指标口径：
# - 本脚本默认开启 `--inverse`，确保在 standard/prior/none 三种缩放模式下，指标都回到原始物理量尺度，便于横向比较。
#
# prior 参数说明：
# - 需要设置环境变量 PRIOR_SCALE（标量或每通道一个值，例如 7 维）
#   例：PRIOR_SCALE="100 100 100 100 100 100 100" bash scripts/wvembs/forecast_etth1_stage01_core.sh
# - 可选 PRIOR_OFFSET（默认 0）

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1
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

run_one () {
  local group="$1"
  local embed="$2"
  shift 2

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_${group}_${embed}" \
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
    --inverse \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" \
    "$@"
}

# D：默认 StandardScaler + timeF（传统基线）
run_one "D_std" "timeF"

# A：no_scale + timeF
run_one "A_noscale" "timeF" --no_scale

# B：no_scale + wv（统一）
run_one "B_noscale" "wv" --no_scale

# C：prior + wv
if [[ -z "${PRIOR_SCALE:-}" ]]; then
  echo "[SKIP] group C (scale_mode=prior): 请先设置 PRIOR_SCALE 环境变量" >&2
  exit 0
fi

read -r -a PRIOR_SCALE_ARR <<< "${PRIOR_SCALE}"
if [[ -n "${PRIOR_OFFSET:-}" ]]; then
  read -r -a PRIOR_OFFSET_ARR <<< "${PRIOR_OFFSET}"
  run_one "C_prior" "wv" --scale_mode prior --prior_scale "${PRIOR_SCALE_ARR[@]}" --prior_offset "${PRIOR_OFFSET_ARR[@]}"
else
  run_one "C_prior" "wv" --scale_mode prior --prior_scale "${PRIOR_SCALE_ARR[@]}"
fi
