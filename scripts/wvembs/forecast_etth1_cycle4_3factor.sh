#!/usr/bin/env bash
set -euo pipefail

# Cycle 4 Step 1: 三因素联合扫描（ETTh1 Forecast, Transformer）
#
# 扫描维度：
#   wv_sampling ∈ {iss, jss}
#   wv_jss_std  ∈ {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}（仅 jss 生效）
#   scale_mode  ∈ {standard, none, prior}
#
# 口径：embed=wv（统一模式），inverse=True（跨 scale_mode 可比）
# 共 3 × (1 ISS + 6 JSS) = 21 组
#
# 预估耗时：~42 min（单 GPU 串行）

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

DES="${DES:-WVEmbsCycle4_3factor}"

# 先验尺度（ETTh1, 7通道, max(|x_train|)×2）
PRIOR_SCALE="47.288 17.682 42.57 13.788 15.778 6.092 92.014"

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# 统一运行函数
run_one () {
  local tag="$1"
  local wv_sampling="$2"
  local wv_jss_std="$3"
  shift 3

  echo "================================================================"
  echo "[RUN] tag=${tag} wv_sampling=${wv_sampling} wv_jss_std=${wv_jss_std}"
  echo "================================================================"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_cycle4_${tag}" \
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
    --embed wv \
    --wv_sampling "${wv_sampling}" \
    --wv_jss_std "${wv_jss_std}" \
    --wv_base 10000.0 \
    "${AMP_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" \
    "$@" || echo "[WARN] tag=${tag} failed (exit=$?), continuing..."
}

# JSS std 扫描范围
JSS_STDS=(0.05 0.1 0.25 0.5 1.0 2.0)

# ─────────────────────────────────────────────
# scale_mode = standard（默认，不加额外参数）
# ─────────────────────────────────────────────
echo ">>> scale_mode=standard"
run_one "sm_standard_iss" "iss" "1.0"
for std in "${JSS_STDS[@]}"; do
  run_one "sm_standard_jss_${std}" "jss" "${std}"
done

# ─────────────────────────────────────────────
# scale_mode = none（--no_scale）
# ─────────────────────────────────────────────
echo ">>> scale_mode=none"
run_one "sm_none_iss" "iss" "1.0" --no_scale
for std in "${JSS_STDS[@]}"; do
  run_one "sm_none_jss_${std}" "jss" "${std}" --no_scale
done

# ─────────────────────────────────────────────
# scale_mode = prior（--scale_mode prior --prior_scale ...）
# ─────────────────────────────────────────────
echo ">>> scale_mode=prior"
read -r -a PS_ARR <<< "${PRIOR_SCALE}"
run_one "sm_prior_iss" "iss" "1.0" --scale_mode prior --prior_scale "${PS_ARR[@]}"
for std in "${JSS_STDS[@]}"; do
  run_one "sm_prior_jss_${std}" "jss" "${std}" --scale_mode prior --prior_scale "${PS_ARR[@]}"
done

echo ""
echo "[DONE] Cycle 4 Step 1 (3-factor scan) completed. DES=${DES}"
echo "Total experiments: 21"
