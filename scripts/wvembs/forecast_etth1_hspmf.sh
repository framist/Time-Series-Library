#!/usr/bin/env bash
set -euo pipefail

# HSPMF 方法验证实验（修复版）
#
# 使用 Transformer_HSPMF 模型（而非 Transformer）

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=96

EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
DES="${DES:-HSPMF_Validation}"

USE_AMP="${USE_AMP:-1}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

WV_ARGS=(
  --wv_sampling iss
  --wv_jss_std 0.25
  --wv_base 10000.0
)

run_one() {
  local use_hspmf="$1"
  local name="$2"
  local model="$3"

  local -a HSPMF_ARGS=()
  if [[ "${use_hspmf}" == "1" ]]; then
    HSPMF_ARGS=(
      --use_hspmf
      --hspmf_n_fourier 16
      --hspmf_period 1.0
      --hspmf_x_range -6 6
      --hspmf_grid_size 64
      --hspmf_beta 1.0
      --hspmf_tau 1.0
      --hspmf_score_mode abs2
    )
  fi

  echo "[RUN] ${name}: model=${model}, use_hspmf=${use_hspmf}"
  
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "HSPMF_${DATASET}_${name}" \
    --model "${model}" \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${PRED_LEN}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --d_ff 2048 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --train_epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${WORKERS}" \
    --checkpoints ./checkpoints_hspmf/ \
    --embed wv \
    --scale_mode standard \
    --inverse \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    "${HSPMF_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
}

echo "========== HSPMF 方法验证实验 =========="
echo "Dataset: ${DATASET}, Pred_len: ${PRED_LEN}, Epochs: ${EPOCHS}"
echo ""

# A. 基线：Transformer + WVEmbs
run_one 0 "baseline" "Transformer"

# B. HSPMF 增强：Transformer_HSPMF + HSPMF
run_one 1 "hspmf" "Transformer_HSPMF"

echo ""
echo "========== 实验完成 =========="
