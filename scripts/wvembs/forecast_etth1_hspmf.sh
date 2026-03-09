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
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints_hspmf/}"

USE_AMP="${USE_AMP:-1}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_MSE="${RUN_MSE:-1}"
RUN_NLL="${RUN_NLL:-1}"
RUN_INFER_DECODE="${RUN_INFER_DECODE:-1}"
BASELINE_DES="${BASELINE_DES:-${DES}}"
BASELINE_LOAD_SETTING="${BASELINE_LOAD_SETTING:-}"
INFER_DECODE_NAME="${INFER_DECODE_NAME:-baseline_inferdecode}"
INFER_BETA_VALUES="${INFER_BETA_VALUES:-0.00390625 0.0078125 0.015625 0.03125 0.0625}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

WV_ARGS=(
  --wv_sampling iss
  --wv_jss_std 0.25
  --wv_base 10000.0
)

baseline_setting_name() {
  if [[ -n "${BASELINE_LOAD_SETTING}" ]]; then
    printf '%s\n' "${BASELINE_LOAD_SETTING}"
    return
  fi

  printf 'long_term_forecast_HSPMF_%s_baseline_Transformer_%s_ftM_sl%s_ll%s_pl%s_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebwv_dtTrue_%s_0\n' \
    "${DATASET}" "${DATASET}" "${SEQ_LEN}" "${LABEL_LEN}" "${PRED_LEN}" "${BASELINE_DES}"
}

run_one() {
  local mode="$1"
  local name="$2"
  local model="$3"

  local -a HSPMF_ARGS=()
  if [[ "${mode}" != "baseline" ]]; then
    HSPMF_ARGS=(
      --use_hspmf
      --hspmf_n_fourier 16
      --hspmf_period 1.0
      --hspmf_x_range -6 6
      --hspmf_grid_size 64
      --hspmf_beta 1.0
      --hspmf_tau 1.0
      --hspmf_score_mode abs2
      --hspmf_loss "${mode}"
    )
    if [[ "${mode}" == "nll" ]]; then
      HSPMF_ARGS+=(
        --hspmf_learn_beta
      )
    fi
  fi

  if [[ "${mode}" == "baseline" ]]; then
    echo "[RUN] ${name}: model=${model}, baseline WVEmbs"
  else
    echo "[RUN] ${name}: model=${model}, hspmf_loss=${mode}"
  fi

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
    --checkpoints "${CHECKPOINT_ROOT}" \
    --embed wv \
    --scale_mode standard \
    --inverse \
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    "${HSPMF_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
}

run_infer_decode() {
  local load_setting
  local checkpoint_path
  local -a beta_args

  load_setting="$(baseline_setting_name)"
  checkpoint_path="${CHECKPOINT_ROOT%/}/${load_setting}/checkpoint.pth"
  if [[ ! -f "${checkpoint_path}" ]]; then
    echo "[ERROR] 未找到 baseline checkpoint: ${checkpoint_path}" >&2
    echo "        可通过 BASELINE_LOAD_SETTING 指定已有 setting。" >&2
    exit 1
  fi

  read -r -a beta_args <<< "${INFER_BETA_VALUES}"

  echo "[RUN] ${INFER_DECODE_NAME}: model=Transformer, infer_decode from ${load_setting}"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "HSPMF_${DATASET}_${INFER_DECODE_NAME}" \
    --model Transformer \
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
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${WORKERS}" \
    --checkpoints "${CHECKPOINT_ROOT}" \
    --embed wv \
    --scale_mode standard \
    --inverse \
    "${WV_ARGS[@]}" \
    --hspmf_infer_decode \
    --hspmf_n_fourier 16 \
    --hspmf_period 1.0 \
    --hspmf_x_range -6 6 \
    --hspmf_grid_size 64 \
    --hspmf_tau 1.0 \
    --hspmf_score_mode abs2 \
    --hspmf_infer_beta_values "${beta_args[@]}" \
    --load_setting "${load_setting}" \
    --itr 1 \
    --des "${DES}"
}

echo "========== HSPMF 方法验证实验 =========="
echo "Dataset: ${DATASET}, Pred_len: ${PRED_LEN}, Epochs: ${EPOCHS}"
echo ""

# A. 基线：Transformer + WVEmbs
if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_one baseline "baseline" "Transformer"
fi

# B. HSPMF 输出头 + 点预测 MSE
if [[ "${RUN_MSE}" == "1" ]]; then
  run_one mse "hspmf_mse" "Transformer_HSPMF"
fi

# C. HSPMF 输出头 + End2End-NLL
if [[ "${RUN_NLL}" == "1" ]]; then
  run_one nll "hspmf_e2e_nll" "Transformer_HSPMF"
fi

# D. 纯 WVEmbs backbone + 推理期 HSPMF 解码
if [[ "${RUN_INFER_DECODE}" == "1" ]]; then
  run_infer_decode
fi

echo ""
echo "========== 实验完成 =========="
