#!/usr/bin/env bash
set -euo pipefail

# RevIN vs WVEmbs 功能重叠消融实验（TimeMixer）
#
# 目标：验证 RevIN（模型内 Normalize）与 WVEmbs 是否存在功能重叠
# 背景：TimeMixer 默认 use_norm=1，叠加 DataLoader 的 scale_mode，形成双重归一化
#
# 四组基线对比：
# A. use_norm=1, embed=timeF    (RevIN-only baseline)
# B. use_norm=0, embed=wv       (WVEmbs-only)
# C. use_norm=1, embed=wv       (RevIN+WVEmbs, 当前默认)
# D. use_norm=0, embed=timeF    (None baseline)
#
# 统一设置：
# - scale_mode=none（避免 DataLoader 层额外归一化干扰）
# - 其他参数对齐 Cycle 2 TimeMixer 配置

ROOT=./dataset/ETT-small
DATA=ETTh1.csv
DATASET=ETTh1

SEQ_LEN=96
PRED_LEN=96

EPOCHS="${EPOCHS:-10}"
PATIENCE="${PATIENCE:-20}"
BATCH_SIZE="${BATCH:-128}"
WORKERS="${WORKERS:-8}"
DES="${DES:-RevIN_Ablation}"

USE_AMP="${USE_AMP:-1}"
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# TimeMixer 专用参数（必须显式指定 down_sampling_method）
TM_ARGS=(
  --down_sampling_layers 3
  --down_sampling_method avg
  --down_sampling_window 2
  --label_len 0
)

run_one() {
  local use_norm="$1"
  local embed="$2"
  local name="$3"

  local -a WV_ARGS=()
  if [[ "${embed}" == "wv" ]]; then
    WV_ARGS=(--wv_sampling iss --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  fi

  echo "[RUN] ${name}: use_norm=${use_norm}, embed=${embed}"
  
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "RevIN_${DATASET}_${name}" \
    --model TimeMixer \
    --data "${DATASET}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 16 \
    --d_ff 32 \
    --e_layers 2 \
    --train_epochs "${EPOCHS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${WORKERS}" \
    --learning_rate 0.01 \
    --checkpoints ./checkpoints_revin/ \
    --embed "${embed}" \
    --use_norm "${use_norm}" \
    --scale_mode none \
    --inverse \
    "${AMP_ARGS[@]}" \
    "${TM_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
}

# 四组基线对比
echo "========== RevIN vs WVEmbs 消融实验 =========="
echo "Dataset: ${DATASET}, Pred_len: ${PRED_LEN}, Epochs: ${EPOCHS}"
echo ""

# A. RevIN-only (use_norm=1, embed=timeF)
run_one 1 "timeF" "RevIN_only"

# B. WVEmbs-only (use_norm=0, embed=wv)
run_one 0 "wv" "WVEmbs_only"

# C. RevIN+WVEmbs (use_norm=1, embed=wv)
run_one 1 "wv" "RevIN_WVEmbs"

# D. None (use_norm=0, embed=timeF)
run_one 0 "timeF" "None"

echo ""
echo "========== 实验完成 =========="
echo "结果汇总请查看: ./checkpoints_revin/"
