#!/usr/bin/env bash
set -euo pipefail

# 阶段 0：ETTh1 + Transformer 的 embed 对照（对齐上游官方脚本的默认超参）
#
# 对照组：
# - timeF：原始基线（TokenEmbedding + TimeFeatureEmbedding）
# - wv_timeF：消融（值 WVEmbs + 时间仍用 TimeFeatureEmbedding）
# - wv：统一模式（值 + 时间通道一起进入 WVEmbs）
#
# 指标口径：不启用 `--inverse`，与上游脚本保持一致（指标在缩放后的无量纲空间）。

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

# 上游脚本未显式设置 d_model/d_ff/n_heads，这里显式写出与 run.py 默认一致的值，便于复现实验
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
WV_JSS_STD="${WV_JSS_STD:-1.0}"
WV_BASE="${WV_BASE:-10000.0}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

for embed in timeF wv_timeF wv; do
  # 统一模式与采样策略有较强交互：默认按最终报告复现实验
  # - wv_timeF: jss
  # - wv: iss
  #
  # 若设置环境变量 WV_SAMPLING，则对 wv_timeF/wv 统一覆盖（便于扫参）。
  WV_ARGS=()
  if [[ "${embed}" == "wv_timeF" ]]; then
    sampling="${WV_SAMPLING:-jss}"
    WV_ARGS=(--wv_sampling "${sampling}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  elif [[ "${embed}" == "wv" ]]; then
    sampling="${WV_SAMPLING:-iss}"
    WV_ARGS=(--wv_sampling "${sampling}" --wv_jss_std "${WV_JSS_STD}" --wv_base "${WV_BASE}")
  fi

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_stage0_${embed}" \
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
    "${AMP_ARGS[@]}" \
    "${WV_ARGS[@]}" \
    --itr 1 \
    --des "${DES}"
done
