#!/usr/bin/env bash
set -euo pipefail

# Cycle 4 Step 2: wv_base 扫描（ETTh1 Forecast, Transformer）
#
# 固定配置：embed=wv, scale_mode=none, wv_sampling=iss
# 扫描维度：wv_base ∈ {100, 500, 1000, 5000, 10000, 50000}
#
# 口径：inverse=True
# 共 6 组
#
# 预估耗时：~12 min（单 GPU 串行）

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

DES="${DES:-WVEmbsCycle4_wvbase}"

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# 统一运行函数
run_one () {
  local wv_base="$1"

  echo "================================================================"
  echo "[RUN] wv_base=${wv_base}"
  echo "================================================================"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_cycle4_base_${wv_base}" \
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
    --no_scale \
    --embed wv \
    --wv_sampling iss \
    --wv_jss_std 1.0 \
    --wv_base "${wv_base}" \
    "${AMP_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] wv_base=${wv_base} failed (exit=$?), continuing..."
}

# 扫描 wv_base
for base in 100 500 1000 5000 10000 50000; do
  run_one "${base}"
done

echo ""
echo "[DONE] Cycle 4 Step 2 (wv_base scan) completed. DES=${DES}"
echo "Total experiments: 6"
