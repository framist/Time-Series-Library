#!/usr/bin/env bash
set -euo pipefail

# Cycle 4 Step 3: ETTh2 跨数据集验证（jss_std 关键区间）
#
# 扫描维度：
#   wv_jss_std ∈ {0.1, 0.25, 0.5}
#   scale_mode ∈ {none, prior}
#
# 目的：验证 Step 1 中最优 jss_std 区间是否跨数据集稳定
# 口径：embed=wv, wv_sampling=jss, inverse=True
# 共 2 × 3 = 6 组
#
# 预估耗时：~12 min（单 GPU 串行）

ROOT=./dataset/ETT-small
DATA=ETTh2.csv
DATASET=ETTh2
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

DES="${DES:-WVEmbsCycle4_ETTh2_jssstd}"

# 先验尺度（ETTh2, 7通道, max(|x_train|)×2）
PRIOR_SCALE="215.786 72.878 186.46 57.472 34.436 62.924 116.875"

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# 统一运行函数
run_one () {
  local tag="$1"
  local wv_jss_std="$2"
  shift 2

  echo "================================================================"
  echo "[RUN] tag=${tag} wv_jss_std=${wv_jss_std}"
  echo "================================================================"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh2_${MODEL}_cycle4_${tag}" \
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
    --wv_sampling jss \
    --wv_jss_std "${wv_jss_std}" \
    --wv_base 10000.0 \
    "${AMP_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" \
    "$@" || echo "[WARN] tag=${tag} failed (exit=$?), continuing..."
}

read -r -a PS_ARR <<< "${PRIOR_SCALE}"

# scale_mode = none
echo ">>> scale_mode=none"
for std in 0.1 0.25 0.5; do
  run_one "sm_none_jss_${std}" "${std}" --no_scale
done

# scale_mode = prior
echo ">>> scale_mode=prior"
for std in 0.1 0.25 0.5; do
  run_one "sm_prior_jss_${std}" "${std}" --scale_mode prior --prior_scale "${PS_ARR[@]}"
done

echo ""
echo "[DONE] Cycle 4 Step 3 (ETTh2 jss_std cross-dataset validation) completed. DES=${DES}"
echo "Total experiments: 6"
