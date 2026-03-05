#!/usr/bin/env bash
set -euo pipefail

# Cycle 5 Step 2: 外推实验（ETTh1 Forecast, Transformer）
#
# 对比 wv_extrap_mode：
#   - direct（默认，不缩放）→ 1 组
#   - scale（x / wv_extrap_scale）→ wv_extrap_scale ∈ {1.5, 2.0, 5.0} → 3 组
#
# 固定配置：embed=wv, wv_sampling=iss, scale_mode=none, inverse=True
# 共 4 组
#
# 注意：此脚本依赖 exp_long_term_forecasting.py 中的域内/域外分组统计功能
# （通过 --wv_extrap_eval 开关启用）
#
# 预估耗时：~8 min（单 GPU 串行）

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

DES="${DES:-WVEmbsCycle5_extrap}"

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# 统一运行函数
run_one () {
  local tag="$1"
  local extrap_mode="$2"
  local extrap_scale="$3"

  echo "================================================================"
  echo "[RUN] tag=${tag} extrap_mode=${extrap_mode} extrap_scale=${extrap_scale}"
  echo "================================================================"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${ROOT}/" \
    --data_path "${DATA}" \
    --model_id "WVEmbs_ETTh1_${MODEL}_cycle5_${tag}" \
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
    --wv_base 10000.0 \
    --wv_extrap_mode "${extrap_mode}" \
    --wv_extrap_scale "${extrap_scale}" \
    --wv_extrap_eval \
    "${AMP_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] tag=${tag} failed (exit=$?), continuing..."
}

# ──── 外推模式扫描 ────

# 基线：direct 模式（无缩放）
run_one "extrap_direct" "direct" "1.0"

# scale 模式：不同缩放因子
for scale in 1.5 2.0 5.0; do
  run_one "extrap_scale_${scale}" "scale" "${scale}"
done

echo ""
echo "[DONE] Cycle 5 Step 2 (extrapolation) completed. DES=${DES}"
echo "Total experiments: 4"
