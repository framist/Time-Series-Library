#!/usr/bin/env bash
set -euo pipefail

# Cycle 5 Step 3: 跨数据集验证（ETTm1 + Weather）
#
# 验证 Cycle 5 在 ETTh1 上发现的最佳配置是否跨数据集成立：
#   1. 最佳掩码：phase_rotate, prob=0.1, dlow_min=4
#   2. 最佳外推：scale=5.0
#   3. 叠加：phase_rotate + scale=5.0
#   4. 无掩码无外推基线（embed=wv, none+iss）
#
# 固定配置：embed=wv, wv_sampling=iss, scale_mode=none, inverse=True
# 共 4 配置 × 2 数据集 = 8 组
#
# 预估耗时：~20 min（ETTm1 ~2min/组, Weather ~3min/组）

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

DES="${DES:-WVEmbsCycle5_crossval}"

# phase_rotate 参数
PHI_MAX=0.39269908169872414  # pi/8

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# ────────────────────────────────────────
# 通用运行函数
# ────────────────────────────────────────
run_one() {
  local dataset_name="$1"    # ETTm1 / Weather
  local root_path="$2"
  local data_path="$3"
  local data_flag="$4"       # ETTm1 / custom
  local enc_in="$5"
  local tag="$6"
  shift 6
  local -a EXTRA_ARGS=("$@")

  echo "================================================================"
  echo "[RUN] dataset=${dataset_name} tag=${tag}"
  echo "================================================================"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${root_path}/" \
    --data_path "${data_path}" \
    --model_id "WVEmbs_${dataset_name}_${MODEL}_cycle5_${tag}" \
    --model "${MODEL}" \
    --data "${data_flag}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${PRED_LEN}" \
    --factor "${FACTOR}" \
    --enc_in "${enc_in}" \
    --dec_in "${enc_in}" \
    --c_out "${enc_in}" \
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
    "${EXTRA_ARGS[@]}" \
    "${AMP_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] ${dataset_name}/${tag} failed (exit=$?), continuing..."
}

# ────────────────────────────────────────
# 对每个数据集跑 4 种配置
# ────────────────────────────────────────
run_dataset() {
  local dname="$1"
  local root="$2"
  local dpath="$3"
  local dflag="$4"
  local enc_in="$5"

  # (A) 基线：无掩码、无外推
  run_one "${dname}" "${root}" "${dpath}" "${dflag}" "${enc_in}" \
    "cv_baseline"

  # (B) 最佳掩码：phase_rotate, prob=0.1, dlow_min=4
  run_one "${dname}" "${root}" "${dpath}" "${dflag}" "${enc_in}" \
    "cv_mask_best" \
    --wv_mask_type phase_rotate \
    --wv_mask_prob 0.1 \
    --wv_mask_phi_max "${PHI_MAX}" \
    --wv_mask_dlow_min 4

  # (C) 最佳外推：scale=5.0
  run_one "${dname}" "${root}" "${dpath}" "${dflag}" "${enc_in}" \
    "cv_extrap_best" \
    --wv_extrap_mode scale \
    --wv_extrap_scale 5.0

  # (D) 叠加：mask + extrap
  run_one "${dname}" "${root}" "${dpath}" "${dflag}" "${enc_in}" \
    "cv_combo" \
    --wv_mask_type phase_rotate \
    --wv_mask_prob 0.1 \
    --wv_mask_phi_max "${PHI_MAX}" \
    --wv_mask_dlow_min 4 \
    --wv_extrap_mode scale \
    --wv_extrap_scale 5.0
}

# ── ETTm1 ──
run_dataset "ETTm1" "./dataset/ETT-small" "ETTm1.csv" "ETTm1" 7

# ── Weather ──
run_dataset "Weather" "./dataset/weather" "weather.csv" "custom" 21

echo ""
echo "[DONE] Cycle 5 Step 3 (cross-dataset validation) completed. DES=${DES}"
echo "Total experiments: 8"
