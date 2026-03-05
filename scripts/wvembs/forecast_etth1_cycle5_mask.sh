#!/usr/bin/env bash
set -euo pipefail

# Cycle 5 Step 1: 掩码消融实验（ETTh1 Forecast, Transformer）
#
# 扫描维度：
#   wv_mask_type ∈ {zero, arcsine, phase_rotate}
#   wv_mask_prob ∈ {0.1, 0.3, 0.5}
#   wv_mask_dlow_min ∈ {0, 4}
#
# 固定配置：embed=wv, wv_sampling=iss, scale_mode=none, inverse=True
# 基线 = wv_mask_prob=0（无掩码），来自 Cycle 4 结果 none+iss MSE=22.857
# 共 3×3×2 = 18 组
#
# 预估耗时：~36 min（单 GPU 串行，每组 ~2 min）

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

DES="${DES:-WVEmbsCycle5_mask}"

# phase_rotate 的最大相位扰动角度
PHI_MAX=0.39269908169872414  # pi/8

USE_AMP="${USE_AMP:-1}"
AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

# 统一运行函数
run_one () {
  local tag="$1"
  local mask_type="$2"
  local mask_prob="$3"
  local dlow_min="$4"

  echo "================================================================"
  echo "[RUN] tag=${tag} mask_type=${mask_type} mask_prob=${mask_prob} dlow_min=${dlow_min}"
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
    --wv_mask_type "${mask_type}" \
    --wv_mask_prob "${mask_prob}" \
    --wv_mask_phi_max "${PHI_MAX}" \
    --wv_mask_dlow_min "${dlow_min}" \
    "${AMP_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] tag=${tag} failed (exit=$?), continuing..."
}

# ──── 掩码类型 × 掩码概率 × dlow_min 三因素扫描 ────

MASK_TYPES=(zero arcsine phase_rotate)
MASK_PROBS=(0.1 0.3 0.5)
DLOW_MINS=(0 4)

for mask_type in "${MASK_TYPES[@]}"; do
  for mask_prob in "${MASK_PROBS[@]}"; do
    for dlow_min in "${DLOW_MINS[@]}"; do
      tag="mt_${mask_type}_mp_${mask_prob}_dl_${dlow_min}"
      run_one "${tag}" "${mask_type}" "${mask_prob}" "${dlow_min}"
    done
  done
done

echo ""
echo "[DONE] Cycle 5 Step 1 (mask ablation) completed. DES=${DES}"
echo "Total experiments: 18"
