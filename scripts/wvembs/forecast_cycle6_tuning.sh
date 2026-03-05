#!/usr/bin/env bash
set -euo pipefail

# Cycle 6 调参实验：针对退化严重的配置进行参数调优
# 目标：修复以下退化案例
#   1. ETTh2 pl336/720 wv_std_jss (+24.5%, +194.9%) → 试更大 jss_std (0.5, 1.0) 和 ISS
#   2. ETTm2 pl336    wv_std_jss (+12.4%)            → 试更大 jss_std (0.5, 1.0) 和 ISS
#   3. ETTm2 pl96/192 wv_none_iss_extrap (+19.5%, +48.6%) → 试更小 extrap_scale (2.0, 3.0)

# 全局固定配置（与 Table 1 一致）
MODEL=Transformer
SEQ_LEN=96
LABEL_LEN=48
E_LAYERS=2
D_LAYERS=1
FACTOR=3
D_MODEL=512
D_FF=2048
N_HEADS=8
EPOCHS="${EPOCHS:-10}"
BATCH=32
WORKERS=8
DES="WVEmbsCycle6_Tuning"

TOTAL=0

# 通用运行函数（简化版）
run_one() {
  local ds_name="$1" root_path="$2" data_path="$3" data_flag="$4"
  local enc_in="$5" freq="$6" pred_len="$7" config_tag="$8" embed="$9"
  shift 9
  local -a EXTRA_ARGS=("$@")

  local model_id="WVEmbs_${ds_name}_${MODEL}_C6tune_${config_tag}_pl${pred_len}"
  TOTAL=$((TOTAL + 1))

  echo "================================================================"
  echo "[TUNE #${TOTAL}] ${ds_name} pl${pred_len} ${config_tag}"
  echo "================================================================"

  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path "${root_path}/" \
    --data_path "${data_path}" \
    --model_id "${model_id}" \
    --model "${MODEL}" \
    --data "${data_flag}" \
    --features M \
    --freq "${freq}" \
    --seq_len "${SEQ_LEN}" \
    --label_len "${LABEL_LEN}" \
    --pred_len "${pred_len}" \
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
    --checkpoints ./checkpoints_wvembs/ \
    --inverse \
    --embed "${embed}" \
    --use_amp \
    "${EXTRA_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] ${ds_name}/pl${pred_len}/${config_tag} failed (exit=$?), continuing..."
}

echo "=============================================="
echo "Part 1: ETTh2 退化修复（pl336, pl720）"
echo "=============================================="

# ETTh2 基线：wv_std_jss(0.25) pl336=319.685(+24.5%), pl720=403.328(+194.9%)
# 策略 A: 增大 jss_std
for PL in 336 720; do
  for STD in 0.5 1.0; do
    run_one "ETTh2" "./dataset/ETT-small" "ETTh2.csv" "ETTh2" 7 h "${PL}" \
      "wv_std_jss${STD}" "wv" \
      --scale_mode standard --wv_sampling jss --wv_jss_std "${STD}"
  done
done

# 策略 B: 改用 ISS (standard + iss)
for PL in 336 720; do
  run_one "ETTh2" "./dataset/ETT-small" "ETTh2.csv" "ETTh2" 7 h "${PL}" \
    "wv_std_iss" "wv" \
    --scale_mode standard --wv_sampling iss
done

# 策略 C: standard + jss + extrap (结合 StandardScaler 的 jss 和外推增益)
for PL in 336 720; do
  run_one "ETTh2" "./dataset/ETT-small" "ETTh2.csv" "ETTh2" 7 h "${PL}" \
    "wv_std_jss0.25_extrap5" "wv" \
    --scale_mode standard --wv_sampling jss --wv_jss_std 0.25 \
    --wv_extrap_mode scale --wv_extrap_scale 5.0
done

echo "=============================================="
echo "Part 2: ETTm2 退化修复"
echo "=============================================="

# ETTm2 wv_std_jss(0.25) pl336=85.608(+12.4%)
# 策略 A: 增大 jss_std
for STD in 0.5 1.0; do
  run_one "ETTm2" "./dataset/ETT-small" "ETTm2.csv" "ETTm2" 7 t 336 \
    "wv_std_jss${STD}" "wv" \
    --scale_mode standard --wv_sampling jss --wv_jss_std "${STD}"
done

# 策略 B: ISS
run_one "ETTm2" "./dataset/ETT-small" "ETTm2.csv" "ETTm2" 7 t 336 \
  "wv_std_iss" "wv" \
  --scale_mode standard --wv_sampling iss

# ETTm2 wv_none_iss_extrap pl96=35.283(+19.5%), pl192=97.952(+48.6%)
# 策略 D: 减小 extrap_scale
for PL in 96 192; do
  for ESCALE in 2.0 3.0; do
    run_one "ETTm2" "./dataset/ETT-small" "ETTm2.csv" "ETTm2" 7 t "${PL}" \
      "wv_none_iss_extrap${ESCALE}" "wv" \
      --no_scale --wv_sampling iss --wv_extrap_mode scale --wv_extrap_scale "${ESCALE}"
  done
done

echo ""
echo "=============================================="
echo "[DONE] Cycle 6 Tuning: ${TOTAL} experiments completed"
echo "=============================================="
