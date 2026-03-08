#!/usr/bin/env bash
set -euo pipefail

# 族群 B（ETTh2 / ETTm2）高优先修复扫描
#
# 目标：
# - 在截止时间前优先验证“共享采样失效时，ISS 或外推增强是否能更稳地修复长预测退化”
# - 保留与主表一致的 backbone / 训练协议，避免再开大规模全量扫
#
# 当前聚焦：
# - ETTh2：pl336 / pl720，优先比较 standard+jss 基线的两条备选修复路线
#   1) standard + iss
#   2) standard + jss(0.25) + extrap(scale=5.0)
# - ETTm2：pl336，先验证 standard + iss 这一条代表性 fallback
#
# 用法：
#   bash scripts/wvembs/forecast_groupb_priority_scan.sh
#   DES=GroupBPriority_20260309 EPOCHS=10 bash scripts/wvembs/forecast_groupb_priority_scan.sh

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
DES="${DES:-GroupBPriorityScan}"
USE_AMP="${USE_AMP:-1}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

TOTAL=0

run_one() {
  local ds_name="$1" root_path="$2" data_path="$3" data_flag="$4"
  local enc_in="$5" freq="$6" pred_len="$7" config_tag="$8"
  shift 8
  local -a EXTRA_ARGS=("$@")

  local model_id="WVEmbs_${ds_name}_${MODEL}_GroupB_${config_tag}_pl${pred_len}"
  TOTAL=$((TOTAL + 1))

  echo "================================================================"
  echo "[GROUP-B #${TOTAL}] ${ds_name} pl${pred_len} ${config_tag}"
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
    --embed wv \
    "${AMP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] ${ds_name}/pl${pred_len}/${config_tag} failed (exit=$?), continuing..."
}

echo "=============================================="
echo "Part 1: ETTh2 长预测修复"
echo "=============================================="

for PL in 336 720; do
  run_one "ETTh2" "./dataset/ETT-small" "ETTh2.csv" "ETTh2" 7 h "${PL}" \
    "wv_std_iss" \
    --scale_mode standard --wv_sampling iss

  run_one "ETTh2" "./dataset/ETT-small" "ETTh2.csv" "ETTh2" 7 h "${PL}" \
    "wv_std_jss0.25_extrap5" \
    --scale_mode standard --wv_sampling jss --wv_jss_std 0.25 \
    --wv_extrap_mode scale --wv_extrap_scale 5.0
done

echo "=============================================="
echo "Part 2: ETTm2 代表性 fallback"
echo "=============================================="

run_one "ETTm2" "./dataset/ETT-small" "ETTm2.csv" "ETTm2" 7 t 336 \
  "wv_std_iss" \
  --scale_mode standard --wv_sampling iss

echo ""
echo "=============================================="
echo "[DONE] Group B priority scan: ${TOTAL} experiments completed"
echo "=============================================="
