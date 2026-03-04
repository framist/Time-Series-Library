#!/usr/bin/env bash
set -euo pipefail

# Cycle 0：GPU 预算探测（ETTh1 + Transformer + features=M）
#
# 目标：
# - 记录每个 epoch 的耗时（exp_long_term_forecasting.py 会打印 "Epoch: X cost time: ..."）
# - 记录训练期间的峰值显存（通过轮询 nvidia-smi）
#
# 用法：
#   bash scripts/wvembs/gpu_budget_probe_etth1.sh
#
# 产物：
# - results/gpu_probe_etth1_YYYYmmdd_HHMMSS/train.log
# - results/gpu_probe_etth1_YYYYmmdd_HHMMSS/gpu_mem.csv

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
PATIENCE="${PATIENCE:-20}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"

USE_AMP="${USE_AMP:-1}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/gpu_probe_etth1_${TS}"
mkdir -p "${OUT_DIR}"
MEM_LOG="${OUT_DIR}/gpu_mem.csv"
TRAIN_LOG="${OUT_DIR}/train.log"

echo "timestamp_s,memory_used_mb" > "${MEM_LOG}"

monitor_gpu () {
  while true; do
    local mem
    mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    echo "$(date +%s),${mem}" >> "${MEM_LOG}"
    sleep 1
  done
}

monitor_gpu &
MON_PID=$!
cleanup () {
  kill "${MON_PID}" 2>/dev/null || true
  wait "${MON_PID}" 2>/dev/null || true
}
trap cleanup EXIT

conda run -n radio python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path "${ROOT}/" \
  --data_path "${DATA}" \
  --model_id "GPUProbe_ETTh1_${MODEL}_std_timeF" \
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
  --patience "${PATIENCE}" \
  --batch_size "${BATCH}" \
  --num_workers "${WORKERS}" \
  --checkpoints ./checkpoints_wvembs/ \
  --embed timeF \
  "${AMP_ARGS[@]}" \
  --itr 1 \
  --des "GPUProbe" \
  >"${TRAIN_LOG}" 2>&1

PEAK_MB="$(awk -F, 'NR>1{m=$2+0; if(m>max)max=m} END{print max+0}' "${MEM_LOG}")"

EPOCH_LINES="$(grep -E '^Epoch: [0-9]+ cost time:' "${TRAIN_LOG}" || true)"
if [[ -z "${EPOCH_LINES}" ]]; then
  echo "[ERROR] 未在日志中找到 epoch 耗时行；请检查 ${TRAIN_LOG}" >&2
  exit 1
fi

EPOCH_N="$(echo "${EPOCH_LINES}" | wc -l | tr -d ' ')"
TOTAL_S="$(echo "${EPOCH_LINES}" | awk -F 'cost time: ' '{sum += $2} END{printf "%.6f", sum+0}')"
AVG_S="$(awk -v total="${TOTAL_S}" -v n="${EPOCH_N}" 'BEGIN{if(n>0){printf "%.6f", total/n}else{print "nan"}}')"

echo "[GPU_PROBE] out_dir=${OUT_DIR}"
echo "[GPU_PROBE] peak_mem_mb=${PEAK_MB}"
echo "[GPU_PROBE] epochs=${EPOCH_N} total_epoch_time_s=${TOTAL_S} avg_epoch_time_s=${AVG_S}"
echo "[GPU_PROBE] epoch_times:"
echo "${EPOCH_LINES}" | sed -E 's/^/  /'
