#!/usr/bin/env bash
set -euo pipefail

# Cycle 6 Table 1: 多数据集 × 多 pred_len × 多配置（Transformer）
# 目标：生成论文 Table 1 的 60 组实验（5 数据集 × 4 预测长度 × 3 配置）

# -------------------------------
# 全局固定配置（支持环境变量覆盖）
# -------------------------------
MODEL=Transformer
SEQ_LEN="${SEQ_LEN:-96}"
LABEL_LEN="${LABEL_LEN:-48}"

E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
FACTOR="${FACTOR:-3}"

D_MODEL="${D_MODEL:-512}"
D_FF="${D_FF:-2048}"
N_HEADS="${N_HEADS:-8}"

EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-8}"
MAX_TRAIN="${MAX_TRAIN:--1}"
MAX_VAL="${MAX_VAL:--1}"
MAX_TEST="${MAX_TEST:--1}"

DES="${DES:-WVEmbsCycle6_Table1}"
USE_AMP="${USE_AMP:-1}"

# 数据集开关（默认全开）
RUN_ETTH1="${RUN_ETTH1:-1}"
RUN_ETTH2="${RUN_ETTH2:-1}"
RUN_ETTM1="${RUN_ETTM1:-1}"
RUN_ETTM2="${RUN_ETTM2:-1}"
RUN_WEATHER="${RUN_WEATHER:-1}"

AMP_ARGS=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_ARGS=(--use_amp)
fi

PRED_LENS=(96 192 336 720)

TOTAL_EXPS=0

# -------------------------------
# 通用单实验运行函数
# -------------------------------
run_one() {
  local dataset_name="$1"
  local root_path="$2"
  local data_path="$3"
  local data_flag="$4"
  local enc_in="$5"
  local freq="$6"
  local pred_len="$7"
  local config_tag="$8"
  local embed="$9"
  shift 9
  local -a EXTRA_ARGS=("$@")

  local model_id="WVEmbs_${dataset_name}_${MODEL}_C6_${config_tag}_pl${pred_len}"

  echo "================================================================"
  echo "[RUN] dataset=${dataset_name} pred_len=${pred_len} config=${config_tag}"
  echo "[RUN] model_id=${model_id}"
  echo "================================================================"

  TOTAL_EXPS=$((TOTAL_EXPS + 1))

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
    --max_train_steps "${MAX_TRAIN}" \
    --max_val_steps "${MAX_VAL}" \
    --max_test_steps "${MAX_TEST}" \
    --checkpoints ./checkpoints_wvembs/ \
    --inverse \
    --embed "${embed}" \
    "${AMP_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    --itr 1 \
    --des "${DES}" || echo "[WARN] ${dataset_name}/pl${pred_len}/${config_tag} failed (exit=$?), continuing..."
}

# -------------------------------
# 数据集批量运行函数（4 pred_len × 3 配置）
# -------------------------------
run_dataset() {
  local dataset_name="$1"
  local root_path="$2"
  local data_path="$3"
  local data_flag="$4"
  local enc_in="$5"
  local freq="$6"

  local pred_len
  for pred_len in "${PRED_LENS[@]}"; do
    # (a) timeF (standard)
    run_one "${dataset_name}" "${root_path}" "${data_path}" "${data_flag}" "${enc_in}" "${freq}" "${pred_len}" \
      "timeF_std" "timeF"

    # (b) wv (standard + jss)
    run_one "${dataset_name}" "${root_path}" "${data_path}" "${data_flag}" "${enc_in}" "${freq}" "${pred_len}" \
      "wv_std_jss" "wv" \
      --scale_mode standard \
      --wv_sampling jss \
      --wv_jss_std 0.25

    # (c) wv (none + iss + extrap)
    run_one "${dataset_name}" "${root_path}" "${data_path}" "${data_flag}" "${enc_in}" "${freq}" "${pred_len}" \
      "wv_none_iss_extrap" "wv" \
      --no_scale \
      --wv_sampling iss \
      --wv_extrap_mode scale \
      --wv_extrap_scale 5.0
  done
}

# -------------------------------
# 运行入口（按开关选择数据集）
# -------------------------------
if [[ "${RUN_ETTH1}" == "1" ]]; then
  run_dataset "ETTh1" "./dataset/ETT-small" "ETTh1.csv" "ETTh1" 7 h
fi

if [[ "${RUN_ETTH2}" == "1" ]]; then
  run_dataset "ETTh2" "./dataset/ETT-small" "ETTh2.csv" "ETTh2" 7 h
fi

if [[ "${RUN_ETTM1}" == "1" ]]; then
  run_dataset "ETTm1" "./dataset/ETT-small" "ETTm1.csv" "ETTm1" 7 t
fi

if [[ "${RUN_ETTM2}" == "1" ]]; then
  run_dataset "ETTm2" "./dataset/ETT-small" "ETTm2.csv" "ETTm2" 7 t
fi

if [[ "${RUN_WEATHER}" == "1" ]]; then
  run_dataset "Weather" "./dataset/weather" "weather.csv" "custom" 21 t
fi

# -------------------------------
# 实验数量汇总
# -------------------------------
SELECTED_DATASETS=0
if [[ "${RUN_ETTH1}" == "1" ]]; then SELECTED_DATASETS=$((SELECTED_DATASETS + 1)); fi
if [[ "${RUN_ETTH2}" == "1" ]]; then SELECTED_DATASETS=$((SELECTED_DATASETS + 1)); fi
if [[ "${RUN_ETTM1}" == "1" ]]; then SELECTED_DATASETS=$((SELECTED_DATASETS + 1)); fi
if [[ "${RUN_ETTM2}" == "1" ]]; then SELECTED_DATASETS=$((SELECTED_DATASETS + 1)); fi
if [[ "${RUN_WEATHER}" == "1" ]]; then SELECTED_DATASETS=$((SELECTED_DATASETS + 1)); fi

PLANNED_EXPS=$((SELECTED_DATASETS * 4 * 3))

echo ""
echo "[DONE] Cycle 6 Table 1 script finished. DES=${DES}"
echo "Planned experiments: ${PLANNED_EXPS}"
echo "Launched experiments: ${TOTAL_EXPS}"
