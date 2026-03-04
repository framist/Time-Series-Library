#!/usr/bin/env bash
set -euo pipefail

# 运行 WVEmbs 阶段 0/1 的最终实验套件（真实数据集）。
#
# 说明：
# - 各子脚本已支持 `USE_AMP/WV_SAMPLING/WV_JSS_STD/WV_BASE` 等环境变量；
# - 本脚本只负责统一预算与 prior 参数，便于“一键复现最终报告”。

DES="${DES:-WVEmbsFinal_20260304}"
export DES

# 更好利用 GPU：默认开启 AMP
# - WV 相关采样策略不要在这里统一覆盖：不同子脚本对 wv_timeF/wv 以及 stage1 的 B/C 组有不同默认（与 Report.md 对齐）。
export USE_AMP="${USE_AMP:-1}"
export WV_JSS_STD="${WV_JSS_STD:-1.0}"
export WV_BASE="${WV_BASE:-10000.0}"

# prior（range-based prior）：使用训练段 max(|x|) 的宽松上界（slack=2）作为尺度先验
# - ETT 系列数值幅度在 O(10~100)，prior_scale 取 2*max_abs（offset 默认 0）
# - Electricity（OT）在本报告中的“最终表”使用手调的 `prior_scale=5000`（该数据集对 prior_scale 量级非常敏感）
PRIOR_SCALE_ETTH1="${PRIOR_SCALE_ETTH1:-47.288 17.682 42.57 13.788 15.778 6.092 92.014}"
PRIOR_SCALE_ETTm1="${PRIOR_SCALE_ETTm1:-48.36 18.218 44.206 14.356 16.326 6.092 92.014}"
PRIOR_SCALE_ECL_OT="${PRIOR_SCALE_ECL_OT:-5000}"

echo "[INFO] DES=${DES}"
echo "[INFO] USE_AMP=${USE_AMP} WV_JSS_STD=${WV_JSS_STD} WV_BASE=${WV_BASE}"
echo "[INFO] WV_SAMPLING 未在此脚本中统一设置：由各子脚本按 Report.md 默认决定；如需扫参可手动设置 WV_SAMPLING/WV_SAMPLING_B/WV_SAMPLING_C。"

echo "[RUN] Forecast ETTh1 stage0"
EPOCHS=10 BATCH=32 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/forecast_etth1_stage0_embed_compare.sh

echo "[RUN] Forecast ETTh1 stage01"
PRIOR_SCALE="${PRIOR_SCALE_ETTH1}" \
  EPOCHS=10 BATCH=32 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/forecast_etth1_stage01_core.sh

echo "[RUN] Forecast ETTm1 stage0"
EPOCHS=10 BATCH=32 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/forecast_ettm1_stage0_embed_compare.sh

echo "[RUN] Forecast ETTm1 stage01"
PRIOR_SCALE="${PRIOR_SCALE_ETTm1}" \
  EPOCHS=10 BATCH=32 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/forecast_ettm1_stage01_core.sh

echo "[RUN] Imputation ETTh1 stage0"
EPOCHS=10 BATCH=16 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/imputation_etth1_stage0_embed_compare.sh

echo "[RUN] Imputation ETTh1 stage01"
PRIOR_SCALE="${PRIOR_SCALE_ETTH1}" \
  EPOCHS=10 BATCH=16 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/imputation_etth1_stage01_core.sh

echo "[RUN] Imputation ETTm1 stage0"
EPOCHS=10 BATCH=16 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/imputation_ettm1_stage0_embed_compare.sh

echo "[RUN] Imputation ETTm1 stage01"
PRIOR_SCALE="${PRIOR_SCALE_ETTm1}" \
  EPOCHS=10 BATCH=16 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/imputation_ettm1_stage01_core.sh

echo "[RUN] Forecast Electricity (ECL/OT) stage01"
PRIOR_SCALE="${PRIOR_SCALE_ECL_OT}" \
  EPOCHS=10 BATCH=32 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/forecast_electricity_stage01_core.sh

echo "[RUN] Anomaly PSM scale_mode compare"
EPOCHS=3 BATCH=128 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/anomaly_psm_stage01_scale_compare.sh
echo "[RUN] Anomaly PSM add embed=wv"
EPOCHS=3 BATCH=128 WORKERS=8 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/anomaly_psm_stage01_add_wv.sh

echo "[RUN] Classification Heartbeat scale_mode compare"
EPOCHS=30 BATCH=16 WORKERS=0 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/classification_heartbeat_stage01_scale_compare.sh
echo "[RUN] Classification Heartbeat add embed=wv"
EPOCHS=30 BATCH=16 WORKERS=0 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/classification_heartbeat_stage01_add_wv.sh

echo "[DONE] final suite finished."
