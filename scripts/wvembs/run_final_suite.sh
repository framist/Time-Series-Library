#!/usr/bin/env bash
set -euo pipefail

# 运行 WVEmbs 阶段 0/1 的最终实验套件（真实数据集）。
#
# 说明：
# - 各子脚本已支持 `USE_AMP/WV_SAMPLING/WV_JSS_STD/WV_BASE` 等环境变量；
# - 本脚本只负责统一预算与 prior 参数，便于“一键复现最终报告”。

DES="${DES:-WVEmbsFinal_20260304}"
export DES

# 更好利用 GPU：默认开启 AMP + JSS
export USE_AMP="${USE_AMP:-1}"
export WV_SAMPLING="${WV_SAMPLING:-jss}"
export WV_JSS_STD="${WV_JSS_STD:-1.0}"
export WV_BASE="${WV_BASE:-10000.0}"

# prior（range-based prior）：使用训练段 max(|x|) 的宽松上界（slack=2）作为尺度先验
# - ETT 系列数值幅度在 O(10~100)，prior_scale 取 2*max_abs（offset 默认 0）
# - Electricity 的 OT 通道最大约 7.64e5，推荐 prior_scale≈1.528e6
PRIOR_SCALE_ETTH1="${PRIOR_SCALE_ETTH1:-47.288 17.682 42.57 13.788 15.778 6.092 92.014}"
PRIOR_SCALE_ETTm1="${PRIOR_SCALE_ETTm1:-48.36 18.218 44.206 14.356 16.326 6.092 92.014}"
PRIOR_SCALE_ECL_OT="${PRIOR_SCALE_ECL_OT:-1528000}"

echo "[INFO] DES=${DES}"
echo "[INFO] USE_AMP=${USE_AMP} WV_SAMPLING=${WV_SAMPLING} WV_JSS_STD=${WV_JSS_STD} WV_BASE=${WV_BASE}"

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

echo "[RUN] Classification Heartbeat scale_mode compare"
EPOCHS=30 BATCH=16 WORKERS=0 MAX_TRAIN=-1 MAX_VAL=-1 MAX_TEST=-1 bash scripts/wvembs/classification_heartbeat_stage01_scale_compare.sh

echo "[DONE] final suite finished."
