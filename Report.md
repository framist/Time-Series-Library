# WVEmbs 实验报告（TSLib / PG 分支）

> 更新时间：2026-03-09

## 结论摘要

- WVEmbs 的稳定收益集中在 Forecast，尤其是 Transformer。
- 当前主推荐配置是 `embed=wv, scale_mode=standard, wv_sampling=jss, wv_jss_std=0.25`。
- 数据集存在两类响应模式：
  - ETTh1 / ETTm1 / Weather：`standard + jss(0.25)` 稳定改善。
  - ETTh2 / ETTm2：短期可改善，但长预测上 JSS 共享采样会失效，ISS 或 `extrap` 更鲁棒。
- 常规预处理口径下，Imputation / Anomaly / Classification 暂无稳定收益；无预处理公平对照仍待补齐后再定稿。
- TimeMixer 上 RevIN-only 最优，说明 RevIN 与 WVEmbs 存在明显功能重叠。
- HSPMF 当前“输出层增强”版本显著差于纯 WVEmbs，仅保留为实验分支。

## 运行环境与口径

- OS：Linux（Arch）
- Python：3.11.5（conda `radio`）
- PyTorch：2.9.1+cu128
- GPU：RTX 4060 Laptop GPU（8GB）

统一口径：

- 阶段 0（embed 对照）：不加 `--inverse`，指标在缩放空间。
- 阶段 1（`scale_mode` 对照与主表）：统一加 `--inverse`，比较原始物理量尺度。
- 无预处理公平对照：统一用 `scale_mode=none`，当前实现已支持 `embed=timeF / linear / wv` 三组输入层对照。
- `prior_scale` 默认按 `max(|x_train|) × 2` 初始化。
- 主表三组配置：

| 标签 | 配置 |
|---|---|
| `timeF_std` | `embed=timeF, scale_mode=standard` |
| `wv_std_jss` | `embed=wv, scale_mode=standard, wv_sampling=jss, wv_jss_std=0.25` |
| `wv_none_iss_extrap` | `embed=wv, scale_mode=none, wv_sampling=iss, wv_extrap_mode=scale, wv_extrap_scale=5.0` |

## Forecast 主表（Transformer）

### ETTh1

| pred_len | timeF_std | wv_std_jss | vs baseline | wv_none_iss_extrap | vs baseline |
|---|---|---|---:|---|---:|
| 96 | `26.585 / 3.293` | **`11.683 / 2.023`** | **-56.1%** | `15.866 / 2.706` | -40.3% |
| 192 | `24.324 / 3.185` | **`15.828 / 2.472`** | **-34.9%** | `20.239 / 2.980` | -16.8% |
| 336 | `21.237 / 3.159` | **`16.007 / 2.656`** | **-24.6%** | `24.234 / 3.484` | +14.1% |
| 720 | `30.438 / 3.503` | **`26.824 / 3.519`** | **-11.9%** | `32.744 / 4.196` | +7.6% |

### ETTh2

| pred_len | timeF_std | wv_std_jss | vs baseline | wv_none_iss_extrap | vs baseline |
|---|---|---|---:|---|---:|
| 96 | `105.755 / 8.462` | **`84.970 / 6.985`** | **-19.7%** | `86.061 / 7.767` | -18.6% |
| 192 | `321.578 / 15.208` | `229.832 / 13.284` | -28.5% | **`92.124 / 8.084`** | **-71.4%** |
| 336 | `256.681 / 13.000` | `319.685 / 15.194` | +24.5% | **`97.576 / 8.430`** | **-62.0%** |
| 720 | `136.782 / 9.417` | `403.328 / 14.928` | +194.9% | **`92.941 / 8.161`** | **-32.1%** |

### ETTm1

| pred_len | timeF_std | wv_std_jss | vs baseline | wv_none_iss_extrap | vs baseline |
|---|---|---|---:|---|---:|
| 96 | `10.198 / 1.844` | `8.087 / 1.598` | -20.7% | **`7.464 / 1.588`** | **-26.8%** |
| 192 | `13.726 / 2.073` | **`9.196 / 1.735`** | **-33.0%** | `10.462 / 1.910` | -23.8% |
| 336 | `18.835 / 2.606` | **`11.831 / 2.032`** | **-37.2%** | `17.897 / 2.499` | -5.0% |
| 720 | `23.064 / 3.043` | **`17.505 / 2.597`** | **-24.1%** | `20.311 / 2.649` | -11.9% |

### ETTm2

| pred_len | timeF_std | wv_std_jss | vs baseline | wv_none_iss_extrap | vs baseline |
|---|---|---|---:|---|---:|
| 96 | `29.525 / 3.809` | **`16.644 / 2.922`** | **-43.6%** | `35.283 / 4.457` | +19.5% |
| 192 | `65.911 / 5.672` | **`60.682 / 5.843`** | **-7.9%** | `97.952 / 7.478` | +48.6% |
| 336 | `76.131 / 6.409` | `85.608 / 6.557` | +12.4% | `77.494 / 7.036` | +1.8% |
| 720 | `162.377 / 9.749` | `167.570 / 9.973` | +3.2% | **`106.150 / 8.573`** | **-34.6%** |

### Weather

| pred_len | timeF_std | wv_std_jss | vs baseline | wv_none_iss_extrap | vs baseline |
|---|---|---|---:|---|---:|
| 96 | `7016.614 / 26.169` | **`4172.800 / 22.767`** | **-40.5%** | `64645.300 / 115.299` | +821.3% |
| 192 | `10806.400 / 33.668` | **`6566.800 / 24.978`** | **-39.2%** | `64876.500 / 115.530` | +500.4% |
| 336 | `11511.500 / 35.521` | **`5134.900 / 23.395`** | **-55.4%** | `65140.500 / 115.802` | +465.9% |
| 720 | `17588.700 / 42.323` | **`5182.900 / 24.850`** | **-70.5%** | `65949.600 / 116.552` | +275.0% |

### 主表观察

- `wv_std_jss` 是最稳健的主配置，在 ETTh1 / ETTm1 / Weather 全面获益。
- ETTh2 / ETTm2 属于异质通道数据集，`wv_std_jss` 的共享采样假设在长预测上失效。
- `wv_none_iss_extrap` 与 `wv_std_jss` 呈互补关系：
  - 对 ETTh2 / ETTm2 的部分长预测明显更强。
  - 对 Weather 会灾难性崩溃，不适合作为通用默认项。
- ETTh2 `pred_len=720` 仍是当前最难点，暂无稳定修复方案。

## 多 Backbone 与其他任务

### ETTh1 Forecast 多 Backbone（`pred_len=96`，stage 0）

| backbone | timeF | 最优 WV 配置 | 结论 |
|---|---|---|---|
| Transformer | `0.898969 / 0.750573` | `wv(iss) = 0.592109 / 0.582957` | 主收益来源 |
| TimesNet | **`0.389437 / 0.412179`** | `wv_timeF(jss) = 0.413257 / 0.430636` | 基线更优 |
| Nonstationary_Transformer | `0.569199 / 0.529879` | `wv(jss,std=0.25) = 0.535381 / 0.506600` | 需小 `jss_std` |
| Autoformer | `0.465239 / 0.467973` | `wv_timeF(iss) = 0.402689 / 0.433453` | 调参后可改善 |
| TimeMixer | `0.384408 / 0.398894` | `wv_timeF(jss) = 0.372538 / 0.396425` | 仅小幅改善 |

### 非 Forecast 任务总结

| 任务 | 代表结果 | 结论 |
|---|---|---|
| Imputation | ETTh1 Transformer：`0.064919 / 0.182104 -> 0.060508 / 0.173853` | 只有个别点位小幅改善，不足以作为主结论 |
| Imputation | ETTm1 TimesNet：`0.026248 / 0.108424 -> 0.039180 / 0.132636` | 整体退化 |
| Anomaly | PSM F-score：`0.9741 -> 0.9736 / 0.9658` | 差异极小或略差 |
| Classification | Heartbeat Accuracy：`0.8098 -> 0.7659 / 0.7561` | 明显退化 |

补充说明：

- 上表仍是“常规预处理口径”的结果，不覆盖当前高优先的“无预处理公平对照”。
- `scripts/wvembs/no_preprocess_fair_suite.sh` 与 `embed=linear` 已就位，并通过小预算链路验证；完整预算结果未回填前，不把这些快速数值写入正式主表。

### Electricity 备注

- 当前预算下，`standard + timeF` 仍最好：`91208.65 / 224.12`。
- `none + timeF` 训练直接 NaN，`none + wv` 极不稳定，`prior + wv` 也劣于标准基线。
- 该数据集对 `prior_scale` 极敏感，且成本高，不纳入当前论文主表。

## 消融与扩展

### JSS / ISS / scale_mode

| 场景 | 最优配置 | 指标 | 结论 |
|---|---|---|---|
| ETTh1 全局最优 | `standard + jss + jss_std=0.25` | `11.683 / 2.023` | 当前主推荐配置 |
| ETTh1 prior 组最优 | `prior + jss + jss_std=2.0` | `18.604 / 2.642` | `prior` 偏好更大谱宽 |
| ETTh2 跨数据集验证 | `prior + jss + jss_std=0.5` | `53.602 / 5.570` | 反向交互现象跨数据集成立 |
| ETTh1 `wv_base` 扫描 | `wv_base=100` | `18.441 / 2.585` | 有帮助，但影响小于 `scale_mode` 与 `jss_std` |

要点：

- `standard/none` 偏好小 `jss_std`（约 `0.1-0.25`）。
- `prior` 偏好大 `jss_std`（约 `0.5-2.0`）。
- `jss_std` 与 `scale_mode` 的强交互是论文里必须保留的结论。

### 掩码与外推

| 设置 | ETTh1 | ETTm1 | Weather | 结论 |
|---|---|---|---|---|
| 最佳掩码 `phase_rotate,p=0.1,d=4` | `19.490 / 2.671`，相对 `none+iss` 为 -14.7% | `9.926 / 1.917`，+5.0% | 变化 < 0.1% | 只在 ETTh1 有效，不纳入默认配置 |
| `extrap_scale=5.0` | `15.866 / 2.706`，-30.6% | `9.142 / 2.007`，-3.3% | 变化 < 0.1% | 主要是数值稳定性收益 |
| `mask + extrap` | 未单列 | `8.799 / 1.902`，-6.9% | 变化 < 0.1% | 迁移性有限 |

### RevIN 功能重叠（TimeMixer，ETTh1）

| 配置 | MSE | MAE | 结论 |
|---|---:|---:|---|
| RevIN-only | **7.885** | **1.461** | 最优 |
| RevIN + WVEmbs | 8.001 | 1.495 | 叠加无增益 |
| None | 8.385 | 1.517 | 明显更差 |
| WVEmbs-only | 8.402 | 1.552 | 无法替代 RevIN |

结论：TimeMixer 上 RevIN 已覆盖大部分分布自适应收益，WVEmbs 不应作为其主卖点。

### HSPMF 初步结果（ETTh1）

| 配置 | Test MSE | Test MAE | 相对纯 WVEmbs |
|---|---:|---:|---:|
| Transformer + WVEmbs | **13.91** | **2.34** | baseline |
| Transformer_HSPMF + HSPMF | 27.29 | 3.20 | +96% |

结论：

- 当前输出层增强版 HSPMF 明显退化，不进入默认实验套件。
- 若继续，应优先尝试“推理期解码”或真正的端到端频域训练。

## 产物与后续

保留的关键产物：

- 无预处理公平对照：`scripts/wvembs/no_preprocess_fair_suite.sh`
- 主实验脚本：`scripts/wvembs/forecast_cycle6_table1.sh`
- 退化调参：`scripts/wvembs/forecast_cycle6_tuning.sh`
- RevIN 消融：`scripts/wvembs/forecast_timemixer_revin_ablation.sh`
- HSPMF：`scripts/wvembs/forecast_etth1_hspmf.sh`
- 可视化：`scripts/wvembs/visualize_paper_samples.py` 与 `results/paper_visualizations/`

当前待完成事项：

- 按 `scripts/wvembs/no_preprocess_fair_suite.sh` 完整重跑 ETTh1 / ETTh2 / Weather 的 Forecast 公平对照，并补 Imputation / Anomaly / Classification 的无预处理结果。
- 若继续 HSPMF，先完成已有调参；无收益则切换到“推理期解码”路线。
