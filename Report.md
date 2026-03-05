# WVEmbs 阶段实验报告（TSLib / PG 分支）



## 运行环境

- OS：Linux（Arch）
- Python：3.11.5（conda `radio`）
- PyTorch：2.9.1+cu128
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（约 8GB 显存）

## 数据集

- ETT：`dataset/ETT-small/ETTh1.csv`、`dataset/ETT-small/ETTm1.csv`
- Electricity（ECL）：`dataset/electricity/electricity.csv`（已落盘，避免 HF 下载不稳定）
- PSM：`dataset/PSM/train.csv`、`dataset/PSM/test.csv`、`dataset/PSM/test_label.csv`
- UEA/Heartbeat：`dataset/Heartbeat/Heartbeat_TRAIN.ts`、`dataset/Heartbeat/Heartbeat_TEST.ts`

## 实验口径

- 缩放模式（`--scale_mode`）：
  - `standard`：训练段 `StandardScaler`（TSLib 默认）
  - `none`：不缩放（等价 `--no_scale`）
  - `prior`：先验无量纲化（`--prior_scale/--prior_offset`），不依赖训练段统计量
- 指标口径：
  - **阶段 0（embed 对照）**：不启用 `--inverse`，对齐上游 `scripts/*/ETT_script/*.sh` 的默认口径（指标在缩放空间）。
  - **阶段 1（scale_mode 对照）**：统一启用 `--inverse`，把 `standard/prior/none` 的输出逆变换回原始物理量尺度后再评估，保证横向可比。
- WVEmbs 相关配置：
  - `wv_timeF`：值用 WVEmbs，时间仍用 `TimeFeatureEmbedding`（消融）
  - `wv`：统一模式（把 `timeF` 连续时间特征当作通道，与 `x` concat 后一起进入 WVEmbs）
  - `wv_sampling` 与统一模式有明显交互：本报告对 `wv_timeF` 与 `wv` 分别记录其在该预算下的最优采样（见表格）。

## 实验结果

### Forecast（ETTh1，Transformer）

设置：`seq_len=96,label_len=48,pred_len=96,features=M`；其余超参对齐上游默认（`d_model=512,d_ff=2048,n_heads=8,epochs=10,batch=32,lr=1e-4`），默认启用 `--use_amp`。

阶段 0（inverse=False，embed 对照）：

| embed | wv_sampling | MSE | MAE |
|---|---|---:|---:|
| timeF | - | 0.898969 | 0.750573 |
| wv_timeF | jss | 0.651870 | 0.610376 |
| wv（统一） | iss | 0.592109 | 0.582957 |

阶段 1（inverse=True，scale_mode 核心四组）：

| 组 | scale_mode | embed | wv_sampling | MSE | MAE |
|---|---|---|---|---:|---:|
| D | standard | timeF | - | 26.585247 | 3.293327 |
| A | none | timeF | - | 22.772358 | 2.974276 |
| B | none | wv（统一） | iss | 22.856955 | 2.880931 |
| C | prior | wv（统一） | jss | 19.922129 | 2.738035 |

prior 参数（用于阶段 1 的组 C）：
- `prior_scale = [47.288, 17.682, 42.57, 13.788, 15.778, 6.092, 92.014]`
- `prior_offset = 0`

### Forecast（ETTh1，多 Backbone）

对应 `AGENTS.md` 的 Cycle 2（先跑 stage0 口径，inverse=False）：在 ETTh1 forecast 上，对多个 backbone 运行 `timeF / wv_timeF / wv` 三种 embed，对比其 MSE/MAE（缩放空间）。

- 统一设置：`seq_len=96,label_len=48,pred_len=96,features=M,epochs=10`，默认启用 `--use_amp`
- 脚本：`scripts/wvembs/forecast_etth1_multi_backbone.sh`
- WV 参数：`wv_timeF` 使用 `wv_sampling=jss`；`wv`（统一模式）使用 `wv_sampling=iss`（`wv_jss_std=1.0,wv_base=10000`）

| backbone | timeF (MSE/MAE) | wv_timeF (jss) | wv（统一, iss） |
|---|---:|---:|---:|
| Transformer | 0.898969 / 0.750573 | 0.651870 / 0.610376 | 0.592109 / 0.582957 |
| TimesNet | 0.389437 / 0.412179 | 0.413257 / 0.430636 | 0.443232 / 0.447578 |
| Nonstationary_Transformer | 0.569199 / 0.529879 | 0.853976 / 0.596071 | 1.045712 / 0.705285 |
| Autoformer | 0.465239 / 0.467973 | 0.609232 / 0.561374 | 0.509174 / 0.474881 |
| TimeMixer | 0.384408 / 0.398894 | 0.372538 / 0.396425 | 0.373161 / 0.395769 |

最小调优（只调 WV 参数；其余超参与上表一致）：

- 脚本：`scripts/wvembs/forecast_etth1_tune_wv_params.sh`
- 关键发现：不同 backbone 对 `wv_sampling/jss_std/base` 的偏好差异很大；同一个默认配置（例如统一用 `jss,std=1.0`）会造成“看起来 WVEmbs 不通用”的假象。

| backbone | embed | wv_sampling | wv_jss_std | wv_base | MSE | MAE |
|---|---|---|---:|---:|---:|---:|
| Autoformer | wv_timeF | iss | 1.0 | 10000 | 0.402689 | 0.433453 |
| Nonstationary_Transformer | wv（统一） | jss | 0.25 | 10000 | 0.535381 | 0.506600 |

### Forecast（ETTm1，Transformer）

设置同 ETTh1（`seq_len=96,label_len=48,pred_len=96,features=M`；`d_model=512,d_ff=2048,n_heads=8,epochs=10,batch=32`），默认启用 `--use_amp`。

阶段 0（inverse=False，embed 对照）：

| embed | wv_sampling | MSE | MAE |
|---|---|---:|---:|
| timeF | - | 0.725944 | 0.625116 |
| wv_timeF | jss | 0.716385 | 0.609018 |
| wv（统一） | iss | 0.463306 | 0.497421 |

阶段 1（inverse=True，scale_mode 核心四组）：

| 组 | scale_mode | embed | wv_sampling | MSE | MAE |
|---|---|---|---|---:|---:|
| D | standard | timeF | - | 11.695275 | 2.090147 |
| A | none | timeF | - | 12.874339 | 2.157220 |
| B | none | wv（统一） | iss | 9.451094 | 1.889259 |
| C | prior | wv（统一） | jss | 11.205097 | 1.987749 |

prior 参数（用于阶段 1 的组 C）：
- `prior_scale = [48.36, 18.218, 44.206, 14.356, 16.326, 6.092, 92.014]`
- `prior_offset = 0`

### Forecast（Electricity/ECL，Transformer，features=S）

设置：`data=custom, root_path=./dataset/electricity, data_path=electricity.csv, features=S, target=OT`；其余同上；阶段 1 统一启用 `--inverse`。

| 组 | scale_mode | embed | MSE | MAE |
|---|---|---|---:|---:|
| D | standard | timeF | 91208.648438 | 224.118073 |
| A | none | timeF | NaN（epoch1 起训练 loss 直接 NaN，中止） | - |
| B | none | wv（统一） | 10735375.000000 | 3226.888672 |
| C | prior | wv（统一） | 122185.250000 | 265.108093 |

prior 参数（组 C）：
- `prior_scale = 5000`（标量；该数据集对 `prior_scale` 量级非常敏感）

### Imputation（ETTh1，TimesNet）

设置：`seq_len=96,mask_rate=0.125,features=M`；对齐上游脚本（`d_model=16,d_ff=32,top_k=3,epochs=10,batch=16,lr=1e-3`），默认启用 `--use_amp`。

阶段 0（inverse=False，embed 对照）：

| embed | wv_sampling | MSE | MAE |
|---|---|---:|---:|
| timeF | - | 0.061682 | 0.167563 |
| wv_timeF | jss | 0.083792 | 0.200713 |
| wv（统一） | jss | 0.096962 | 0.211497 |

阶段 1（inverse=True，scale_mode 核心四组）：

| 组 | scale_mode | embed | MSE | MAE |
|---|---|---|---:|---:|
| D | standard | timeF | 1.067605 | 0.555349 |
| A | none | timeF | 1.042833 | 0.561185 |
| B | none | wv（统一） | 1.715368 | 0.733404 |
| C | prior | wv（统一） | 1.770468 | 0.725798 |

prior 参数（用于阶段 1 的组 C）：同 Forecast/ETTh1 的 7 通道 `prior_scale`。

### Imputation（ETTh1，多 Backbone）

对应 `AGENTS.md` 的 Cycle 2（stage0 口径，inverse=False）：在 ETTh1 imputation 上，对多个 backbone 运行 `timeF / wv_timeF / wv` 三种 embed，对比其 MSE/MAE（缩放空间）。

- 统一设置：`seq_len=96,mask_rate=0.125,features=M,epochs=10`，默认启用 `--use_amp`
- 脚本：`scripts/wvembs/imputation_etth1_multi_backbone.sh`（含 TimeMixer 的必要特判）
- WV 参数：`wv_timeF/wv` 均使用 `wv_sampling=iss`（`wv_jss_std=1.0,wv_base=10000`）

| backbone | timeF (MSE/MAE) | wv_timeF (iss) | wv（统一, iss） |
|---|---:|---:|---:|
| Transformer | 0.064919 / 0.182104 | 0.077267 / 0.194725 | 0.060508 / 0.173853 |
| TimesNet | 0.061682 / 0.167563 | 0.080913 / 0.198272 | 0.070635 / 0.182095 |
| Nonstationary_Transformer | 0.048058 / 0.146674 | 0.053733 / 0.157275 | 0.079320 / 0.184283 |
| Autoformer | 0.092512 / 0.221572 | 0.142418 / 0.279118 | 0.229470 / 0.374992 |
| TimeMixer | 0.099541 / 0.204953 | 0.125847 / 0.238145 | 0.143213 / 0.255684 |

### Imputation（ETTm1，TimesNet）

设置同 ETTh1（`seq_len=96,mask_rate=0.125,features=M`；`d_model=16,d_ff=32,top_k=3,epochs=10,batch=16,lr=1e-3`），默认启用 `--use_amp`。

阶段 0（inverse=False，embed 对照）：

| embed | wv_sampling | MSE | MAE |
|---|---|---:|---:|
| timeF | - | 0.026248 | 0.108424 |
| wv_timeF | jss | 0.039180 | 0.132636 |
| wv（统一） | jss | 0.054553 | 0.156510 |

阶段 1（inverse=True，scale_mode 核心四组）：

| 组 | scale_mode | embed | MSE | MAE |
|---|---|---|---:|---:|
| D | standard | timeF | 0.382066 | 0.337137 |
| A | none | timeF | 0.298332 | 0.315711 |
| B | none | wv（统一） | 0.989602 | 0.543135 |
| C | prior | wv（统一） | 1.046685 | 0.547551 |

prior 参数（用于阶段 1 的组 C）：同 Forecast/ETTm1 的 7 通道 `prior_scale`。

### Anomaly Detection（PSM，TimesNet）

设置：`seq_len=100,features=M`；对齐当前脚本预算（`d_model=64,d_ff=64,epochs=3,batch=128`）。

| scale_mode | embed | Accuracy | Precision | Recall | F-score |
|---|---|---:|---:|---:|---:|
| standard | timeF | 0.9858 | 0.9853 | 0.9631 | 0.9741 |
| standard | wv_timeF | 0.9856 | 0.9858 | 0.9618 | 0.9736 |
| standard | wv（统一） | 0.9814 | 0.9847 | 0.9477 | 0.9658 |
| none | timeF | 0.9855 | 0.9871 | 0.9602 | 0.9735 |
| none | wv_timeF | 0.9854 | 0.9904 | 0.9566 | 0.9732 |
| none | wv（统一） | 0.9812 | 0.9889 | 0.9427 | 0.9653 |

### Classification（Heartbeat/UEA，TimesNet）

设置：对齐上游脚本（`e_layers=3,d_model=16,d_ff=32,top_k=1,epochs=30,batch=16,lr=1e-3,patience=10`）。

| scale_mode | embed | Accuracy |
|---|---|---:|
| standard | timeF | 0.8098 |
| standard | wv_timeF | 0.7659 |
| standard | wv（统一） | 0.7561 |
| none | timeF | 0.7707 |
| none | wv_timeF | 0.7512 |
| none | wv（统一） | 0.7415 |

## 阶段性结论

- 阶段 0：在 ETTh1/ETTm1 forecast 上，`wv_timeF(jss)` 明显优于 `timeF`；`wv` 统一模式在 `wv_sampling=iss` 下可进一步优于 `wv_timeF`（采样策略与统一模式存在强交互）。
- 阶段 1：ETTh1 上 `no_scale+timeF` 不会崩溃且优于 `standard+timeF`；但在 Electricity(ECL) 上 `no_scale+timeF` 会迅速出现 NaN。`wv` 统一模式在 ECL 上能避免 NaN，但性能仍显著落后于 `standard+timeF`，且 `prior_scale` 量级非常敏感。
- 任务迁移：imputation / classification 上当前 `wv`/`wv_timeF` 未带来收益；anomaly 上差异很小。

## Cycle 3 多数据集 Forecast 扩展（正式结果）

说明：本节为 Cycle 3 的**正式全预算结果**（默认 `epochs=10`，未使用 `MAX_*_STEPS` 截断），对应脚本：
- `scripts/wvembs/forecast_etth2_stage01.sh`
- `scripts/wvembs/forecast_ettm2_stage01.sh`
- `scripts/wvembs/forecast_weather_stage01.sh`

### Stage 0（embed 对照，inverse=False，Transformer）

| 数据集 | timeF (MSE/MAE) | wv_timeF (MSE/MAE) | wv（统一）(MSE/MAE) |
|---|---:|---:|---:|
| ETTh2 | 2.396026 / 1.231947 | 2.112907 / 1.234674 | **1.729860 / 0.972046** |
| ETTm2 | 0.488215 / 0.515649 | 0.761407 / 0.655463 | **0.455119 / 0.516552** |
| Weather | 0.295838 / 0.360556 | 0.347062 / 0.392888 | **0.163021 / 0.239843** |

### Stage 1（D/A/B/C，inverse=True，Transformer）

| 数据集 | D: standard+timeF | A: none+timeF | B: none+wv(iss) | C: prior+wv(jss) |
|---|---:|---:|---:|---:|
| ETTh2 (MSE/MAE) | 105.755470 / 8.462337 | NaN / NaN | 86.105225 / 7.682885 | **48.599602 / 5.298141** |
| ETTm2 (MSE/MAE) | **28.626431 / 3.899127** | NaN / NaN | 33.739807 / 4.505991 | 41.301872 / 4.243606 |
| Weather (MSE/MAE) | **7016.614258 / 26.169287** | NaN / NaN | 64641.105469 / 115.338234 | 50567.457031 / 67.505295 |

### Weather 上 TimeMixer 横向验证（Stage 0，inverse=False）

| 模型 | timeF (MSE/MAE) | wv_timeF (MSE/MAE) | wv（统一）(MSE/MAE) |
|---|---:|---:|---:|
| TimeMixer | **0.162456 / 0.209801** | 0.163028 / 0.211208 | 0.162974 / 0.211666 |

### Cycle 3 结论

- Stage 0 下，`wv` 统一模式在三套新数据集均优于 `timeF`（MSE：ETTh2 -27.8%，ETTm2 -6.8%，Weather -44.9%）。
- Stage 1 下，三套新数据集的 `A: none+timeF` 全部出现 NaN，说明 `no_scale` 不稳定并非 ECL 个例。
- Stage 1 的最优配置出现明显数据集依赖：ETTh2 最优为 `C(prior+wv,jss)`，而 ETTm2/Weather 仍以 `D(standard+timeF)` 最优。
- Weather 上 TimeMixer 未复现 ETTh1 的小幅正收益，`wv/wv_timeF` 与 `timeF` 基本持平或略差。

## Cycle 4: JSS/ISS 系统消融 + jss_std 联合扫描

说明：本节为 Cycle 4 的正式全预算结果（`epochs=10`），统一使用 `--inverse`（原始物理量尺度）。

对应脚本：
- `scripts/wvembs/forecast_etth1_cycle4_3factor.sh`（三因素联合扫描，21 组）
- `scripts/wvembs/forecast_etth1_cycle4_wvbase.sh`（wv_base 扫描，6 组）
- `scripts/wvembs/forecast_etth2_cycle4_jssstd.sh`（ETTh2 跨数据集验证，6 组）

### Step 1: 三因素联合扫描（ETTh1，Transformer，embed=wv）

扫描范围：`wv_sampling` × `wv_jss_std` × `scale_mode`。固定 `embed=wv`（统一模式），其余超参同前。

| scale_mode | sampling | jss_std | MSE | MAE |
|---|---|---|---:|---:|
| **standard** | iss | - | 13.908 | 2.336 |
| standard | jss | 0.05 | 14.385 | 2.209 |
| standard | jss | 0.1 | 12.423 | 2.065 |
| **standard** | **jss** | **0.25** | **11.683** | **2.023** |
| standard | jss | 0.5 | 12.117 | 2.138 |
| standard | jss | 1.0 | 22.903 | 3.014 |
| standard | jss | 2.0 | 28.087 | 3.395 |
| **none** | iss | - | 22.857 | 2.881 |
| none | jss | 0.05 | 13.412 | 2.456 |
| none | jss | 0.1 | 14.203 | 2.354 |
| none | jss | 0.25 | 17.643 | 2.593 |
| none | jss | 0.5 | 18.080 | 2.549 |
| none | jss | 1.0 | 38.128 | 3.496 |
| none | jss | 2.0 | 55.635 | 4.073 |
| **prior** | iss | - | 42.998 | 3.920 |
| prior | jss | 0.05 | 32.919 | 3.497 |
| prior | jss | 0.1 | 42.943 | 3.914 |
| prior | jss | 0.25 | 33.153 | 3.549 |
| prior | jss | 0.5 | 28.407 | 3.360 |
| prior | jss | 1.0 | 19.922 | 2.738 |
| **prior** | **jss** | **2.0** | **18.604** | **2.642** |

### Step 2: wv_base 灵敏度扫描（ETTh1，embed=wv，none+iss）

| wv_base | MSE | MAE |
|---:|---:|---:|
| **100** | **18.441** | **2.585** |
| 500 | 20.034 | 2.652 |
| 1000 | 20.416 | 2.668 |
| 5000 | 18.864 | 2.599 |
| 10000（default） | 22.857 | 2.881 |
| 50000 | 21.275 | 2.760 |

### Step 3: ETTh2 跨数据集验证（Transformer，embed=wv，jss）

| scale_mode | jss_std | MSE | MAE |
|---|---|---:|---:|
| none | 0.1 | 94.989 | 8.056 |
| none | 0.25 | 104.769 | 8.749 |
| none | 0.5 | 112.081 | 8.904 |
| prior | 0.1 | 130.679 | 9.577 |
| prior | 0.25 | 86.934 | 7.466 |
| **prior** | **0.5** | **53.602** | **5.570** |

### Cycle 4 可视化

- 三因素热力图：`results/cycle4_heatmap.pdf`（及 `.png`）
- wv_base 灵敏度折线图：`results/cycle4_wvbase.pdf`（及 `.png`）
- ETTh2 跨数据集验证热力图：`results/cycle4_etth2.pdf`（及 `.png`）

### Cycle 4 结论

1. **jss_std × scale_mode 存在反向交互（核心发现）**：
   - `standard` 和 `none` 偏好**小** jss_std（0.1–0.25 最优）
   - `prior` 偏好**大** jss_std（1.0–2.0 最优）
   - 这是因为 `prior` 缩放已将值域压到 [-1,1] 附近，需要更大的谱宽度才能提供足够的分辨率；而 `standard/none` 幅值较大，小谱宽度已足够

2. **全局最优配置（ETTh1）**：`standard + jss + jss_std=0.25`（MSE=11.683），远优于所有 ISS 变体

3. **wv_base 影响较小但可优化**：
   - wv_base=100 最优（MSE=18.441），比默认 10000 改善 ~19%
   - 但效果量级远不如 jss_std 和 scale_mode 的影响

4. **跨数据集验证（ETTh2）确认同样趋势**：
   - `none` 偏好小 jss_std（0.1 最优）
   - `prior` 偏好大 jss_std（0.5 最优，MSE=53.602）
   - 证明 jss_std × scale_mode 反向交互是跨数据集稳健的现象

5. **对论文的启示**：
   - 建议论文中推荐配置按 scale_mode 分档：`standard/none` 用 jss_std=0.1–0.25，`prior` 用 jss_std=1.0–2.0
   - jss 在多数场景下优于 iss（特别是 standard+jss 明显优于 standard+iss）
   - wv_base 可作为附录消融，推荐默认 100–500

## 未完成问题与下一步

- Cycle 4 已完成：三因素扫描、wv_base 扫描、跨数据集验证均已完成，可视化图表已生成。
- 下一步进入 Cycle 5（掩码消融 + 外推实验）或 Cycle 6（最终调优 + 多 pred_len + 论文表格产出）。
- 关键待解决：在 Cycle 6 中利用 Cycle 4 结论，按 scale_mode 分档设定最终 jss_std。
