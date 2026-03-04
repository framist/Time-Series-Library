# WVEmbs 阶段 0/1 实验报告（TSLib / PG 分支）

> 更新时间：2026-03-05  
> 目标：推进 `AGENTS.md` 的阶段 0/1（统一时间嵌入 + `scale_mode` 对照），并在真实数据集上给出可复现的最终结果。

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

## 未完成问题与下一步

- `wv` 统一模式：需要解释并系统化 `wv_sampling` 与 `scale_mode` 的交互（目前 ETTh1 上 B 组偏好 `iss`，C 组偏好 `jss`）。
- prior 的“物理先验”尚未完成：ETT 目前用的是训练段 `max(abs(x))×2` 的初始化；ECL 目前是手调量级（1e5）以避免离谱尺度。
- GPU 利用率仍偏低：已加入 DataLoader `pin_memory/persistent_workers/prefetch_factor`、训练侧 `non_blocking` 与 AMP；后续可尝试更大 batch、TF32/`torch.compile`、减少训练 loop 的每 step 同步等。
- 覆盖面：阶段 0/1 已覆盖 ETTh1/ETTm1/ECL；Weather/ETTh2/ETTm2 等留待阶段 3 扩展。
