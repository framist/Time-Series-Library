# WVEmbs 正式实验报告（TSLib / PG 分支）

> 本报告由仓库内脚本与 `run.py` 产生的日志/指标整理而来，面向“本机（RTX 4060 Laptop 8GB）可承受的训练预算”。  
> 日期：2026-03-02

## 运行环境

- OS：Linux（Arch）
- Python：3.11
- PyTorch：2.9.1+cu128
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU（约 7.6GB 显存）
- CPU：13th Gen Intel i9-13900H
- 内存：约 30GB

## 数据集准备

本次实验使用（或缓存）以下数据文件：

- ETT：`dataset/ETT-small/ETTh1.csv`、`dataset/ETT-small/ETTm1.csv`
- PSM：`dataset/PSM/train.csv`、`dataset/PSM/test.csv`、`dataset/PSM/test_label.csv`
- UEA/Heartbeat：`dataset/Heartbeat/Heartbeat_TRAIN.ts`、`dataset/Heartbeat/Heartbeat_TEST.ts`

说明：
- 若本地文件不存在，部分 loader 会尝试从 HF 数据集 `thuml/Time-Series-Library` 自动下载；网络不稳定时建议提前放置到 `dataset/`。

## 统一实验约定

- 对照核心：同一主干网络、同一训练预算下，对比 `--embed timeF` 与 `--embed wv_timeF`
- 训练预算：根据任务与显存约束设定（见各小节命令），默认启用 GPU
- 指标口径：
  - Forecast/Imputation：`mse`、`mae`（来自 TSLib 默认输出）
  - Anomaly：Accuracy / Precision / Recall / F-score（来自默认输出）
  - Classification：Accuracy（来自默认输出）

## 实验结果

### 长时预测（ETTh1）主干网络对照

设置：
- 任务：`long_term_forecast`（`seq_len=96,label_len=48,pred_len=96`，`features=M`，`freq=h`）
- 训练：`train_epochs=3,batch_size=16,lr=1e-4,num_workers=4`
- 模型：`d_model=64,d_ff=128,n_heads=4,e_layers=2,d_layers=1,factor=3`

结果（MSE/MAE 越小越好；Δ 为 `wv_timeF - timeF`）：

| 模型 | timeF MSE | timeF MAE | wv_timeF MSE | wv_timeF MAE | ΔMSE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|
| Autoformer | 0.510953 | 0.469594 | 0.426242 | 0.442946 | -0.084711 | -0.026648 |
| FEDformer | 0.410093 | 0.440166 | 0.475925 | 0.474007 | 0.065833 | 0.033841 |
| Informer | 0.888499 | 0.693660 | 1.153928 | 0.810152 | 0.265429 | 0.116492 |
| TimesNet | 0.409573 | 0.419543 | 0.451166 | 0.452910 | 0.041593 | 0.033367 |
| Transformer | 0.940712 | 0.759465 | 1.158875 | 0.802239 | 0.218163 | 0.042775 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/forecast_etth1_backbones.tsv`
- `results/wvembs_report_20260302/logs/long_term_forecast_ETTh1_*`

### 长时预测（ETTm1）跨数据集对照

设置：
- 任务：`long_term_forecast`（`seq_len=96,label_len=48,pred_len=96`，`features=M`，`freq=t`）
- 训练：`train_epochs=3,batch_size=16,lr=1e-4,num_workers=4`
- 模型：`Transformer`（`d_model=64,d_ff=128,n_heads=4,e_layers=2,d_layers=1,factor=3`）

结果：

| 模型 | timeF MSE | timeF MAE | wv_timeF MSE | wv_timeF MAE | ΔMSE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|
| Transformer | 0.602886 | 0.539150 | 0.784893 | 0.624808 | 0.182007 | 0.085659 |

备注：
- 原计划补充 `TimesNet`，但 ETTm1 训练步数显著更多（约为 ETTh1 的 4 倍），在当前预算下耗时过长，本轮先不做该项。

原始日志与解析 TSV：
- `results/wvembs_report_20260302/forecast_ettm1.tsv`
- `results/wvembs_report_20260302/logs/long_term_forecast_ETTm1_*`

### Masking 消融（ETTh1 + Transformer）

设置：
- Backbone：Transformer，`embed=wv_timeF`
- `wv_mask_prob=0.75`，`wv_mask_phi_max=pi/8`

结果：

| mask_type | mask_prob | dlow_min | MSE | MAE |
|---|---:|---:|---:|---:|
| none | 0.75 | 0 | 1.158875 | 0.802239 |
| zero | 0.75 | 0 | 1.126555 | 0.793911 |
| arcsine | 0.75 | 0 | 1.115022 | 0.783431 |
| phase_rotate | 0.75 | 0 | 1.151436 | 0.799291 |
| phase_rotate_dlow | 0.75 | 24 | 1.152383 | 0.799558 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/forecast_etth1_masking.tsv`
- `results/wvembs_report_20260302/logs/long_term_forecast_ETTh1_Transformer_wv_timeF_mask_*`

### ISS vs JSS（ETTh1 + Transformer）

设置：
- Backbone：Transformer，`embed=wv_timeF`
- `wv_sampling ∈ {iss,jss}`，`wv_jss_std=1.0`

结果：

| sampling | jss_std | MSE | MAE |
|---|---:|---:|---:|
| iss | 1.0 | 1.158875 | 0.802239 |
| jss | 1.0 | 0.966115 | 0.732580 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/forecast_etth1_sampling.tsv`
- `results/wvembs_report_20260302/logs/long_term_forecast_ETTh1_Transformer_wv_timeF_*ss.log`

### no_scale（ETTh1 + Transformer）

设置：
- Backbone：Transformer
- 关闭数据集级标准化：`--no_scale`

结果：

| embed | MSE | MAE |
|---|---:|---:|
| timeF | 22.610636 | 3.109319 |
| wv_timeF | 28.241539 | 3.207755 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/forecast_etth1_noscale.tsv`
- `results/wvembs_report_20260302/logs/long_term_forecast_ETTh1_Transformer_*_no_scale.log`

### Imputation（ETTh1）

设置：
- 任务：`imputation`（`seq_len=96,mask_rate=0.125`）
- Backbone：TimesNet（`d_model=32,d_ff=64,e_layers=2,top_k=3`）
- 训练：`train_epochs=3,batch_size=32,lr=1e-4,num_workers=4`

结果：

| embed | MSE | MAE |
|---|---:|---:|
| timeF | 0.081190 | 0.190553 |
| wv_timeF | 0.174784 | 0.295634 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/imputation_etth1.tsv`
- `results/wvembs_report_20260302/logs/imputation_ETTh1_*`

### Anomaly Detection（PSM）

设置：
- 任务：`anomaly_detection`（`seq_len=100`）
- Backbone：TimesNet（`d_model=32,d_ff=64,e_layers=2,top_k=3`）
- 训练：`train_epochs=3,batch_size=128,lr=1e-4,num_workers=4`

结果：

| embed | Accuracy | Precision | Recall | F-score |
|---|---:|---:|---:|---:|
| timeF | 0.9803 | 0.9839 | 0.9443 | 0.9637 |
| wv_timeF | 0.9685 | 0.9840 | 0.9011 | 0.9408 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/anomaly_psm.tsv`
- `results/wvembs_report_20260302/logs/anomaly_detection_PSM_*`

### Classification（Heartbeat / UEA）

设置：
- 任务：`classification`（UEA）
- 数据：Heartbeat（`model_id=Heartbeat`）
- Backbone：TimesNet（`e_layers=3,d_model=16,d_ff=32,top_k=1`）
- 训练：`train_epochs=30,batch_size=16,lr=1e-3,patience=10,num_workers=0`

结果：

| embed | Accuracy |
|---|---:|
| timeF | 0.8098 |
| wv_timeF | 0.7756 |

原始日志与解析 TSV：
- `results/wvembs_report_20260302/classification_heartbeat.tsv`
- `results/wvembs_report_20260302/logs/classification_Heartbeat_*`

## 小结与下一步

本轮在“默认数据标准化（StandardScaler）+ 固定训练预算”的设定下，WVEmbs（`wv_timeF`）并未稳定优于基线（`timeF`），且在多个任务/主干上出现明显退化；其中：

- 在 ETTh1 上，WVEmbs 仅在 Autoformer 上带来明显提升；在 Transformer/Informer 上退化较大。
- 在 Transformer + ETTh1 的 ISS/JSS 消融中，JSS（最小实现）比 ISS 更好，提示“联合谱采样”方向值得继续做更严格版本（相关性先验、协方差结构等）。
- 关闭 `StandardScaler` 的 no\_scale 设定下，二者均显著崩溃，说明“端到端分布无关”还需要更完整的无量纲化/尺度先验方案与训练协议支持。

下一步建议（在本机预算内可继续推进）：
- 针对 WVEmbs 做更合理的训练超参（学习率、dropout、`wv_base`、以及 WV-Lift 的 mixer/proj 结构）与更公平的预算（例如更长训练或更小模型）。
- 把 `wv_extrap_mode=scale` 的评估协议做成“训练域 vs 外推域”的严格对照（当前实现更像 embedding 侧的全局缩放开关）。
- 对于 classification / anomaly 任务，补充更多 UEA/工业数据集，检验 JSS 是否更可能在“多通道相关性强”的任务中收益。

