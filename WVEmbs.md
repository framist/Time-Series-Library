# WVEmbs（Wide Value Embedding）在 TSLib 的最小跑通记录

> 更新时间：2026-03-05

## 背景速记（对齐论文表述）

WVEmbs 的核心是把连续物理量视为值域上的 Dirac 测度 \(\delta_x\)，其在对偶谱上的特征函数为 \(\chi_\omega(x)=e^{-i\omega x}\)。用一组有限谱点 \(\{\omega_k\}\) 采样该特征函数即可得到分布无关的嵌入；工程实现上用实值向量拼接 \([\cos(\omega_k x),\ \sin(\omega_k x)]\) 表示复频域特征。

在多变量时间序列上，论文中的 WV-Lift Adapter 采用：
- 逐变量 WVEmbs lifting：\(\mathbf{x}\in\mathbb{R}^{T\times M}\to \mathbf{Z}\in\mathbb{R}^{T\times M\times D}\)
- 多通道交互：沿变量轴做显式混合（1x1 Conv / MLP）
- 形状对齐投影：投影回主干网络期望的输入形状

本仓库先以“最小可跑通”为目标，优先把 WVEmbs 接到现有主干（Transformer / Informer / TimesNet 等）里跑起来。

## 本仓库当前实现（最小版本）

实现要点：
- 仍沿用 TSLib 统一入口 `run.py` 与数据管线。
- 通过扩展 `--embed` 参数启用 WVEmbs，不改动各个模型文件的调用方式。

启用方式（与原有 `timeF/fixed/learned` 完全兼容）：
- `--embed timeF|fixed|learned`：原始行为（TokenEmbedding）
- `--embed wv_timeF|wv_fixed|wv_learned`：启用 WVEmbs + WV-Lift（value embedding），时间特征编码仍按后缀决定（**消融对照**）
- `--embed wv`：**统一模式（论文核心配置）**：将 `x` 与 `x_mark(timeF)` 沿通道维拼接后一起进入 WV-Lift，不再使用 `TemporalEmbedding/TimeFeatureEmbedding`

代码位置：
- `layers/Embed.py`：新增 `WVEmbs` 与 `WVLiftEmbedding`，并在 `DataEmbedding/DataEmbedding_wo_pos` 内按 `--embed` 自动切换
- `utils/embed_utils.py`：解析 `--embed` 的小工具
- `utils/scalers.py`：新增 `scale_mode=prior` 的先验无量纲化 scaler（不依赖训练集统计量）
- `data_provider/data_factory.py`：修正 `timeenc` 判定，使 `wv_timeF` 仍走 `timeF` 的时间特征生成
- `models/TemporalFusionTransformer.py`：同步修正 `timeF` 判定（避免 `wv_timeF` 被误判）
- `exp/exp_basic.py`：设备选择更稳健（CUDA/MPS 不可用时自动回退 CPU，并同步 `args.use_gpu/args.device`）
- `run.py` + `exp/*`：新增 `--max_train_steps/--max_val_steps/--max_test_steps`（算力受限时用于快速对照；已覆盖 forecast/imputation/anomaly/classification/zero-shot test）

当前 WVEmbs 频率集合采用 RoPE 风格的确定性对数频率（log-spaced）作为最小基线；已实现训练期的频域掩码增强（zero/arcsine/phase-rotate + dlow\_limited）。此外提供了**最小版联合谱采样（JSS）**与**相位缩放外推（direct/scale）**开关，用于对齐论文中的方法组件与消融需求。

## 最小跑通（无需数据集的 smoke test）

用于快速验证“能前向 + 能反传 + 不崩”：

```bash
python scripts/wvembs/smoke_forward.py
```

默认会对 `Transformer,Informer,TimesNet` 用 `--embed wv`（统一模式）做一次随机输入的前向与反传。

如果需要顺便覆盖“掩码增强”的代码路径，可加参数，例如：

```bash
python scripts/wvembs/smoke_forward.py --models Transformer --wv_mask_prob 1 --wv_mask_type phase_rotate
```

## 多任务 smoke test（无需数据集）

用于覆盖更多任务分支（forecast/imputation/anomaly/classification）：

```bash
python scripts/wvembs/smoke_tasks.py
```

同样支持掩码参数，例如：

```bash
python scripts/wvembs/smoke_tasks.py --wv_mask_prob 1 --wv_mask_type arcsine
```

## 频域掩码增强（Masking）

掩码只在训练阶段生效（推理时自动关闭），对应论文中 “Frequency Masking / Phase Rotate Masking”。

已接入 `run.py` 参数（默认全部关闭）：
- `--wv_mask_prob`：每个样本启用掩码的概率
- `--wv_mask_type`：`none|zero|arcsine|phase_rotate`
- `--wv_mask_phi_max`：`phase_rotate` 的最大相位扰动（弧度），默认 \(\pi/8\)
- `--wv_mask_dlow_min`：dlow 下界；0 表示覆盖全频段，>0 表示限制只掩码更低频尾部（对应 dlow\_limited 变体）

## 值域外推（direct / scale）

最小实现采用“相位缩放”近似论文中的 scale 外推：当 `--wv_extrap_mode scale` 时，embedding 内部使用 `x / s` 替代 `x`（`s=--wv_extrap_scale`），从而降低相位推进速度，缓解外推域相位折叠。

参数：
- `--wv_extrap_mode`：`direct|scale`
- `--wv_extrap_scale`：缩放因子 \(s>0\)

## 联合谱采样（JSS）

TODO 注意：如果后端网络原生支持多通道 Embs 输入，则不使用通道混合的 WVEmbs，并标明

最小实现提供两种谱采样方式：
- `--wv_sampling iss`：逐变量独立采样（ISS，默认；即先对每个通道做 WVEmbs，再做显式混合）
- `--wv_sampling jss`：联合谱采样（JSS；对多通道向量采样随机频率向量并做内积，再取 cos/sin） 

参数：
- `--wv_sampling`：`iss|jss`
- `--wv_jss_std`：JSS 随机频率向量的标准差（控制角度尺度）

## 最小训练（需要数据集）

在本仓库中，若本地 `root_path/data_path` 不存在，部分数据集 loader（ETT/Custom/PSM/SWaT 等）会尝试通过 `huggingface_hub` 从 HF 数据集仓库 `thuml/Time-Series-Library` 自动下载对应 CSV；若网络/SSL 受限，请手动将文件放到本地目录（例如 `./dataset/ETT-small/ETTh1.csv`）。

Cycle 0 起补充了“下载并落盘”的脚本（推荐用于离线复现与迁移）：

```bash
conda run -n radio python scripts/wvembs/download_datasets.py --all
```

准备好数据后，可参考 README 的 quick test，把 `--embed` 换成 `wv`（统一模式）并减小规模以适配小 GPU：

```bash
python -u run.py \
  --task_name long_term_forecast --is_training 1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv \
  --model_id wvembs_smoke --model Transformer --data ETTh1 --features M \
  --seq_len 96 --label_len 48 --pred_len 96 \
  --enc_in 7 --dec_in 7 --c_out 7 \
  --d_model 64 --d_ff 128 --n_heads 4 --e_layers 2 --d_layers 1 \
  --train_epochs 1 --batch_size 8 --num_workers 2 \
  --max_train_steps 50 --max_val_steps 10 --max_test_steps 10 \
  --embed wv \
  --itr 1
```

对照实验可将 `--embed wv` 改回 `--embed timeF`（传统基线），或改为 `--embed wv_timeF`（消融：值用 WVEmbs、时间仍用传统嵌入）。


## 阶段 0/1 最终结论（2026-03-05）

- 对齐上游口径：
  - 阶段 0（embed 对照）：不传 `--inverse`（指标在缩放空间，对齐 `scripts/*/ETT_script/*.sh` 默认）
  - 阶段 1（scale_mode 对照）：显式传 `--inverse`（把 `standard/prior/none` 的指标统一回原始物理量尺度）
- 最终结果与表格：见 `Report.md`（该文件只保留最终结果，不包含弃用/最小预算尝试）
- 当前已覆盖的真实数据集与任务：
  - Forecast：ETTh1、ETTm1、Electricity(ECL)
  - Imputation：ETTh1、ETTm1
  - Anomaly：PSM
  - Classification：UEA/Heartbeat
- 关键发现（供后续阶段 2/3 设计超参扫描时参考）：
  - ETTh1 forecast 上，`wv_timeF` 在 `wv_sampling=jss` 下显著优于 `timeF`
  - `wv` 统一模式与 `wv_sampling` 有强交互：在 ETTh1 forecast 上 `wv_sampling=iss` 能显著改善 `wv`，但 `jss` 可能导致退化（需要解释原因并形成统一默认）
  - `scale_mode=none` 的“基线崩溃”并非在所有数据集上成立：ETTh1 不崩溃，但 Electricity(ECL) 上 `no_scale+timeF` 会出现 NaN
  - prior 的“物理先验”仍待补齐：ETT 目前用训练段 `max(abs(x))×slack` 做初始化；ECL 上 `prior_scale` 对结果非常敏感（目前观察到 `target=OT` 时 `prior_scale≈5000` 明显优于 `1e5` 量级），需要更稳健且可解释的先验设定流程
  - anomaly/classification 任务通常没有显式时间特征（`x_mark=None`），因此 `embed=wv` 更像是“值 WVEmbs + 额外零时间通道”的补齐项；脚本里已明确标注该解释。

## 各任务最小脚本（需要数据集）

- 数据集下载落盘：`scripts/wvembs/download_datasets.py`
- GPU 预算探测（ETTh1 forecast）：`scripts/wvembs/gpu_budget_probe_etth1.sh`
- Forecast（ETTh1）：`scripts/wvembs/forecast_etth1_backbones.sh`
- Forecast（ETTh1 + no\_scale）：`scripts/wvembs/forecast_etth1_noscale.sh`
- Forecast（ETTh1 + 阶段0 embed 对照）：`scripts/wvembs/forecast_etth1_stage0_embed_compare.sh`
- Forecast（ETTh1 + 阶段0/1核心四组对照）：`scripts/wvembs/forecast_etth1_stage01_core.sh`
- Forecast（ETTh1 + 多-backbone 对照）：`scripts/wvembs/forecast_etth1_multi_backbone.sh`
- Forecast（ETTh1 + WV 参数最小调优）：`scripts/wvembs/forecast_etth1_tune_wv_params.sh`
- Forecast（ETTm1 + 阶段0 embed 对照）：`scripts/wvembs/forecast_ettm1_stage0_embed_compare.sh`
- Forecast（ETTm1 + 阶段0/1核心四组对照）：`scripts/wvembs/forecast_ettm1_stage01_core.sh`
- Forecast（ETTh1 + ISS vs JSS）：`scripts/wvembs/forecast_etth1_jss.sh`
- Imputation（ETTh1）：`scripts/wvembs/imputation_etth1_quick.sh`
- Imputation（ETTh1 + 阶段0 embed 对照）：`scripts/wvembs/imputation_etth1_stage0_embed_compare.sh`
- Imputation（ETTh1 + 阶段0/1核心四组对照）：`scripts/wvembs/imputation_etth1_stage01_core.sh`
- Imputation（ETTh1 + 多-backbone 对照）：`scripts/wvembs/imputation_etth1_multi_backbone.sh`
- Imputation（ETTm1 + 阶段0 embed 对照）：`scripts/wvembs/imputation_ettm1_stage0_embed_compare.sh`
- Imputation（ETTm1 + 阶段0/1核心四组对照）：`scripts/wvembs/imputation_ettm1_stage01_core.sh`
- Anomaly Detection（PSM）：`scripts/wvembs/anomaly_psm_quick.sh`
- Anomaly Detection（PSM + scale_mode 对照）：`scripts/wvembs/anomaly_psm_stage01_scale_compare.sh`
- Anomaly Detection（PSM + 补齐 embed=wv）：`scripts/wvembs/anomaly_psm_stage01_add_wv.sh`
- Classification（Heartbeat/UEA）：`scripts/wvembs/classification_heartbeat_quick.sh`
- Classification（Heartbeat/UEA + scale_mode 对照）：`scripts/wvembs/classification_heartbeat_stage01_scale_compare.sh`
- Classification（Heartbeat/UEA + 补齐 embed=wv）：`scripts/wvembs/classification_heartbeat_stage01_add_wv.sh`

## 备注（与“分布无关”目标的差距）

目前 TSLib 的多数数据集仍默认使用 `StandardScaler`（依赖训练集统计量），因此即便输入 embedding 改为 WVEmbs，也还没有完全实现“端到端分布无关”的数据管线。为更贴近论文设定，本仓库已新增：
- `--scale_mode standard|prior|none`：分别对应 StandardScaler / 物理先验无量纲化 / 不缩放（`--no_scale` 等价于 `--scale_mode none`）
- `--scale_mode prior` 需要提供 `--prior_scale`（可为标量或每通道一个值；`--prior_offset` 可选）
  - 可先用 `python scripts/wvembs/estimate_prior_scale.py ...` 做一个“宽松上界”的初始化，再按物理意义修订并记录
  - 若要在 `standard/prior/none` 三种模式间横向比较指标，建议显式加 `--inverse`（将输出/标签逆变换回原始物理量尺度）
- 更系统的外推与掩码策略（例如只对低频尾部掩码的更细粒度控制、更多外推变体与协议）
- 更严格的多通道联合谱采样（JSS）的物理相关性先验注入（例如协方差结构、与任务相关的采样分布设计）
