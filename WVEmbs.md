# WVEmbs 在 TSLib 的实现速查

> 更新时间：2026-03-09

## 方法概览

WVEmbs 将连续值视为值域上的 Dirac 测度，在对偶谱上采样其特征函数；工程实现使用实值形式
`[cos(ωx), sin(ωx)]` 表示复频域特征。当前仓库走的是“最小侵入式接入”路线：尽量复用 TSLib 原有数据流、训练入口和 backbone，只在 embedding 与相关实验开关处扩展。

## 仓库中的实现

### embed 模式

| `--embed` | 含义 |
|---|---|
| `timeF` / `fixed` / `learned` | 原始 TSLib 行为 |
| `linear` / `linear_timeF` / `linear_fixed` / `linear_learned` | 线性输入层基线；其中 `linear` 为统一模式，把值与 `timeF` 时间特征拼接后做线性投影 |
| `wv_timeF` / `wv_fixed` / `wv_learned` | 值走 WVEmbs，时间特征仍走原始时间嵌入 |
| `wv` | 统一模式；把值与 `timeF` 时间特征拼接后一起送入 WVEmbs |

### 关键参数

| 参数 | 作用 |
|---|---|
| `--scale_mode standard|prior|none` | 数据层缩放策略 |
| `--prior_scale` / `--prior_offset` | `scale_mode=prior` 的先验无量纲化参数 |
| `--wv_sampling iss|jss` | 逐通道独立采样 / 联合谱采样 |
| `--wv_jss_std` | JSS 采样尺度 |
| `--wv_base` | 对数频率基底 |
| `--wv_mask_prob` / `--wv_mask_type` / `--wv_mask_dlow_min` | 训练期掩码增强 |
| `--wv_extrap_mode direct|scale` / `--wv_extrap_scale` | 值域外推与相位缩放 |
| `--use_hspmf` / `--hspmf_loss mse|nll` / `--hspmf_learn_beta` 及其余 `--hspmf_*` | HSPMF 实验开关、训练目标与 beta 学习方式 |

### 关键代码位置

- `layers/Embed.py`：`WVEmbs`、`WVLiftEmbedding`、`DataEmbedding*` 的切换逻辑
- `data_provider/data_factory.py`：`wv_timeF` 的 `timeenc` 路径
- `run.py`：CLI 参数入口
- `exp/exp_long_term_forecasting.py`：Forecast 训练、测试、外推评估逻辑
- `layers/HSPMF.py`、`models/Transformer_HSPMF.py`：HSPMF 实验实现

## 实验口径

- 阶段 0（embed 对照）：不加 `--inverse`，指标在缩放空间。
- 阶段 1（`scale_mode` 对照）：统一加 `--inverse`，比较原始物理量尺度上的指标。
- 无预处理公平对照：统一使用 `scale_mode=none`，优先比较 `timeF / linear / wv` 三种输入层；`linear` 用来更严格地回答“只替换输入层后，WVEmbs 是否仍有优势”。
- `prior_scale` 默认按 `max(|x_train|) × 2` 初始化。
- `scale_mode=none` 只是对照项，不是稳健默认项。

## 当前推荐配置

| 用途 | 配置 | 适用结论 |
|---|---|---|
| 主推荐 | `embed=wv, scale_mode=standard, wv_sampling=jss, wv_jss_std=0.25` | Forecast 主结果；对 ETTh1 / ETTm1 / Weather 最稳健 |
| 增强 | 主推荐 + `wv_extrap_mode=scale, wv_extrap_scale=5.0` | 可作为数值稳定性增强；部分退化点有效 |
| 异质通道备选 | `embed=wv, scale_mode=none, wv_sampling=iss, wv_extrap_mode=scale, wv_extrap_scale=5.0` | ETTh2 / ETTm2 部分长预测更鲁棒 |

## 经验总结

### 数据集族群

- 族群 A：ETTh1、ETTm1、Weather。它们的通道统计关系相对更稳定，使用“标准化 + 联合谱采样（`jss_std=0.25`）”通常能稳定改善。
- 族群 B：ETTh2、ETTm2。它们的通道异质性更强、长预测更容易退化，JSS 共享采样更容易失效，ISS 或 `extrap` 更值得尝试。

### backbone 偏好

| backbone | 当前最优或最稳妥结论 |
|---|---|
| Transformer | `wv` 收益最大，是主展示 backbone |
| TimeMixer | 小幅收益且与 RevIN 重叠，默认不作为 WVEmbs 主卖点 |
| Autoformer | `wv_timeF + iss` 可改善，但要单独调参 |
| Nonstationary_Transformer | 需较小 `jss_std` 才可能获益 |
| TimesNet | 当前预算下 `timeF` 更稳妥 |

### 掩码、外推、RevIN、HSPMF

- 掩码增强中 `phase_rotate` 在 ETTh1 上最佳，但跨数据集不稳，不进入默认配置。
- `wv_extrap_scale` 本质是数值稳定性旋钮，不应表述为“自动获得 OOD 鲁棒性”。
- TimeMixer 上 RevIN-only 最优，说明 WVEmbs 与强归一化机制有功能重叠。
- HSPMF 当前实验分支已支持“正频率谱头 + 共轭对称重建 + End2End-NLL”；测试阶段会额外写出 `results/.../hspmf_dist_metrics.json`，记录 `nll / crps / beta`。
- 现有已完成的 ETTh1 退化结果主要对应旧版点预测头；新版 End2End-NLL 仍在完整预算重跑中。

## 兼容性矩阵

| 状态 | 模型 |
|---|---|
| 已验证可接入 | Transformer、Informer、FEDformer、Mamba、MICN、TimesNet、TimeMixer、Autoformer、Nonstationary_Transformer |
| 不适合直接套当前实现 | iTransformer、PatchTST、DLinear、TimeXer、Crossformer |

说明：后一类通常拥有独立的 patch / channel / inverted embedding 体系，不能直接复用当前 `DataEmbedding` 接口。

## 脚本速查

### 冒烟与工具

- `scripts/wvembs/smoke_forward.py`
- `scripts/wvembs/smoke_tasks.py`
- `scripts/wvembs/estimate_prior_scale.py`
- `scripts/wvembs/download_datasets.py`

### 主实验

- 无预处理公平对照：`scripts/wvembs/no_preprocess_fair_suite.sh`
  - 仅补跑 WVEmbs 宽松参数时，可用：`FAIR_EMBEDS="wv" WV_EXTRAP_MODE=scale WV_EXTRAP_SCALE=5.0 bash scripts/wvembs/no_preprocess_fair_suite.sh`
- 无预处理结果汇总：`scripts/wvembs/summarize_no_preprocess_results.py`
  - 当前会同时输出 `vs_raw_timeF` 与 `vs_linear`，方便在 `raw_timeF` 发散时直接比较 `linear` 与 `WVEmbs`
- 逐预测步误差曲线导出：`scripts/wvembs/export_horizon_error_curves.py`
- Forecast 主表生成：`scripts/wvembs/forecast_cycle6_table1.sh`
- Forecast 退化点修复扫描：`scripts/wvembs/forecast_cycle6_tuning.sh`
- 族群 B 高优先修复扫描：`scripts/wvembs/forecast_groupb_priority_scan.sh`
- 当前完整实验套件入口：`scripts/wvembs/run_final_suite.sh`

### 关键消融

- 因子交互与 `wv_base` 扫描：`scripts/wvembs/forecast_etth1_cycle4_3factor.sh`、`scripts/wvembs/forecast_etth1_cycle4_wvbase.sh`
- ETTh2 的 `jss_std` 修复扫描：`scripts/wvembs/forecast_etth2_cycle4_jssstd.sh`
- 掩码与外推消融：`scripts/wvembs/forecast_etth1_cycle5_mask.sh`、`scripts/wvembs/forecast_etth1_cycle5_extrap.sh`
- 交叉验证：`scripts/wvembs/forecast_cycle5_crossval.sh`
- RevIN 功能重叠消融：`scripts/wvembs/forecast_timemixer_revin_ablation.sh`
- HSPMF 验证：`scripts/wvembs/forecast_etth1_hspmf.sh`、`scripts/wvembs/forecast_hspmf_e2e.py`

### 可视化

- 退化修复相关作图：`scripts/wvembs/plot_cycle4.py`、`scripts/wvembs/plot_cycle5.py`
- 逐预测步误差曲线：`scripts/wvembs/export_horizon_error_curves.py`
- 论文版式示意图：`scripts/wvembs/visualize_paper_samples.py`

## 已知限制

- 常规预处理口径下，Forecast 之外的三类任务暂无稳定正收益；但“无预处理公平对照”仍在补跑，相关结论未最终锁定。
- Electricity 对 `prior_scale` 极敏感，且成本高；目前不纳入主表。
- `exp/exp_long_term_forecasting.py` 中阈值估计顺序问题仍需修复后重跑受影响结果。
- HSPMF 若继续推进，应优先尝试“推理期解码”，不要继续把输出层补丁当作主线。
