## 目标

背景资料优先参考 [正在书写的论文](../latex) 与思源中的 [大论文](siyuan://blocks/20250111201630-mbi59gv) / [WVEmbs：Dirac 测度在对偶谱上的取样](siyuan://blocks/20250712140205-01pdiso)。[HSPMF 相关实验](../HSPMF) 也可以参考。

本仓库用于逐步完成 WVEmbs 在时间序列任务上的实验、复盘与论文材料整理。默认先确定目前的任务，探索上下文，再做代码修改、测试、完整实验、记录与评估、commit 的循环；三份文档只保留最新结论，不保留已经失效的历史过程稿。

## 环境

- 可在 `PG` 分支自由提交。
- 默认使用 `conda run -n radio ...`。
- 优先使用 CUDA；若环境不支持最优做法，不降级凑合，直接记录阻塞并告知用户。

## 输出要求

- 与用户优先使用中文。
- 代码保留必要中文注释或 docstring。
- 文档、图表、可视化中的解释文字优先中文。

可视化建议：
```python
# 设置 plt 字体以支持中文显示
font_config = {
    "font.sans-serif": ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans", "Arial Unicode MS"],
    "axes.unicode_minus": False,
}
plt.rcParams.update(font_config)
```

## 文档分工

- `AGENTS.md`：协作约束、当前计划、可复用经验。
- `WVEmbs.md`：方法实现、参数、脚本、兼容性速查。
- `Report.md`：最终实验结果、核心表格、消融与结论。

## 当前计划

- 高优先：
  - 补一组“无预处理公平对照”实验：固定同一 backbone、同一训练协议与同一数据划分，仅替换输入层为“原始/线性嵌入”与 `WVEmbs`，并统一关闭数据预处理（至少关闭 `standard/prior` 与 RevIN 一类依赖统计量的变换）。优先在 Forecast 主线的 `Transformer` 上覆盖 ETTh1 / ETTh2 / Weather，目标是回答“双方都不做预处理时，WVEmbs 是否仍具备更好的可训练性、泛化性与性能增益”。若 `none + timeF/linear` 出现 NaN、发散或明显不稳，不视为无效结果，而应作为“WVEmbs 更适合无预处理部署”的证据单独记录。
    - 也要包括无预处理场景下 Imputation / Anomaly / Classification 的实验结果。
    - 为了体现无预处理场景的实际意义，WVEmbs 要求的预处理（对数据集线性变换）可以改为设置宽松的 WVEmbs 内部参数
    - 当前实现进度：已补 `embed=linear` / `linear_timeF` 这类线性输入层基线，并新增 `scripts/wvembs/no_preprocess_fair_suite.sh` 统一入口；已完成小预算链路验证，下一步是完整预算重跑并回填 `Report.md`
  - HSPMF 检查现有实现是否正确合理，特别目前缺少了 End2End-NLL 的训练方法，可以参考 ../HSPMF 中的实验方法和一些已有结果；注意需要用分布拟合相关指标来体现 HSPMF 的优势点；另外也尝试验证“纯 WVEmbs backbone + 推理期 HSPMF 解码”路线。
    - 当前实现进度：已把“正频率谱头 + 共轭对称重建 + `hspmf_loss=nll` + test 侧 `nll/crps` 指标落盘”接入 Forecast 主线；下一步是 ETTh1 完整预算比较 `point-MSE / End2End-NLL / 推理期解码`
  - 类如 cycle 这些实验名在交付论文的图表，不要有这些内部名、脚本与代码名，而是用自然语言描述。族群 A、族群 B 也需要加以说明
- 中优先：
  - 围绕 ETTh2 / ETTm2 的长预测退化继续做“修复型”扫描，优先级顺序是：`wv_sampling` 切换、`wv_extrap_scale`、`jss_std`、必要时再看 `prior_scale`。目标不是再找一个全局默认，而是给出“族群 B 为何需要备选配置”的更强证据。
  - 给主表再补一组真实残差分布或误差随预测步增长曲线，正文目前已有真实样例图，但还缺“误差如何随 horizon 累积”的图。
  - 若需要把“JSS 与 `scale_mode` 强交互”写得更实，可补一个小规模网格图，只画 ETTh1 与 ETTh2 两个代表数据集即可，不必再全数据集铺开。
  - Electricity 的大规模补充扫描；该数据集对 `prior_scale` 极敏感，且显存/时间成本高。
- 低优先：
  - 补统一的工程开销表：在同一 batch size、相同精度策略下，对 `timeF_std`、`wv_std_jss`、`wv_none_iss_extrap` 统计显存峰值、训练吞吐量、单步推理时延。第 5 章现在只给了环境与配置，缺正式 cost table。
  - 对 Forecast 主表做固定 3 个随机种子的重复实验，至少覆盖 ETTh1 / ETTm1 / Weather 的主推荐配置，以及 ETTh2 / ETTm2 的长预测修复点（336、720）。论文正文已经能写“趋势性结论”，但若要写成更硬的结论，需要均值和方差。
  - Forecast 多 backbone 的正式重跑。当前筛查已经足够支持“Transformer 是主战场”的写法，除非论文答辩前明确需要更系统的 backbone 表，否则不建议消耗预算。
  - 附录级的更多合成示意图或案例图，这类材料有助于展示。

## 快速入口

- 核心实现：
  - `layers/Embed.py`
  - `run.py`
  - `exp/exp_long_term_forecasting.py`
  - `layers/HSPMF.py`
  - `models/Transformer_HSPMF.py`
- 核心脚本：
  - `scripts/wvembs/no_preprocess_fair_suite.sh`
  - `scripts/wvembs/run_final_suite.sh`
  - `scripts/wvembs/forecast_cycle6_table1.sh`
  - `scripts/wvembs/forecast_cycle6_tuning.sh`
  - `scripts/wvembs/forecast_groupb_priority_scan.sh`
  - `scripts/wvembs/forecast_timemixer_revin_ablation.sh`
  - `scripts/wvembs/forecast_etth1_hspmf.sh`
  - `scripts/wvembs/forecast_hspmf_e2e.py`
- 可视化：
  - `scripts/wvembs/visualize_paper_samples.py`
  - `scripts/wvembs/visualize_predictions.py`
  - `scripts/wvembs/export_horizon_error_curves.py`
  - `results/paper_visualizations/`

## 实验口径

- 阶段 0（embed 对照）：不加 `--inverse`，指标在缩放空间，对齐上游默认口径。
- 阶段 1（`scale_mode` 对照）：必须显式加 `--inverse`，否则 `standard/prior/none` 不可比。
- 无预处理公平对照：统一用 `scale_mode=none`，优先比较 `embed=timeF / linear / wv`；其中 `linear` 表示“值与 `timeF` 时间特征统一拼接后做线性投影”的输入层基线。
- `prior_scale` 统一按 `max(|x_train|) × 2` 初始化，再按数据集修正。
- `scale_mode=none` 不是稳健基线；`none + timeF` 在 ETTh2 / ETTm2 / Weather / ECL 上都出现过 NaN。
- `zsh` 下包含 `*`、`[]`、反斜杠的参数必须加引号。

## 当前推荐配置

| 场景 | 推荐配置 | 说明 |
|---|---|---|
| 主推荐 | `embed=wv, scale_mode=standard, wv_sampling=jss, wv_jss_std=0.25` | 族群 A（ETTh1 / ETTm1 / Weather）最稳健 |
| 增强 | 主推荐 + `wv_extrap_mode=scale, wv_extrap_scale=5.0` | 用作数值稳定性旋钮，部分退化点可显著修复 |
| 异质通道备选 | `embed=wv, scale_mode=none, wv_sampling=iss, wv_extrap_mode=scale, wv_extrap_scale=5.0` | ETTh2 / ETTm2 的部分长预测更鲁棒，但不适合 Weather |

## prior_scale 速查

策略：`max(|x_train|) × 2`，估算脚本为 `scripts/wvembs/estimate_prior_scale.py`。

| 数据集 | prior_scale |
|---|---|
| ETTh1 | `47.288 17.682 42.57 13.788 15.778 6.092 92.014` |
| ETTm1 | `48.36 18.218 44.206 14.356 16.326 6.092 92.014` |
| ETTh2 | `215.786 72.878 186.46 57.472 34.436 62.924 116.875` |
| ETTm2 | `215.786 72.878 186.46 59.616 34.436 62.924 116.875` |
| Weather | `2040.14 69.6 618.26 41 200 111.34 48.32 84.2 30.8 49.06 2637.04 19998 45.8 720 22.4 1200 2230.58 4263.52 19998 98.18 19998` |
| Electricity | 标量 `1.528e+06`，极敏感 |

## 信息与经验

### 任务与模型

- Forecast 是 WVEmbs 的核心优势场景；其余三类任务目前没有稳定收益。
- Transformer 是主战场；TimeMixer 仅在少数设置下小幅改善，且会被 RevIN 压缩增益空间。
- Autoformer、Nonstationary_Transformer 需要单独调参；TimesNet 当前预算下不适合作为 WVEmbs 主结果。

### 数据集族群

- 族群 A：ETTh1、ETTm1、Weather。`standard + jss(0.25)` 稳定改善。
- 族群 B：ETTh2、ETTm2。短期可改善，但长预测上 JSS 共享采样容易失效；ISS 或 `extrap` 更有机会修复。
- Weather 是特殊样本：`wv_std_jss` 的改善会随 `pred_len` 变大而增强。

### 归一化与稳定性

- `jss_std` 与 `scale_mode` 强交互：
  - `standard/none` 偏好小 `jss_std`（0.1-0.25）。
  - `prior` 偏好大 `jss_std`（0.5-2.0）。
- `wv_extrap_scale` 的作用主要是降低相位折叠，不应表述成 OOD 鲁棒性本身。
- 当 backbone 内部已有 RevIN/Normalize 时，优先避免再叠加 DataLoader 层缩放。

### RevIN 与 HSPMF

- RevIN 消融结论：TimeMixer 上 `RevIN-only` 最优，`RevIN + WVEmbs` 不优于它。
- HSPMF 旧版点预测头在 ETTh1 上曾把基线从 `13.91 / 2.34` 退化到 `27.29 / 3.20`，因此未进入默认实验套件；新版 End2End-NLL 与分布指标已接线，但完整预算重跑尚未完成。

### 维护约定

- 更新实验后，优先同步 `WVEmbs.md`、`Report.md`、`AGENTS.md`，只保留当前有效结论。
- 清理仓库时，优先删除缓存、临时文件、空目录；较大的 `checkpoints/`、`results/`、`test_results/` 除非明确确认，否则不要直接删。
