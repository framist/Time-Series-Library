## 目标

背景资料主要参考 [正在书写的论文](../latex) 正在书写的论文内容与 [大论文](siyuan://blocks/20250111201630-mbi59gv) 下属文档（核心理论与进展在 [WVEmbs：Dirac 测度在对偶谱上的取样](siyuan://blocks/20250712140205-01pdiso) 及其相关联的文档（内链文档）

关键字说明：WVEmb (wvemb) === WVEmbs (wvembs)，但优先用 WVEmbs 来指代论文中的方法。

在此储存库一步一步完成 WVEmbs 的时间序列相关实验。

先探索，保证完全理解背景和此代码仓库再动手，你可以修改 AGENTS.md 以 记录信息与经验。

进行 制定计划 -> 进行代码修改 -> 测试（伴随调参再进行测试） -> 整理记录（可以使用 git 在 PG 分支自由推进 commit）的循环直到完成用户要求下的所有所需实验 -- 不同数据集、不同任务、论文中提到的不同方法，
记录必要的实验说明到 WVEmbs.md，实验结果整理并总结到 Report.md，运行实验的可复用经验记录到 AGENTS.md。务必一旦有值得记录的内容就及时总结和记录，不必等全流程完成后再回头总结。这三个文档都可以**清理过时内容**，只保持最新的实验进展和结论。特别避免 AGENTS.md 内容过于冗杂，保持它作为一个“快速信息与经验”文档的清晰和实用。


## 环境

- 你可以使用 git 在 PG 分支自由推进 commit。
- 使用 conda radio 环境；可以自由安装需要的包；优先使用 cuda 加速；
- 根据考虑并行运行实验充分利用 GPU 资源，提高效率

## 对助理的要求

使用中文与用户交互

代码需要有必要的中文注释，类如 docstring

考虑只做总指挥，低难度或检索类的工作可以指派子 agent 完成


# Plan: WVEmbs 全流程实验（修订版 v2，2026-03-05）

注意：计划的细节不一定正确，实现前需要加以判断，可以调整。暂时无法完成可以跳过，但需要记录原因和后续计划。

**TL;DR**：Cycle 0-2 已完成。基于已有实验结论，**聚焦 Forecast 任务**（WVEmbs 的核心优势场景），将剩余工作重组为 Cycle 3-6。核心策略：先**多数据集扩宽覆盖面（Cycle 3）**，再**深入 JSS/采样策略的系统消融（Cycle 4）**，最后**掩码/外推 + 最终调优出表（Cycle 5-6）**。

每完成一个循环阶段，更新 WVEmbs.md Report.md 和 AGENTS.md，确保实验结果和经验得到及时总结和记录。并提交 PG 分支的 commit 来标记每个阶段的完成状态。

---

## 已完成回顾（Cycle 0-2）

### ✅ Cycle 0: 基础设施（已完成）

数据集、prior_scale 估算、GPU 预算探测均已完成。详见"信息与经验"部分。

### ✅ Cycle 1: ETTm1 补齐 + 统一模式全覆盖（已完成）

ETTm1 forecast/imputation + PSM anomaly + Heartbeat classification 均已包含 `embed=wv` 统一模式结果。

### ✅ Cycle 2: 多 Backbone（ETTh1 锚点）（已完成）

5 个 backbone（Transformer / TimesNet / Nonstationary_Transformer / Autoformer / TimeMixer）× 3 种 embed × 2 任务（forecast + imputation）。含最小调优。

### Cycle 0-2 关键结论（指导后续计划）

1. **Forecast 是 WVEmbs 的主战场**：
   - Transformer：ETTh1 MSE 0.899→0.592 (-34%)，ETTm1 MSE 0.726→0.463 (-36%)（`wv(iss)` 统一模式）
   - TimeMixer：0.384→0.373 (-3%)（小幅收益）
   - Autoformer：`wv_timeF(iss)` 0.465→0.403 (-13%)（需指定 iss）
   - Nonstationary_Transformer：需调低 `jss_std=0.25` 才能获益（0.569→0.535）
   - TimesNet：当前预算下 WVEmbs 未带来收益

2. **Imputation / Anomaly / Classification 当前无收益**：
   - Imputation：仅 Transformer 上 `wv` 统一模式小幅改善（0.065→0.061），其余 backbone 均退化
   - Anomaly（PSM）：差异极小（< 1% F-score），`wv` 略低于 `timeF`
   - Classification（Heartbeat）：`wv` 全面落后于 `timeF`（约 3-7%）
   - **决策**：后续深度实验聚焦 forecast；imputation/anomaly/classification 不再扩展新数据集，仅在最终套件中用已有配置产出论文表格

3. **WV 参数与 backbone 强交互**：
   - 不同 backbone 对 `wv_sampling(iss/jss)` 和 `wv_jss_std` 偏好差异大
   - `jss_std=1.0`（默认）对多数场景偏大，`0.1~0.25` 更稳健
   - `scale_mode` × `wv_sampling` 存在交互：B 组(none+wv) 偏好 iss，C 组(prior+wv) 偏好 jss

4. **prior_scale 敏感性**：ETT 系列相对稳定（`max(|x|)×2`），ECL 极敏感（`prior_scale` 量级差一个数量级就会剧烈变化）

---

## 待执行计划（Cycle 3-6）

### Cycle 3: 多数据集 Forecast 扩展（广度优先）

**目标**：在 ETTh2 / ETTm2 / Weather 上复现 Cycle 1-2 的 forecast 实验，证明 WVEmbs 在不同数据分布上的稳健性。

**优先级**：🔴 高（论文核心表格需要多数据集支撑）

**Steps**

1. **ETTh2 forecast**（阶段 0 + 阶段 1 核心四组，Transformer）：
   - 复用 ETTh1 脚本模板，改 `--data ETTh2 --data_path ETTh2.csv`
   - prior_scale：`215.786 72.878 186.46 57.472 34.436 62.924 116.875`
   - 新建脚本 `scripts/wvembs/forecast_etth2_stage01.sh`

2. **ETTm2 forecast**（阶段 0 + 阶段 1 核心四组，Transformer）：
   - 同上，`--data ETTm2 --data_path ETTm2.csv`
   - prior_scale：`215.786 72.878 186.46 59.616 34.436 62.924 116.875`
   - 新建脚本 `scripts/wvembs/forecast_ettm2_stage01.sh`

3. **Weather forecast**（阶段 0 + 阶段 1 核心四组，Transformer）：
   - `--data custom --root_path ./dataset/weather --data_path weather.csv --features M --enc_in 21 --dec_in 21 --c_out 21 --freq t`
   - prior_scale：Cycle 0 已估算的 21 维向量
   - 注意：21 通道 + `d_model=512` 预计显存 ~1-2GB（可接受）
   - 新建脚本 `scripts/wvembs/forecast_weather_stage01.sh`

4. **关键 backbone 横向验证**（仅在 Weather 上追加）：
   - 选 **TimeMixer**（Cycle 2 中 forecast 表现最好的非 Transformer backbone）在 Weather 上跑 `timeF/wv/wv_timeF` 对照
   - 目的：验证"WVEmbs 对 TimeMixer 小幅改善"是否跨数据集成立

5. **结果汇总**：Report.md 新增"多数据集 Forecast 扩展"章节

**实验组数**：ETTh2(7) + ETTm2(7) + Weather(7) + Weather-TimeMixer(3) ≈ **24 组**

**Verification**
- 3 个新数据集各有阶段 0 embed 对照 (timeF/wv_timeF/wv) + 阶段 1 核心四组 (D/A/B/C)
- 跨数据集趋势分析表：WVEmbs 改善率 (%) 的汇总
- Weather 上 TimeMixer 结果与 ETTh1 趋势对比

---

### Cycle 4: JSS / ISS 系统消融 + jss_std 联合扫描

**目标**：系统化理解 `wv_sampling × wv_jss_std × scale_mode` 三因素交互，为论文中"采样策略"一节提供充分数据支撑。

**优先级**：🔴 高（核心理论对应，论文消融实验必须项）

**背景**：Cycle 2 + 阶段性补充已发现 `jss_std` 极敏感、且与 `scale_mode` 强交互，但缺乏系统扫描。

**Steps**

1. **三因素联合扫描**（ETTh1 forecast，Transformer）：
   - 固定 `embed=wv`（统一模式），扫描：
     - `wv_sampling` ∈ {iss, jss}
     - `wv_jss_std` ∈ {0.05, 0.1, 0.25, 0.5, 1.0, 2.0}（仅 jss 生效）
     - `scale_mode` ∈ {standard, none, prior}
   - 口径：`inverse=True`（统一到原始尺度，跨 scale_mode 可比）
   - 共 3×(1 + 6) = **21 组**
   - 扩展已有 `forecast_etth1_jss.sh`

2. **wv_base 扫描**（ETTh1 forecast，Transformer，`embed=wv, scale_mode=none, wv_sampling=iss`）：
   - `wv_base` ∈ {100, 500, 1000, 5000, 10000, 50000}
   - 共 **6 组**（在最优 iss 配置下验证频率基底宽度的影响）

3. **在 Cycle 3 最佳新数据集上重复关键扫描**：
   - 选 Cycle 3 中 WVEmbs 改善最大的数据集（预计 ETTm2 或 Weather）
   - 只扫 `jss_std` ∈ {0.1, 0.25, 0.5} × `scale_mode` ∈ {none, prior}（**6 组**）
   - 验证最优 `jss_std` 是否跨数据集稳定

4. **可视化与解释**：
   - 绘制 `jss_std × scale_mode` 热力图（MSE 值）
   - 绘制 `wv_base` 灵敏度折线图
   - 对 B/C 组偏好 iss/jss 的现象给出定性解释并写入 Report

**实验组数**：21 + 6 + 6 ≈ **33 组**

**Verification**
- 热力图/折线图产出（matplotlib PDF）
- 最优 `jss_std` 区间确认（预期 0.1~0.25）
- `wv_base` 最优区间确认
- 交互现象的解释文字写入 Report.md

---

### Cycle 5: 掩码消融 + 外推实验

**目标**：验证频域掩码增强和值域外推策略的效果（论文消融实验补充项）。

**优先级**：🟡 中（消融完整性需要，但非核心结论）

**Steps**

1. **掩码消融**（ETTh1 forecast，Transformer，`embed=wv, wv_sampling=iss, scale_mode=none`）：
   - `wv_mask_type` ∈ {zero, arcsine, phase_rotate}
   - `wv_mask_prob` ∈ {0.1, 0.3, 0.5}
   - `wv_mask_dlow_min` ∈ {0, 4}
   - 基线 = `wv_mask_prob=0`（无掩码，Cycle 2 已有）
   - 共 3×3×2 = **18 组**有效组合
   - 修改 `forecast_etth1_mask_ablation.sh`

2. **外推实验**（ETTh1 forecast，Transformer，`embed=wv`）：
   - 设计值域外推评估：在测试集上统计样本中 `max(|x|)` 超过训练集 max 的子集，单独报告其 MSE
   - 对比 `wv_extrap_mode=direct` vs `scale`，`wv_extrap_scale` ∈ {1.5, 2.0, 5.0}
   - 共 **4 组**（1 direct + 3 scale）
   - 新建脚本 `scripts/wvembs/forecast_etth1_extrap.sh`
   - 需在 `exp/exp_long_term_forecasting.py` 的 test 阶段增加域内/域外分组统计逻辑

3. **在最佳掩码配置上跨数据集验证**：
   - 若掩码有正面效果，在 ETTm1 + Weather 上各跑 1 组验证（**2 组**）

**实验组数**：18 + 4 + 2 ≈ **24 组**

**Verification**
- 掩码消融表格（mask_type × mask_prob × dlow_min → MSE/MAE）
- 外推实验域内/域外分立 MSE 报告
- 判定掩码/外推是否应纳入最终配置

---

### Cycle 6: 最终调优 + 论文表格产出

**目标**：固定最优配置，在全部目标组合上产出论文最终数据。

**优先级**：🔴 高（论文交付物）

**Steps**

1. **确定最终推荐配置**（基于 Cycle 3-5 结果）：
   - 最终 embed 模式：大概率为 `wv`（统一模式）
   - 最终 wv_sampling：按 backbone 分档推荐（到此步应有明确结论）
   - 最终 wv_jss_std / wv_base：固化
   - 是否启用掩码增强：基于 Cycle 5 结论决定

2. **论文核心表 1：多数据集 Forecast（Transformer, features=M）**：
   - 数据集：ETTh1 / ETTh2 / ETTm1 / ETTm2 / Weather
   - 行：timeF(standard) / wv(none,iss) / wv(prior,jss)
   - 列向包含多个 pred_len（96 / 192 / 336 / 720，对齐论文惯例）
   - 这是新增工作量最大的一步（每数据集 ×4 pred_len ×3 embed = 60 组；但可复用 Cycle 3 的 pred_len=96 结果）
   - **关键决策**：如果 GPU 时间不够覆盖所有 pred_len，优先跑 96 和 336

3. **论文核心表 2：多 Backbone Forecast（ETTh1, pred_len=96）**：
   - 复用 Cycle 2 结果 + 调优后的最新数据
   - 对每个 backbone 标注最优 WV 配置

4. **论文附表：其他任务（Imputation / Anomaly / Classification）**：
   - 复用 Cycle 1 已有结果（不再扩展新数据集）
   - 对"WVEmbs 未带来收益"的任务，在论文中提供分析性解释

5. **模型兼容性矩阵**：更新 WVEmbs.md
   - 已验证兼容：Transformer / TimesNet / Nonstationary_Transformer / Autoformer / TimeMixer / Informer / FEDformer / Mamba / MICN
   - 不兼容（独立嵌入体系）：iTransformer / PatchTST / DLinear / TimeXer / Crossformer

6. **最终提交**：更新 `run_final_suite.sh`，git commit 标记

**实验组数**：~60（多 pred_len）+ ~10（补充/重跑）≈ **70 组**

**Verification**
- 论文核心表 1/2 数据完整
- Report.md 全面更新，含可视化图表
- 论文 LaTeX 表格可直接引用
- 最终 git tag

---

## 实验总量估算与时间规划（修订版）

| Cycle | 状态 | 实验组数(估) | 用途 |
|---|---|---|---|
| 0 | ✅ 已完成 | 2 | 数据准备 + GPU 探测 |
| 1 | ✅ 已完成 | ~16 | ETTm1 补齐 + 统一模式全覆盖 |
| 2 | ✅ 已完成 | ~30 | 多 backbone × embed × 任务 |
| **3** | ⏳ 待执行 | **~24** | **多数据集 Forecast 扩展** |
| **4** | ⏳ 待执行 | **~33** | **JSS/ISS 系统消融** |
| **5** | ⏳ 待执行 | **~24** | **掩码消融 + 外推** |
| **6** | ⏳ 待执行 | **~70** | **最终调优 + 多 pred_len + 论文表格** |
| | | **~151 剩余** | |

- 单实验耗时估算：ETT ~5min，Weather ~8min（21ch），ECL(M) ~15min+
- 剩余 GPU 时间：约 **15-25 小时**（ETT 为主，不含 ECL(M)）
- 可并行策略：Cycle 3 的 3 个数据集互不依赖，可同时跑

## Decisions（修订版）

- **聚焦 Forecast**：Imputation/Anomaly/Classification 不再扩展新实验，复用现有结果入论文附表
- **Backbone 重点**：主力用 Transformer（收益最大）；TimeMixer 作为次要验证（MLP 类代表）；其余 backbone 只保留 Cycle 2 已有 ETTh1 结果
- **ECL(M) 降优先级**：321 通道显存成本高且 prior_scale 极敏感，除非 Cycle 3 完成后仍有余量，否则跳过
- **多 pred_len 是 Cycle 6 的主要新增工作量**：论文惯例需要 96/192/336/720 四档，但可按时间预算取舍（优先 96+336）
- **实验口径不变**：阶段 0 用 inverse=False（对齐上游），阶段 1 用 inverse=True（跨 scale_mode 可比）
- **prior_scale 策略不变**：`max(|x_train|) × 2`


# 信息与经验

## 代码与参数速查

### embed 模式

| `--embed` | 说明 |
|---|---|
| `wv` | **统一模式（核心）**：x 与 timeF 时间特征 concat 后一起进 WVEmbs |
| `wv_timeF` / `wv_fixed` / `wv_learned` | 消融对照：值用 WVEmbs，时间仍用传统嵌入 |
| `timeF` / `fixed` / `learned` | 原始行为（TokenEmbedding） |

### 关键代码位置

- `layers/Embed.py`：`WVEmbs` / `WVLiftEmbedding`，在 `DataEmbedding` / `DataEmbedding_wo_pos` 内按 `--embed` 自动切换
- `data_provider/data_factory.py`：用 `time_embed_type` 判定 `timeenc`（`wv_timeF` 须走 `timeF` 时间特征路径）
- `run.py`：`--scale_mode standard|prior|none`、`--prior_scale/--prior_offset`、`--wv_sampling iss|jss`、`--wv_jss_std`、`--wv_base`
- 掩码增强（训练期）：`--wv_mask_prob` / `--wv_mask_type` / `--wv_mask_phi_max` / `--wv_mask_dlow_min`
- 外推：`--wv_extrap_mode direct|scale`、`--wv_extrap_scale`
- Smoke test：`scripts/wvembs/smoke_forward.py`（前向+反传）、`scripts/wvembs/smoke_tasks.py`（多任务覆盖）

### 实验口径要点

- **阶段 0**（embed 对照）：`inverse=False`，对齐上游口径，指标在缩放空间
- **阶段 1**（scale_mode 对照）：**必须显式加 `--inverse`**，否则 `standard/prior` 与 `none` 的指标不可比
- UEA Classification：`--model_id` 是数据集前缀（如 Heartbeat），实验标签用 `--des`
- `no_scale + timeF` 在 ECL 上会 NaN（ETT 系列不会）

## prior_scale 速查表

策略：`max(|x_train|) × 2`。估算脚本：`scripts/wvembs/estimate_prior_scale.py`

| 数据集 | 维度 | prior_scale |
|---|---|---|
| ETTh1 | 7D | `47.288 17.682 42.57 13.788 15.778 6.092 92.014` |
| ETTm1 | 7D | `48.36 18.218 44.206 14.356 16.326 6.092 92.014` |
| ETTh2 | 7D | `215.786 72.878 186.46 57.472 34.436 62.924 116.875` |
| ETTm2 | 7D | `215.786 72.878 186.46 59.616 34.436 62.924 116.875` |
| Weather | 21D | `2040.14 69.6 618.26 41 200 111.34 48.32 84.2 30.8 49.06 2637.04 19998 45.8 720 22.4 1200 2230.58 4263.52 19998 98.18 19998` |
| Electricity | 321D | 标量 `1.528e+06`（`--reduce global_max`），极敏感 |

## GPU 基准

- 设备：RTX 4060 Laptop (8GB)
- ETTh1 Transformer forecast（epochs=10, batch=32, d_model=512, AMP）：**~11s/epoch，峰值显存 ~869MB**

## 关键实验发现（Cycle 0-2 精华）

### WV 参数与 backbone 偏好

| backbone | forecast 最优 WV 配置 | 备注 |
|---|---|---|
| Transformer | `wv(iss)` | ETTh1 -34%, ETTm1 -36% |
| TimeMixer | `wv(iss)` | 小幅收益 -3% |
| Autoformer | `wv_timeF(iss)` | -13%，需指定 iss |
| Nonstationary_Transformer | `wv(jss, std=0.25)` | 默认 std=1.0 会退化 |
| TimesNet | `timeF` 更优 | 当前预算下 WVEmbs 无收益 |

### jss_std 敏感性（ETTh1 forecast, Transformer, embed=wv）

- 阶段 0（inverse=False）最优：`jss_std=0.25`（MSE=0.525）≈ `0.1`（MSE=0.527）>> `0.5`（MSE=0.583）
- **⚠️ 阶段 0 最优 `jss_std` 不可直接迁移到阶段 1 的 prior 组**：`prior + wv(jss,std=0.25)` MSE=33.15（崩溃）
- 结论：`jss_std` 与 `scale_mode` 强交互，Cycle 4 需分 scale_mode 独立扫描

### 其他任务现状

- Imputation / Anomaly / Classification 中 WVEmbs 均未带来收益（详见 Plan 的"Cycle 0-2 关键结论"）
- 后续不再扩展这些任务的新数据集
