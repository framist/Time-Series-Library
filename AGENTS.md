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

# 信息与经验

## WVEmbs 在本仓库的落地记录（最小跑通）

- 通过扩展 `--embed` 参数启用 WVEmbs：
  - `wv`：统一模式（时间作为通道输入 WVEmbs）
  - `wv_timeF` / `wv_fixed` / `wv_learned`：消融对照（值用 WVEmbs，时间仍用传统嵌入）
- 代码入口：`layers/Embed.py` 新增 `WVEmbs` / `WVLiftEmbedding`，并在 `DataEmbedding` / `DataEmbedding_wo_pos` 内自动切换 value embedding。
- 数据时间特征：`data_provider/data_factory.py` 必须用解析后的 `time_embed_type` 判定 `timeenc`，否则 `wv_timeF` 会被误判为“非 timeF”，导致时间特征维度不匹配。
- Scale mode：`run.py` 新增 `--scale_mode standard|prior|none` 与 `--prior_scale/--prior_offset`；anomaly/classification loader 也会遵循该开关。
- 对比 `standard/prior/none` 或 `--no_scale` 时建议显式加 `--inverse`：否则 standard/prior 的指标在“缩放空间”，而 none 在“原始空间”，横向数值不可比。
- Imputation 指标口径：`exp/exp_imputation.py` 已支持 `test_data.scale and --inverse` 下的逆变换评估；跨 `standard/prior/none` 对照时同样建议显式加 `--inverse`。
- Classification（UEA）任务中 `--model_id` 用作数据集文件前缀（例如 Heartbeat），不要用它做实验命名；实验标签建议放在 `--des`。
- Smoke test：`scripts/wvembs/smoke_forward.py`（随机输入前向 + 反传，快速验证“能跑通”；默认 `--embed wv`，可用 `--embed wv_timeF` 做消融）。
- 多任务 smoke：`scripts/wvembs/smoke_tasks.py`（forecast/imputation/anomaly/classification 随机输入覆盖；同上）。
- 设备稳健性：`exp/exp_basic.py` 在 CUDA/MPS 不可用时自动回退 CPU，并同步 `args.use_gpu` / `args.device`，避免无 GPU 环境直接崩溃。
  - 掩码增强参数（训练期生效）：`--wv_mask_prob` / `--wv_mask_type` / `--wv_mask_phi_max` / `--wv_mask_dlow_min`。

## Cycle 0 产物：数据集准备与 prior_scale 粗估（2026-03-05）

### 数据集文件（已落盘到 `./dataset/`）

- 下载脚本：`scripts/wvembs/download_datasets.py`
- 已补齐（计划 Cycle 0）：
  - ETT：`dataset/ETT-small/ETTh2.csv`、`dataset/ETT-small/ETTm2.csv`
  - Weather：`dataset/weather/weather.csv`
  - UEA 分类：`dataset/EthanolConcentration/*`、`dataset/FaceDetection/*`（Heartbeat 原已存在）
    - 备注：示例数据集 `HandMovementDirection` 在 HF 仓库中缺失（脚本会 warn 并跳过）

### prior_scale 建议值（slack=2.0，训练段 max(|x|) × slack）

估算脚本：`scripts/wvembs/estimate_prior_scale.py`

- ETTh1 (features=M, 7D)：`47.288 17.682 42.57 13.788 15.778 6.092 92.014`
- ETTm1 (features=M, 7D)：`48.36 18.218 44.206 14.356 16.326 6.092 92.014`
- ETTh2 (features=M, 7D)：`215.786 72.878 186.46 57.472 34.436 62.924 116.875`
- ETTm2 (features=M, 7D)：`215.786 72.878 186.46 59.616 34.436 62.924 116.875`
- Weather (custom, features=M, 21D)：`2040.14 69.6 618.26 41 200 111.34 48.32 84.2 30.8 49.06 2637.04 19998 45.8 720 22.4 1200 2230.58 4263.52 19998 98.18 19998`
- Electricity (custom, features=M, 321D)：建议用 **标量**（global max）`1.528e+06`（脚本 `--reduce global_max`）

### GPU 预算探测（ETTh1 Forecast, Transformer, features=M）

脚本：`scripts/wvembs/gpu_budget_probe_etth1.sh`

- 设备：RTX 4060 Laptop (8GB)
- 配置：`epochs=10, batch=32, d_model=512, d_ff=2048, n_heads=8, e_layers=2, d_layers=1, seq_len=96, pred_len=96, embed=timeF, use_amp=1`
- 训练 epoch 耗时（秒）：`13.47, 10.45, 9.84, 12.67, 11.66, 10.88, 12.34, 8.43, 10.18, 9.18`
  - 总计（仅 epoch 内训练段，不含 vali/test）：`109.11s`
  - 平均：`10.91s/epoch`
- 峰值显存（nvidia-smi 轮询）：`~869 MB`
  - 结果目录：`results/gpu_probe_etth1_20260305_010832/`

### 进度核对（对照 Cycle 0-6）

- ✅ Cycle 0 已完成（数据补齐、prior_scale 粗估、GPU 预算探测；已提交到 PG 分支）
- ✅ Cycle 1 已完成（补齐 ETTm1 forecast/imputation + PSM/Heartbeat 的 `embed=wv`；Report 已更新并提交）
- ✅ Cycle 2 已完成（ETTh1 forecast/imputation 的多-backbone 对照 + 最小调优；Report 已更新并提交）
- ⏳ Cycle 3-6 尚未完成（多数据集扩展、JSS 深化、掩码/外推、超参调优与最终套件）
- ✅ `scripts/wvembs/run_final_suite.sh` 已修订并对齐 `Report.md`：不再全局覆盖 `WV_SAMPLING`，ECL `prior_scale` 默认值为 `5000`，并补跑 PSM/Heartbeat 的 `embed=wv`。

### Cycle 2 发现：WV 参数与 backbone 强交互（2026-03-05）

- Forecast（ETTh1）：
  - Nonstationary_Transformer：默认配置下 `wv_timeF(jss,std=1.0)`/`wv(iss)` 会大幅退化；但 `wv(jss,std=0.25)` 与 `wv_timeF(jss,std=0.25)` 可恢复并优于 `timeF`（见 `scripts/wvembs/forecast_etth1_tune_wv_params.sh` 与 Report 对应表）。
  - Autoformer：`wv_timeF(iss)` 在同预算下显著优于 `timeF`；`wv_timeF(jss)` 则明显退化。
  - TimesNet：在本预算下仍更偏好 `timeF`；调小 `wv_base` 可略微缓解退化但不足以反超。
- Imputation（ETTh1, TimesNet）：
  - `wv_sampling=iss` 明显优于 `jss`（`jss_std=0.25` 在该任务上会显著变差）。
  - `wv_base` 从 `1e4` 调到 `1e2~1e3` 可进一步改善 `wv`（但仍略落后于 `timeF` 基线）。


## Plan: WVEmbs 全流程实验（迭代循环）

注意：计划的细节不一定正确，实现前需要加以判断，可以调整。暂时无法完成可以跳过，但需要记录原因和后续计划。

**TL;DR**：基于 AGENTS.md 的阶段 0-5，将未完成实验组织为 7 个迭代循环（Cycle 0-6），每个循环遵循"计划→代码→测试→记录"的流程。核心目标是同时推进**广度覆盖**（多 backbone × 多数据集 × 多任务）和**深度论证**（无量纲化核心价值 + JSS 深化 + 掩码消融）。兼容模型限制为 WVEmbs 适配的模型（iTransformer / PatchTST / DLinear 均**不兼容**）。

每完成一个循环阶段，更新 WVEmbs.md Report.md 和 AGENTS.md，确保实验结果和经验得到及时总结和记录。并提交 PG 分支的 commit 来标记每个阶段的完成状态。

---

### Cycle 0: 基础设施 — 数据准备 + GPU 预算探测

**目标**：确保所有数据集就绪，探测单实验耗时以校准后续规划。

**Steps**

1. **下载缺失数据集**：通过 HF 自动下载或手动准备以下数据集到 dataset 目录：
   - `dataset/ETT-small/ETTh2.csv` — 自动下载（`--data ETTh2`）
   - `dataset/ETT-small/ETTm2.csv` — 自动下载（`--data ETTm2`）
   - `dataset/weather/weather.csv` — 通过 `--data custom --root_path ./dataset/weather --data_path weather.csv` 触发下载
   - UEA 分类数据集 —— 至少补充 2-3 个常用数据集（如 `EthanolConcentration`、`FaceDetection`、`HandMovementDirection`），通过 HF 自动下载

2. **扩展** estimate_prior_scale.py：当前仅支持 ETT 系列，需新增对 Weather / Electricity（多变量） / 其他 Custom 数据集的 prior_scale 估算，并将结果记录到 AGENTS.md

3. **计算所有数据集的 prior_scale**：对 ETTh1/ETTh2/ETTm1/ETTm2/Weather/Electricity(M) 分别运行估算脚本，确定宽松先验参数并固化

4. **GPU 预算探测**：在 RTX 4060 Laptop(8GB) 上运行一个完整的 ETTh1 Transformer forecast 实验（`epochs=10, batch=32, features=M`），记录：
   - 总训练时间、每 epoch 时间
   - 峰值显存占用
   - 以此估算后续实验总耗时，决定是否需要缩减 epochs 或 batch

**Verification**
- 所有数据集文件存在于 dataset 对应目录
- `estimate_prior_scale.py` 可处理所有目标数据集
- GPU 时间记录写入 AGENTS.md

---

### Cycle 1: 补齐阶段 0/1 缺口 — ETTm1 + 统一模式全覆盖

**目标**：补齐当前 Report.md 中缺失的数据点，让阶段 0/1 结果完整。

**Steps**

1. **Forecast ETTm1**（阶段 0 + 阶段 1，Transformer）：运行已有脚本
   - forecast_ettm1_stage0_embed_compare.sh — embed 对照（`timeF` vs `wv_timeF` vs `wv`）
   - forecast_ettm1_stage01_core.sh — 核心四组（D/A/B/C）

2. **Imputation ETTm1**（阶段 0 + 阶段 1，TimesNet）：运行已有脚本
   - imputation_ettm1_stage0_embed_compare.sh
   - imputation_ettm1_stage01_core.sh

3. **Anomaly Detection PSM 补充 `wv` 统一模式**：当前只测了 `wv_timeF`，需新增：
   - `standard + wv`（统一模式）
   - `none + wv`（统一模式）
   - 修改 anomaly_psm_stage01_scale_compare.sh 或新建脚本

4. **Classification Heartbeat 补充 `wv` 统一模式**：同理，当前只测了 `wv_timeF`，需新增 `wv` 统一模式实验

5. **结果汇总**：更新 Report.md 的对应表格

**Verification**
- ETTm1 forecast/imputation 结果出现在 Report.md
- PSM anomaly 和 Heartbeat classification 表格中包含 `wv` 统一模式行
- 与 ETTh1 已有结果做交叉验证（趋势一致性）

---

### Cycle 2: 多 Backbone 实验 — ETTh1 为锚点

**目标**：证明 WVEmbs 是 backbone-agnostic 的通用嵌入方案。

**选定 Backbones**（基于兼容性 + 代表性）：
| 模型 | 代表类型 | 已有结果 |
|---|---|---|
| **Transformer** | 经典注意力 | ✅ |
| **TimesNet** | 2D 卷积时频 | ✅（仅 imputation/anomaly/classification） |
| **Nonstationary_Transformer** | 非平稳注意力 | ❌ |
| **Autoformer** | 自相关分解 | ❌（仅 smoke test） |
| **TimeMixer** | MLP-Mixer 变体 | ❌ |

**Steps**

1. **新建脚本** `scripts/wvembs/forecast_etth1_multi_backbone.sh`：在 ETTh1 forecast 上，对 5 个 backbone 分别运行 `timeF` / `wv_timeF` / `wv` 三种 embed 模式（共 15 组实验），使用统一超参（`d_model=512, d_ff=2048, epochs=10` 对齐上游默认）

2. **注意 Autoformer** 使用 `DataEmbedding_wo_pos`，确认 `wv` 模式在该路径下正常工作

3. **TimeMixer** 同样使用 `DataEmbedding_wo_pos`，且结构不同（无 decoder），需确认参数适配

4. **结果表格**：横轴 = backbone，纵轴 = embed 模式，指标 = MSE/MAE（inverse=False，标准化空间，对齐上游口径）

5. **同步测 Imputation 任务**：对 5 个 backbone 在 ETTh1 imputation 上重复相同的 3 组 embed 对照（15 组）

**Verification**
- 15×2=30 组实验全部完成
- 生成"backbone × embed"交叉表
- 确认 WVEmbs 在至少 3/5 个 backbone 上有正面或持平的 MSE

---

### Cycle 3: 多数据集扩展 — Weather + ETTh2/ETTm2 + Electricity(M)

**目标**：扩大数据集覆盖面，验证结论在不同数据分布上的稳健性。

**Steps**

1. **Weather forecast**（阶段 0 + 阶段 1 核心四组），backbone 用 Transformer：
   - Weather 是 21 通道 `custom` 数据集，需确定 `enc_in=21, dec_in=21, c_out=21, freq=t`（10min 级）
   - 先验参数：用 Cycle 0 估算的 `prior_scale`
   - 新建脚本 `scripts/wvembs/forecast_weather_stage01_core.sh`

2. **ETTh2 / ETTm2 forecast**（阶段 0 + 阶段 1），复用 ETTh1/ETTm1 脚本模板，只改数据路径和 prior_scale

3. **Electricity 多变量 (features=M)**：当前 ECL 实验是 `features=S` 单变量，补充 `features=M`（321 通道）的实验
   - 显存可能不够用 `d_model=512`，需根据 Cycle 0 探测结果调整

4. **Weather imputation**（阶段 0 + 阶段 1），backbone 用 TimesNet

5. **汇总**：Report.md 新增"多数据集扩展"章节

**Verification**
- Weather / ETTh2 / ETTm2 至少各有阶段 0 embed 对照 + 阶段 1 核心四组结果
- 跨数据集的趋势分析（WVEmbs 在哪些数据集上有收益/持平/负面）

---

### Cycle 4: JSS 深化（阶段 2）

**目标**：系统验证联合谱采样（JSS）的参数敏感性与协方差先验初始化。

**Steps**

1. **ISS vs JSS 系统扫描**：在 ETTh1 forecast 上，固定 `embed=wv`（统一模式），扫描：
   - `wv_sampling=iss` vs `jss`
   - `wv_jss_std` ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
   - `scale_mode` ∈ {standard, none, prior}
   - 共 2 + 5×3 = 17 组（iss 不受 jss_std 影响）
   - 修改已有 forecast_etth1_jss.sh 扩充扫描范围

2. **JSS 协方差先验初始化**：实现 `--wv_jss_cov_init` 选项，用训练数据的协方差矩阵主成分来初始化 JSS 的频率向量方向（而非随机方向），需修改 Embed.py 中的 `WVEmbs` 类

3. **统一模式下 JSS 的通道数变化**：`wv` 模式拼入了时间特征，通道数为 `M+D_time`，确认 JSS 在更高维度下是否仍有效

4. **复现 iss/jss 交互现象的解释**：根据 AGENTS.md 记录的"ETTh1 上 B 组偏好 iss、C 组偏好 jss"，设计消融实验来解释原因

**Verification**
- `jss_std` 扫描产出热力图/折线图
- 协方差初始化 vs 随机初始化的对比结果
- 关于 iss/jss 交互的定性解释写入 Report.md

---

### Cycle 5: 掩码消融 + 外推实验（阶段 4）

**目标**：验证频域掩码增强和值域外推策略的效果。

**Steps**

1. **掩码消融**（ETTh1 forecast，Transformer，`embed=wv`）：修改 forecast_etth1_mask_ablation.sh 覆盖：
   - `wv_mask_type` ∈ {none, zero, arcsine, phase_rotate}
   - `wv_mask_prob` ∈ {0.0, 0.1, 0.3, 0.5}
   - `wv_mask_dlow_min` ∈ {0, 4, 8}（限制掩码频段）
   - 基线 = `wv_mask_prob=0`（无掩码），共约 12-15 组有效组合

2. **外推实验协议**（阶段 4）：
   - 设计"训练域 vs 外推域"分割：在 ETTh1 上，人为将测试集中值域超出训练集范围的样本单独统计
   - 对比 `wv_extrap_mode=direct` vs `scale`，`wv_extrap_scale` ∈ {1.0, 2.0, 5.0}
   - 新建脚本 `scripts/wvembs/forecast_etth1_extrap.sh`

3. **在最优 backbone + 数据集配置上重跑掩码消融**（如果 Cycle 2 发现某个 backbone 表现特别好）

**Verification**
- 掩码消融表格（mask_type × mask_prob × dlow_min）
- 外推实验的域内 vs 域外 MSE 分立报告
- 确认掩码/外推是否带来统计显著的改善

---

### Cycle 6: 超参调优 + 最终收尾

**目标**：在前几轮确定的最优配置上做精细调优，产出论文最终表格。

**Steps**

1. **关键超参网格搜索**（在 Cycle 2/3 表现最好的 backbone + 数据集上）：
   - `wv_base` ∈ {100, 1000, 10000, 100000}
   - 学习率 ∈ {5e-5, 1e-4, 5e-4}
   - `d_model` ∈ {64, 128, 256, 512}（如果显存允许）
   - `wv_jss_std` 精扫（基于 Cycle 4 结果的最优区间）
   - 共约 20-30 组

2. **最终实验套件**：固定最优超参，在所有数据集 × 所有任务上跑一遍最终版，更新 run_final_suite.sh

3. **模型兼容性矩阵**：在 WVEmbs.md 中维护一张表，标注每个 TSLib 模型的 WVEmbs 兼容状态（已验证的已知兼容模型：Transformer / TimesNet / Nonstationary_Transformer / Autoformer / TimeMixer / Informer / FEDformer / Mamba / MICN 等；不兼容：iTransformer / PatchTST / DLinear / TimeXer / Crossformer）

4. **最终 Report.md**：
   - 汇总所有数据集 × backbone × embed 模式的完整表格
   - 绘制关键对比图（可用 matplotlib 生成 PDF）
   - 阶段性结论更新

**Verification**
- 超参网格搜索结果表
- 最终表格与论文 LaTeX 表格对齐
- Git commit 标记最终实验状态

---

### 实验总量估算与时间规划

| Cycle | 实验组数(估) | 用途 |
|---|---|---|
| 0 | 1-2 | 探测 + 数据准备 |
| 1 | ~16 | 补齐 ETTm1 + 统一模式 |
| 2 | ~30 | 多 backbone(5) × embed(3) × 任务(2) |
| 3 | ~24 | 多数据集(3-4) × 核心四组 |
| 4 | ~20 | JSS std 扫描 + 协方差初始化 |
| 5 | ~20 | 掩码消融 + 外推 |
| 6 | ~30 | 超参调优 + 最终版 |
| **合计** | **~140** | |

如果单实验约 5-10 分钟（ETTh1/Transformer/10epoch），总计约 12-24 小时 GPU 时间。大数据集（Weather 21ch / ECL 321ch）耗时更长，需在 Cycle 0 后校准。

---

### Decisions

- **Backbone 选择**：Transformer / TimesNet / Nonstationary_Transformer / Autoformer / TimeMixer（5 个），排除 iTransformer / PatchTST / DLinear（因使用独立嵌入体系，WVEmbs 无法透传）
- **实验口径**：阶段 0 用 inverse=False（对齐上游），阶段 1 用 inverse=True（跨 scale_mode 可比）
- **Cycle 顺序**：先补齐已有脚本的缺口（Cycle 1），再扩展 backbone/数据集（Cycle 2-3），最后做深度分析和调优（Cycle 4-6）
- **prior_scale 策略**：统一用 `max(|x_train|) × 2` 作为宽松上界，将 `estimate_prior_scale.py` 扩展到支持所有数据集并固化参数

## 阶段性总结（2026-03-04）

### 已完成的实现

- 阶段 0：`--embed wv` 统一模式（时间特征作为“物理量通道”拼入 WVEmbs），并保留 `wv_timeF/wv_fixed/wv_learned` 作为消融对照。
- 阶段 1：实现 `--scale_mode standard|prior|none`（含 `--prior_scale/--prior_offset`），并让 anomaly/classification loader 同步遵循该开关。
- 指标口径补齐：`exp/exp_imputation.py` 支持在 `test_data.scale && --inverse` 下做逆变换评估（用于跨 `standard/prior/none` 可比）。
- GPU/吞吐相关：DataLoader 侧加入 `pin_memory/persistent_workers/prefetch_factor`；训练侧多处 `.to(..., non_blocking=True)`；多个任务脚本默认加 `--use_amp`。

### 已完成的真实数据集实验（最终结果见 Report.md）

- Forecast：ETTh1（Transformer，阶段 0 embed 对照 + 阶段 1 核心四组）
- Forecast：Electricity/ECL（Transformer，阶段 1 核心四组；验证 `no_scale` 崩溃现象）
- Imputation：ETTh1（TimesNet，阶段 0 embed 对照 + 阶段 1 核心四组）
- Anomaly Detection：PSM（TimesNet，`standard/none × timeF/wv_timeF`）
- Classification：Heartbeat/UEA（TimesNet，`standard/none × timeF/wv_timeF`）

### 与上游基线口径对齐的要点（避免“对不上表”）

- 上游 ETT 复现脚本（`scripts/*/ETT_script/*.sh`）通常**不显式传 `--inverse`**，因此默认评估在“标准化空间”；本仓库 `run.py` 的 `--inverse` 默认也是 False。
- Forecast 测试阶段只会在 `test_data.scale && args.inverse` 时做 `inverse_transform` 后再算指标；因此要做 `standard/prior/none` 横向对比，建议显式加 `--inverse`。
- Imputation 原始实现默认在“标准化空间”评估（无 inverse 分支）；本仓库已在 `exp/exp_imputation.py` 补齐 inverse 评估以满足阶段 1 对照需求。

### 关键经验与仍未完成的问题

- `wv` 统一模式与 `wv_sampling`/`scale_mode` 有强交互：目前观察到 ETTh1 上 `wv_sampling=iss` 能显著改善 `no_scale + wv`，而 `prior + wv` 更偏好 `jss`（需要系统扫描并决定统一默认）。
- 阶段 1 的“no_scale 基线崩溃”并非对所有数据集都成立：ETTh1 不崩溃，但 Electricity/ECL 上 `no_scale + timeF` 会在 epoch1 直接 NaN。
- prior 的“物理先验”尚未完成：ETT 目前用训练段 `max(abs(x))×slack` 做初始化；ECL（`target=OT`，训练段 `max≈5653`）上 `prior_scale` 对结果非常敏感，已观察到 `prior_scale≈5000` 明显优于 `1e5` 量级（仍需形成可解释的先验设定流程并记录推导过程）。
- GPU 利用率仍偏低：已做 DataLoader/AMP/non_blocking 优化，并将 ETT/Custom 数据缓存为 float32、训练/验证 loss 统计改为“张量累加+每 epoch `.item()` 一次”（减少每 step 同步）；后续可优先尝试更大 batch、TF32、`torch.compile` 等进一步提升吞吐。

## 阶段性补充（2026-03-05）

### JSS std 扫描：ETTh1 Forecast（Transformer，阶段 0，inverse=False）

结论：在统一模式 `--embed wv` 下，`wv_sampling=jss` 的 `wv_jss_std` 对结果极其敏感，且最优区间显著小于 `1.0`。

记录（来自 `result_long_term_forecast.txt` 中 `JSSStdScan` 条目；同一训练预算/口径下的对比）：

- `embed=wv, wv_sampling=jss`
  - `wv_jss_std=0.25`：MSE=0.525155，MAE=0.518654
  - `wv_jss_std=0.1`：MSE=0.526762，MAE=0.511221（MAE 更优，但 MSE 略差）
  - `wv_jss_std=0.5`：MSE=0.582721，MAE=0.557427（明显退化）
- `embed=wv_timeF, wv_sampling=jss`
  - `wv_jss_std=0.25`：MSE=0.547726，MAE=0.530639（相较此前 `wv_jss_std=1.0` 的 `wv_timeF(jss)` 明显改善）

对比：当前 `Report.md` 的阶段 0“最终表”仍采用 `wv(iss)` 与 `wv_timeF(jss,std=1.0)`；上述扫描结果尚未写入 `Report.md`（先作为阶段性发现保存在这里，等待确定新的最终配置与可复现脚本后再更新 Report）。

### 交互现象（重要）

同样在 ETTh1 Forecast 的阶段 1（inverse=True、`scale_mode=prior` 的组 C）下，直接沿用 `jss_std=0.25` 会显著变差：
- `C: prior + wv(jss,std=0.25)`：MSE=33.1530，MAE=3.5491（远差于当前报告中 prior + wv 的结果）

提示：`wv_jss_std` 与 `scale_mode/是否 inverse` 可能存在强交互；阶段 0 的最优 `jss_std` 不应直接迁移到阶段 1 的 prior 对照里，需要分别扫描或设计解释性消融。
