## 目标

背景资料主要参考 [正在书写的论文](../latex) 正在书写的论文内容与 [大论文](siyuan://blocks/20250111201630-mbi59gv) 下属文档（核心理论与进展在 [WVEmbs：Dirac 测度在对偶谱上的取样](siyuan://blocks/20250712140205-01pdiso) 及其相关联的文档（内链文档）

关键字说明：WVEmb (wvemb) === WVEmbs (wvembs)，但优先用 WVEmbs 来指代论文中的方法。

在此储存库一步一步完成 WVEmbs 的时间序列相关实验。

先探索，保证完全理解背景和此代码仓库再动手，你可以修改 AGENTS.md 以 记录信息与经验。

进行 制定计划 -> 进行代码修改 -> 测试（伴随调参再进行测试） -> 整理记录（可以使用 git 在 PG 分支自由推进 commit）的循环直到完成用户要求下的所有所需实验 -- 不同数据集、不同任务、论文中提到的不同方法，
记录必要的实验说明到 WVEmbs.md，实验结果整理并总结到 Report.md，运行实验的可复用经验记录到 AGENTS.md。务必一旦有值得记录的内容就及时总结和记录，不必等全流程完成后再回头总结


## 环境

- 你可以使用 git 在 PG 分支自由推进 commit。
- 使用 conda radio 环境；可以自由安装需要的包；优先使用 cuda 加速；

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


## TODO（后续实验/实现）

注意：计划的细节不一定正确，实现前需要加以判断，可以调整

### 阶段 0（高优先级）：WVEmbs 统一时间嵌入

**目标**：实现"时间作为物理量"的统一 WVEmbs 嵌入，使 `--embed wv` 成为论文核心配置。在 WVEmbs 的语境下，时间也被视为一种物理量，与其他输入物理量定位相同

1. **新增统一嵌入模式 `wv`**：（默认，已完成）
   - 在 `DataEmbedding` / `DataEmbedding_wo_pos` 中，当 `embed=wv` 时：将 `x` 与 `x_mark` 沿通道维 concat → `[B, T, M+D_time]` → `WVLiftEmbedding(c_in=M+D_time, ...)`
   - 不再调用 `TemporalEmbedding` / `TimeFeatureEmbedding`
   - `c_in` 需要动态调整：`enc_in + time_feature_dim`（`time_feature_dim` 由 `freq` 决定，如 `h→4, t→5`）
   
2. **保留兼容模式 `wv_timeF` / `wv_fixed` / `wv_learned`**：（已完成）
   - 作为消融对照组（值用 WVEmbs + 时间用传统嵌入），不删除
   - 论文中可作为消融实验的一个变体

3. **参数与数据管道调整**：（已完成）
   - 由 `--embed wv` 自动决定“时间入通道”，无需新增额外 flag
   - data_factory.py：当统一模式时，`timeenc` 仍用 `timeF` 的连续浮点特征（最适合作为物理量输入 WVEmbs）
   - `time_feature_dim` 已自动由 `freq` 推断（`utils.timefeatures.time_features_dim`）

4. **Smoke test + 快速验证**：（已完成）确认统一模式前向/反传通过

**实验对比组**（ETTh1 + Transformer，验证统一嵌入的效果）：
| 配置 | 值嵌入 | 时间嵌入 | 意义 |
|---|---|---|---|
| `timeF` | TokenEmbedding | TimeFeatureEmbedding | 原始基线 |
| `wv_timeF` | WVEmbs | TimeFeatureEmbedding | 值 WVEmbs + 传统时间（消融） |
| `wv` | WVEmbs（含时间通道） | 无（统一） | **论文核心配置** |

### 阶段 1（高优先级）：no_scale + 物理先验尺度无量纲化（核心论证）

与之前相同，但所有实验改为以 **`wv`（统一模式）** 为主角：

5. **实现物理先验尺度无量纲化**（已完成）：`--scale_mode standard|prior|none` + `--prior_scale/--prior_offset`
6. **扩展 anomaly/classification data loader 的 scale 开关**（已完成）
7. **核心四组对照**：
   - A：`--no_scale` + `timeF` → 基线崩溃
   - B：`--no_scale` + `wv`（统一） → **核心论证点**
   - C：`--scale_mode prior` + `wv` → 最佳实践
   - D：默认 StandardScaler + `timeF` → 传统基线

无量纲化模式 prior 的参数需要根据数据集本身性质进行全局的先验设定（例如 Electricity 的电压范围，Weather 的温度范围等），这是在实验前的一次性工作，并记录参数确定计算过程和结果。确定参数，特别是最大值域需要非常宽松，保证其鲁棒性。

### 阶段 2：JSS 深化

8. **JSS 协方差先验初始化**（特征向量对齐主成分）
9. **JSS std 扫描**
10. **JSS 在统一模式下的重新验证**（`wv` + JSS，通道数变为 M+D_time）

### 阶段 3：更多 backbone + 数据集覆盖

11. 补充 TimeMixer / Nonstationary_Transformer / Mamba 等已适配 backbone
12. 多数据集：ETTh1 / ETTm1 / Weather / Electricity
13. 多任务：Imputation / Anomaly / Classification

### 阶段 4：值域外推实验协议

14. 训练域 vs 外推域分割评估
15. `direct` / `scale` / `log_scale` 变体对比

### 阶段 5：超参调优 + 收尾

16. `wv_base`、学习率、dropout、mixer 结构网格搜索
17. Masking 消融在最优配置下重跑
18. 模型兼容性矩阵标注

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
- prior 的“物理先验”尚未完成：ETT 目前用训练段 `max(abs(x))×slack` 做初始化；ECL 的 `prior_scale` 对结果非常敏感，需要更稳健的先验设定流程（并记录推导过程）。
- GPU 利用率仍偏低：已做 DataLoader/AMP/non_blocking 优化；后续可优先尝试更大 batch、TF32、`torch.compile`、减少训练 loop 的每 step 同步（`loss.item()` 等）以进一步提升吞吐。
