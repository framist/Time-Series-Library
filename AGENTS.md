
背景资料主要参考 [大论文](siyuan://blocks/20250111201630-mbi59gv) 下属文档（核心理论与进展在 [WVEmbs：Dirac 测度在对偶谱上的取样](siyuan://blocks/20250712140205-01pdiso) 及其相关联的文档（内链文档）


使用 conda radio 环境；可以自由安装需要的包；优先使用 cuda 加速；

在此子目录存在 git 环境，使用 PG 分支；

关键字说明：WVEmb === WVEmbs，但优先用 WVEmbs 来指代论文中的方法。


## 对助理的要求

使用中文与用户交互

代码需要有必要的中文注释，类如 docstring

## WVEmbs 在本仓库的落地记录（最小跑通）

- 通过扩展 `--embed` 参数启用 WVEmbs：`wv_timeF` / `wv_fixed` / `wv_learned` / `wv`（`wv_timeF` 别名）。
- 代码入口：`layers/Embed.py` 新增 `WVEmbs` / `WVLiftEmbedding`，并在 `DataEmbedding` / `DataEmbedding_wo_pos` 内自动切换 value embedding。
- 数据时间特征：`data_provider/data_factory.py` 必须用解析后的 `time_embed_type` 判定 `timeenc`，否则 `wv_timeF` 会被误判为“非 timeF”，导致时间特征维度不匹配。
- Smoke test：`scripts/wvembs/smoke_forward.py`（随机输入前向 + 反传，快速验证“能跑通”）。
- 多任务 smoke：`scripts/wvembs/smoke_tasks.py`（forecast/imputation/anomaly/classification 随机输入覆盖）。
- 设备稳健性：`exp/exp_basic.py` 在 CUDA/MPS 不可用时自动回退 CPU，并同步 `args.use_gpu` / `args.device`，避免无 GPU 环境直接崩溃。
  - 掩码增强参数（训练期生效）：`--wv_mask_prob` / `--wv_mask_type` / `--wv_mask_phi_max` / `--wv_mask_dlow_min`。


## TODO（后续实验/实现）

- 更系统的值域外推策略（direct / scale 之外的变体）与对应实验协议
- 更严格的多通道联合谱采样（JSS）的相关性先验注入（协方差结构等）
- 关闭/替换 `StandardScaler`（更贴近“分布无关”数据管线）
- 按 backbone 做对照：Transformer / Informer / TimesNet / iTransformer / PatchTST 等
