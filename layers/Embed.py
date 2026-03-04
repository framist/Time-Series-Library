import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from utils.embed_utils import parse_embed_arg, is_wv_unified
from utils.timefeatures import time_features_dim


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = int(time_features_dim(freq))
        if d_inp <= 0:
            raise ValueError(f"timeF 在 freq={freq!r} 下的时间特征维度为 {d_inp}，无法构造 TimeFeatureEmbedding")
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class WVEmbs(nn.Module):
    """
    WVEmbs（Wide Value Embedding）：
    对特征函数 e^{-i ω x} 进行离散谱点采样，并用实值形式输出。

    - 频率集合：采用 RoPE 风格的确定性对数频率（log-spaced），并按“高频在前、低频在后”排序。
    - 输出：将复数特征按 [cos(ωx), sin(ωx)] 拼接为实值向量，维度必须为偶数。
    """

    def __init__(self, dim, base=10000.0):
        super(WVEmbs, self).__init__()
        if dim < 2:
            raise ValueError(f"WVEmbs dim must be >= 2, got {dim}")
        if dim % 2 != 0:
            raise ValueError(f"WVEmbs dim must be even (cos/sin pairs), got {dim}")

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: (...,)
        angles = x.unsqueeze(-1) * self.inv_freq  # (..., dim/2)
        emb = torch.stack((angles.cos(), angles.sin()), dim=-1)  # (..., dim/2, 2)
        return emb.flatten(-2)  # (..., dim)


class WVLiftEmbedding(nn.Module):
    """
    WV-Lift Adapter（前端适配器，最小版本）：
    1) 逐变量 WVEmbs lifting：x -> Z，形状从 [B,T,M] 到 [B,T,M,D]
    2) 多通道交互：沿变量轴 M 做显式混合（此处用轻量 MLP；等价于 1x1 Conv 的一种实现）
    3) 形状对齐投影：将 [M*D] 投影回主干网络期望的 `d_model`

    同时提供 `wv_sampling=jss` 的最小联合谱采样：
    - 使用随机频率向量 ω_k ∈ R^M，对多通道向量做内积 angle_k=<ω_k, x>
    - 输出维度固定为 D（不随通道数 M 线性增长），用于验证论文中的 JSS 思路
    """

    def __init__(self, c_in, d_model, dropout=0.1, base=10000.0, wv_cfg=None):
        super(WVLiftEmbedding, self).__init__()
        wv_dim = d_model if (d_model % 2 == 0) else (d_model + 1)
        self.wv_dim = wv_dim
        self.c_in = c_in
        self.d_model = d_model

        self.sampling = str(getattr(wv_cfg, "wv_sampling", "iss")) if wv_cfg is not None else "iss"
        self.jss_std = float(getattr(wv_cfg, "wv_jss_std", 1.0)) if wv_cfg is not None else 1.0
        if self.sampling not in ("iss", "jss"):
            raise ValueError(f"未知 wv_sampling={self.sampling!r}，可选 iss/jss")
        if self.jss_std <= 0:
            raise ValueError(f"wv_jss_std 必须为正数，但得到 {self.jss_std}")

        self.extrap_mode = str(getattr(wv_cfg, "wv_extrap_mode", "direct")) if wv_cfg is not None else "direct"
        self.extrap_scale = float(getattr(wv_cfg, "wv_extrap_scale", 1.0)) if wv_cfg is not None else 1.0
        if self.extrap_mode not in ("direct", "scale"):
            raise ValueError(f"未知 wv_extrap_mode={self.extrap_mode!r}，可选 direct/scale")
        if self.extrap_scale <= 0:
            raise ValueError(f"wv_extrap_scale 必须为正数，但得到 {self.extrap_scale}")

        wv_base = float(getattr(wv_cfg, "wv_base", base)) if wv_cfg is not None else base

        self.mask_prob = float(getattr(wv_cfg, "wv_mask_prob", 0.0)) if wv_cfg is not None else 0.0
        self.mask_type = str(getattr(wv_cfg, "wv_mask_type", "none")) if wv_cfg is not None else "none"
        self.mask_phi_max = float(getattr(wv_cfg, "wv_mask_phi_max", math.pi / 8)) if wv_cfg is not None else (math.pi / 8)
        self.mask_dlow_min = int(getattr(wv_cfg, "wv_mask_dlow_min", 0)) if wv_cfg is not None else 0

        if self.sampling == "iss":
            self.wv = WVEmbs(dim=wv_dim, base=wv_base)
            self.jss_W = None
            self.var_mixer = nn.Identity() if c_in <= 1 else nn.Sequential(
                nn.Linear(c_in, c_in),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.proj_iss = nn.Linear(c_in * wv_dim, d_model)
            self.proj_jss = None
        else:
            self.wv = None
            k = wv_dim // 2
            w = torch.randn(k, c_in) * self.jss_std
            norms = torch.norm(w, dim=1)
            idx = torch.argsort(norms, descending=True)
            self.register_buffer("jss_W", w[idx])
            self.var_mixer = None
            self.proj_iss = None
            self.proj_jss = nn.Identity() if wv_dim == d_model else nn.Linear(wv_dim, d_model)

    def _apply_freq_mask_pairs(self, z_pairs):
        """
        频域掩码增强（仅训练阶段生效）。

        参数来自 `args`：
        - `wv_mask_prob`
        - `wv_mask_type`: none/zero/arcsine/phase_rotate
        - `wv_mask_phi_max`
        - `wv_mask_dlow_min`（dlow_limited 变体）
        """
        if (not self.training) or (self.mask_prob <= 0) or (self.mask_type in ("none", "None", "")):
            return z_pairs
        if z_pairs.shape[-1] != 2:
            raise ValueError(f"z_pairs 最后一维必须为 2（cos/sin），但得到 {z_pairs.shape[-1]}")

        bsz = z_pairs.shape[0]
        k = z_pairs.shape[-2]
        device = z_pairs.device

        apply = torch.rand((bsz,), device=device) < self.mask_prob  # [B]
        if not apply.any():
            return z_pairs

        dlow_min = max(0, min(self.mask_dlow_min, k))
        d_low = torch.randint(dlow_min, k + 1, (bsz,), device=device)  # [B], inclusive k

        freq_mask = torch.arange(k, device=device).unsqueeze(0) >= d_low.unsqueeze(1)  # [B,K]
        freq_mask = freq_mask & apply.unsqueeze(1)
        if not freq_mask.any():
            return z_pairs

        freq_mask = freq_mask.view(bsz, *([1] * (z_pairs.dim() - 3)), k, 1)

        if self.mask_type == "zero":
            z_pairs = torch.where(freq_mask, torch.zeros(1, device=device, dtype=z_pairs.dtype), z_pairs)
        elif self.mask_type == "arcsine":
            phi = torch.rand(z_pairs.shape[:-1], device=device) * (2 * math.pi)
            repl = torch.stack((phi.cos(), phi.sin()), dim=-1).to(dtype=z_pairs.dtype)
            z_pairs = torch.where(freq_mask, repl, z_pairs)
        elif self.mask_type == "phase_rotate":
            delta = (torch.rand(z_pairs.shape[:-1], device=device) * 2 - 1) * self.mask_phi_max
            cos_d = delta.cos().to(dtype=z_pairs.dtype)
            sin_d = delta.sin().to(dtype=z_pairs.dtype)

            cos, sin = z_pairs[..., 0], z_pairs[..., 1]
            rot_cos = cos * cos_d - sin * sin_d
            rot_sin = sin * cos_d + cos * sin_d
            repl = torch.stack((rot_cos, rot_sin), dim=-1)
            z_pairs = torch.where(freq_mask, repl, z_pairs)
        else:
            raise ValueError(f"未知 wv_mask_type={self.mask_type!r}，可选 none/zero/arcsine/phase_rotate")

        return z_pairs

    def forward(self, x):
        # x: [B, T, M]
        if self.extrap_mode == "scale" and self.extrap_scale != 1.0:
            x = x / self.extrap_scale

        if self.sampling == "iss":
            z = self.wv(x)  # [B, T, M, wv_dim]
            bsz, t, m, dim = z.shape
            k = dim // 2
            z_pairs = z.view(bsz, t, m, k, 2)
            z_pairs = self._apply_freq_mask_pairs(z_pairs)
            z = z_pairs.view(bsz, t, m, dim)

            z = z.permute(0, 1, 3, 2)  # [B, T, wv_dim, M]
            z = self.var_mixer(z)
            z = z.permute(0, 1, 3, 2)  # [B, T, M, wv_dim]
            z = z.reshape(z.shape[0], z.shape[1], -1)  # [B, T, M*wv_dim]
            return self.proj_iss(z)  # [B, T, d_model]

        # wv_sampling = jss
        angles = torch.matmul(x, self.jss_W.t()) / math.sqrt(max(1, self.c_in))  # [B, T, K]
        z_pairs = torch.stack((angles.cos(), angles.sin()), dim=-1)  # [B, T, K, 2]
        z_pairs = self._apply_freq_mask_pairs(z_pairs)
        z = z_pairs.flatten(-2)  # [B, T, 2K]
        return self.proj_jss(z)  # [B, T, d_model]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, wv_cfg=None):
        super(DataEmbedding, self).__init__()

        self.wv_unified = is_wv_unified(embed_type)

        time_embed_type, value_embed_type = parse_embed_arg(embed_type)

        if self.wv_unified:
            # 统一 WVEmbs：将时间特征视为“物理量通道”，与 x 沿通道维拼接后一起进入 WV-Lift
            self.time_feat_dim = int(time_features_dim(freq))
            self.value_embedding = WVLiftEmbedding(
                c_in=c_in + self.time_feat_dim,
                d_model=d_model,
                dropout=dropout,
                wv_cfg=wv_cfg,
            )
            self.temporal_embedding = None
        else:
            self.time_feat_dim = None
            self.value_embedding = WVLiftEmbedding(c_in=c_in, d_model=d_model, dropout=dropout, wv_cfg=wv_cfg) \
                if value_embed_type == 'wv' else TokenEmbedding(c_in=c_in, d_model=d_model)
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=time_embed_type,
                                                        freq=freq) if time_embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=time_embed_type, freq=freq)

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.wv_unified:
            if x_mark is None:
                x_mark = torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    int(self.time_feat_dim or 0),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                if x_mark.shape[1] != x.shape[1]:
                    raise ValueError(f"wv 统一模式要求 x 与 x_mark 的序列长度一致，但得到 {x.shape[1]} vs {x_mark.shape[1]}")
                if x_mark.shape[-1] != int(self.time_feat_dim or 0):
                    raise ValueError(
                        f"wv 统一模式要求 x_mark 最后一维为 time_feat_dim={self.time_feat_dim}，但得到 {x_mark.shape[-1]}"
                    )
                if x_mark.dtype != x.dtype:
                    x_mark = x_mark.to(dtype=x.dtype)

            x_cat = torch.cat([x, x_mark], dim=-1) if int(self.time_feat_dim or 0) > 0 else x
            x = self.value_embedding(x_cat) + self.position_embedding(x)
            return self.dropout(x)

        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, wv_cfg=None):
        super(DataEmbedding_wo_pos, self).__init__()

        self.wv_unified = is_wv_unified(embed_type)

        time_embed_type, value_embed_type = parse_embed_arg(embed_type)

        if self.wv_unified:
            self.time_feat_dim = int(time_features_dim(freq))
            self.value_embedding = WVLiftEmbedding(
                c_in=c_in + self.time_feat_dim,
                d_model=d_model,
                dropout=dropout,
                wv_cfg=wv_cfg,
            )
            self.temporal_embedding = None
        else:
            self.time_feat_dim = None
            self.value_embedding = WVLiftEmbedding(c_in=c_in, d_model=d_model, dropout=dropout, wv_cfg=wv_cfg) \
                if value_embed_type == 'wv' else TokenEmbedding(c_in=c_in, d_model=d_model)
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=time_embed_type,
                                                        freq=freq) if time_embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=time_embed_type, freq=freq)

        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if self.wv_unified:
            if x_mark is None:
                x_mark = torch.zeros(
                    x.shape[0],
                    x.shape[1],
                    int(self.time_feat_dim or 0),
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                if x_mark.shape[1] != x.shape[1]:
                    raise ValueError(f"wv 统一模式要求 x 与 x_mark 的序列长度一致，但得到 {x.shape[1]} vs {x_mark.shape[1]}")
                if x_mark.shape[-1] != int(self.time_feat_dim or 0):
                    raise ValueError(
                        f"wv 统一模式要求 x_mark 最后一维为 time_feat_dim={self.time_feat_dim}，但得到 {x_mark.shape[-1]}"
                    )
                if x_mark.dtype != x.dtype:
                    x_mark = x_mark.to(dtype=x.dtype)

            x_cat = torch.cat([x, x_mark], dim=-1) if int(self.time_feat_dim or 0) > 0 else x
            x = self.value_embedding(x_cat)
            return self.dropout(x)

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
