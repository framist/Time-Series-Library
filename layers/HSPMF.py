"""
HSPMF - Hierarchical Soft Posterior Matched Filtering
级联软后验匹配滤波解码器

将 WVEmb 的特征函数采样可微地解码为连续物理量。
"""

import math
from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


ScoreMode = Literal["real", "abs2"]


def make_symmetric_omegas(n_fourier: int, period: float, device=None) -> torch.Tensor:
    """
    构造对称角频率集合：ω_n = 2π n / period，其中 n=-N..N（包含 0）。

    Args:
        n_fourier: 最大傅里叶阶数 N
        period: 周期
        device: torch 设备

    Returns:
        omegas: (2N+1,) 频率张量，按 n=-N..N 排列
    """
    if n_fourier < 1:
        raise ValueError("n_fourier 必须 >= 1")
    if period <= 0:
        raise ValueError("period 必须 > 0")
    n = torch.arange(-n_fourier, n_fourier + 1, dtype=torch.float32, device=device)
    return (2.0 * math.pi / float(period)) * n


def wvemb_complex(x: torch.Tensor, omegas: torch.Tensor, x_lo: float) -> torch.Tensor:
    """
    WVEmb / 特征函数采样（复数形式）：
        φ(x)[k] = exp(-i ω_k (x - x_lo))

    Args:
        x: 任意形状 (...,)
        omegas: (K,) 频率张量
        x_lo: 平移偏置

    Returns:
        复数张量，形状 (..., K)
    """
    if omegas.ndim != 1:
        raise ValueError("omegas 必须是 1D 张量 (K,)")
    u = x - float(x_lo)
    return torch.exp((-1j) * u[..., None] * omegas)


class HSPMFDecoder(nn.Module):
    """
    HSPMF 解码器：将频域特征解码为值域的后验分布和点估计。

    实现分层软后验匹配滤波：
    - 低频层（粗）：确定大致区间，抑制混叠
    - 高频层（细）：在缩定窗口内精修
    """

    def __init__(
        self,
        n_fourier: int,
        period: float,
        x_lo: float,
        x_hi: float,
        grid_size: int = 128,
        hier_levels: Tuple[int, ...] = None,
        beta: float = 1.0,
        tau: float = 1.0,
        score_mode: ScoreMode = "real",
    ):
        """
        Args:
            n_fourier: 傅里叶阶数 N（对应 2N+1 个频率点）
            period: 频率周期参数
            x_lo, x_hi: 值域范围 [x_lo, x_hi]
            grid_size: 解码网格大小 M
            hier_levels: 分层级数，如 (4, 16, N)。None 则自动选择。
            beta: 锐化系数
            tau: 温度系数
            score_mode: "real" 或 "abs2"
        """
        super().__init__()
        self.n_fourier = n_fourier
        self.period = period
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.grid_size = grid_size
        self.beta = beta
        self.tau = tau
        self.score_mode = score_mode

        # 注册频率张量（非可学习）
        omegas = make_symmetric_omegas(n_fourier, period)
        self.register_buffer("omegas", omegas)  # (2N+1,)

        # 注册解码网格（非可学习）
        x_grid = torch.linspace(x_lo, x_hi, grid_size, dtype=torch.float32)
        self.register_buffer("x_grid", x_grid)  # (M,)

        # 构建层级掩码
        k = 2 * n_fourier + 1
        n = torch.arange(-n_fourier, n_fourier + 1)

        if hier_levels is None:
            # 默认分层：4、16、N（或根据 N 调整）
            if n_fourier <= 4:
                levels = [n_fourier]
            else:
                levels = [min(4, n_fourier), min(16, n_fourier), n_fourier]
                levels = [v for v in levels if 0 < v <= n_fourier]
                levels = sorted(dict.fromkeys(levels))
        else:
            levels = list(hier_levels)

        if not levels or levels[-1] != n_fourier:
            raise ValueError("hier_levels 必须非空且最后一项等于 n_fourier")

        # 为每一层构建频率掩码（嵌套子集：K_1 ⊂ K_2 ⊂ ... ⊂ K_L）
        # 对齐论文定义：每层包含所有 |n| <= n_max 的频率
        masks = []
        for n_max in levels:
            mask = n.abs() <= n_max  # 嵌套：包含所有低频到 n_max
            masks.append(mask)
        
        self.n_levels = len(levels)
        self.n_levels = len(levels)
        for i, mask in enumerate(masks):
            self.register_buffer(f"mask_lvl{i}", mask.nonzero(as_tuple=True)[0])
        self.levels = levels

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码复数频域特征 y 为值域后验分布和点估计。

        Args:
            y: (..., K) 复数张量，K = 2*n_fourier + 1

        Returns:
            x_hat: (...) 点估计（后验均值）
            p_final: (..., M) 最终层离散后验
        """
        orig_shape = y.shape[:-1]
        y = y.reshape(-1, y.shape[-1])  # (B, K)
        B = y.shape[0]
        M = self.grid_size
        device = y.device

        # 预计算模板 φ(x) = exp(-i ω x) 的共轭
        # u = x_grid - x_lo
        u = self.x_grid - self.x_lo  # (M,)

        # 分层解码
        # 确保 dtype 一致（处理 AMP 混合精度）
        target_dtype = y.dtype
        log_prior = torch.zeros(B, M, dtype=torch.float32, device=device)

        for lvl in range(self.n_levels):
            mask = getattr(self, f"mask_lvl{lvl}")
            omegas_sub = self.omegas[mask]  # (Ksub,)
            y_sub = y[:, mask]  # (B, Ksub)
            ksub = float(omegas_sub.numel())

            # 计算 φ(x) = exp(-i ω (x - x_lo)) 的共轭 = exp(+i ω u)
            # phi_conj: (Ksub, M)
            phi_conj = torch.exp((1j) * omegas_sub[:, None] * u[None, :])
            
            # 确保 dtype 匹配（AMP 兼容性）
            if phi_conj.dtype != target_dtype:
                phi_conj = phi_conj.to(target_dtype)

            # 匹配滤波内积：ip = y @ conj(phi) = y @ phi_conj
            # (B, Ksub) @ (Ksub, M) -> (B, M)
            ip = y_sub @ phi_conj

            # 计算分数
            if self.score_mode == "abs2":
                score = (ip.abs() ** 2) / ksub
            else:  # "real"
                score = ip.real / ksub

            # 后验：p ∝ exp(beta * score / tau)
            logits = log_prior + (self.beta * score) / self.tau
            logits = logits - logits.max(dim=1, keepdim=True).values
            p = F.softmax(logits, dim=1)
            log_prior = torch.log(p + 1e-12)
        # 确保 dtype 一致（处理 AMP 混合精度）
        target_dtype = y.dtype
        log_prior = torch.zeros(B, M, dtype=torch.float32, device=device)

        for lvl in range(self.n_levels):
            mask = getattr(self, f"mask_lvl{lvl}")
            omegas_sub = self.omegas[mask]  # (Ksub,)
            y_sub = y[:, mask]  # (B, Ksub)
            ksub = float(omegas_sub.numel())

            # 计算 φ(x) = exp(-i ω (x - x_lo)) 的共轭 = exp(+i ω u)
            # phi_conj: (Ksub, M)
            phi_conj = torch.exp((1j) * omegas_sub[:, None] * u[None, :])
            
            # 确保 dtype 匹配（AMP 兼容性）
            if phi_conj.dtype != target_dtype:
                phi_conj = phi_conj.to(target_dtype)
        log_prior = torch.zeros(B, M, dtype=torch.float32, device=device)

        for lvl in range(self.n_levels):
            mask = getattr(self, f"mask_lvl{lvl}")
            omegas_sub = self.omegas[mask]  # (Ksub,)
            y_sub = y[:, mask]  # (B, Ksub)
            ksub = float(omegas_sub.numel())

            # 计算 φ(x) = exp(-i ω (x - x_lo)) 的共轭 = exp(+i ω u)
            # phi_conj: (Ksub, M)
            phi_conj = torch.exp((1j) * omegas_sub[:, None] * u[None, :])

            # 匹配滤波内积：ip = y @ conj(phi) = y @ phi_conj
            # (B, Ksub) @ (Ksub, M) -> (B, M)
            ip = y_sub @ phi_conj

            # 计算分数
            if self.score_mode == "abs2":
                score = (ip.abs() ** 2) / ksub
            else:  # "real"
                score = ip.real / ksub

            # 后验：p ∝ exp(beta * score / tau)
            logits = log_prior + (self.beta * score) / self.tau
            logits = logits - logits.max(dim=1, keepdim=True).values
            p = F.softmax(logits, dim=1)
            log_prior = torch.log(p + 1e-12)

        # 最终后验的点估计（后验均值）
        x_hat = (p * self.x_grid.unsqueeze(0)).sum(dim=1)  # (B,)

        # 恢复原始形状
        x_hat = x_hat.reshape(orig_shape)
        p_final = p.reshape(*orig_shape, M)

        return x_hat, p_final

    def extra_repr(self) -> str:
        return (
            f"n_fourier={self.n_fourier}, period={self.period:.3f}, "
            f"x_range=[{self.x_lo:.2f}, {self.x_hi:.2f}], grid_size={self.grid_size}, "
            f"levels={self.levels}, beta={self.beta}, tau={self.tau}"
        )
