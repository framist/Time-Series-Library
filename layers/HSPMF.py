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
Reduction = Literal["mean", "sum", "none"]


def make_symmetric_omegas(n_fourier: int, period: float, device=None) -> torch.Tensor:
    """
    构造对称角频率集合：ω_n = 2π n / period，其中 n=-N..N（包含 0）。
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
    """
    if omegas.ndim != 1:
        raise ValueError("omegas 必须是 1D 张量 (K,)")
    u = x - float(x_lo)
    return torch.exp((-1j) * u[..., None] * omegas)


def build_conjugate_symmetric_y(
    y_pos: torch.Tensor, y0: complex = 1.0 + 0.0j
) -> torch.Tensor:
    """
    由正频率部分 y_pos 构造完整的共轭对称频谱：
        [y_-N, ..., y_-1, y_0, y_1, ..., y_N]
    """
    if y_pos.ndim < 1:
        raise ValueError("y_pos 至少需要 1 个维度")
    if not torch.is_complex(y_pos):
        raise ValueError("y_pos 必须是复数张量")

    y0_t = torch.as_tensor(y0, dtype=y_pos.dtype, device=y_pos.device)
    y0_t = y0_t.expand(*y_pos.shape[:-1], 1)
    y_neg = torch.conj(y_pos).flip(-1)
    return torch.cat([y_neg, y0_t, y_pos], dim=-1)


def targets_to_grid_index(x: torch.Tensor, x_grid: torch.Tensor) -> torch.Tensor:
    """
    把任意形状的目标值映射到解码网格索引。
    """
    if x_grid.ndim != 1:
        raise ValueError("x_grid 必须是一维张量")
    if x_grid.numel() < 2:
        raise ValueError("x_grid 至少包含两个点")

    dx = float((x_grid[1] - x_grid[0]).item())
    if abs(dx - 1.0) < 1e-6 and torch.allclose(
        x_grid, torch.round(x_grid), atol=1e-6, rtol=0.0
    ):
        lo = int(torch.round(x_grid[0]).item())
        idx = torch.round(x).to(torch.int64) - lo
    else:
        idx = torch.round((x - x_grid[0]) / dx).to(torch.int64)
    return idx.clamp(0, int(x_grid.numel()) - 1)


def nll_loss_from_pmf(
    p: torch.Tensor,
    idx: torch.Tensor,
    *,
    eps: float = 1e-12,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    离散网格上的负对数似然。
    """
    if p.ndim < 2:
        raise ValueError("p 至少需要 2 个维度 (..., M)")
    if idx.shape != p.shape[:-1]:
        raise ValueError("idx 的形状必须与 p 去掉最后一维后的形状一致")

    p_flat = p.reshape(-1, p.shape[-1])
    idx_flat = idx.reshape(-1)
    ar = torch.arange(p_flat.shape[0], device=p.device)
    values = -torch.log(p_flat[ar, idx_flat] + float(eps))

    if reduction == "none":
        return values.reshape(idx.shape)
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"未知 reduction: {reduction}")


def crps_from_pmf(
    p: torch.Tensor,
    target: torch.Tensor,
    x_grid: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    在均匀解码网格上计算离散 CRPS。
    """
    if p.ndim < 2:
        raise ValueError("p 至少需要 2 个维度 (..., M)")
    if target.shape != p.shape[:-1]:
        raise ValueError("target 的形状必须与 p 去掉最后一维后的形状一致")
    if x_grid.ndim != 1 or x_grid.numel() < 2:
        raise ValueError("x_grid 必须是一维且长度至少为 2")

    p_flat = p.reshape(-1, p.shape[-1])
    target_flat = target.reshape(-1)
    cdf = torch.cumsum(p_flat, dim=-1)
    truth_cdf = (x_grid.unsqueeze(0) >= target_flat.unsqueeze(-1)).to(cdf.dtype)
    dx = torch.abs(x_grid[1] - x_grid[0]).to(dtype=cdf.dtype, device=cdf.device)
    values = ((cdf - truth_cdf) ** 2).sum(dim=-1) * dx

    if reduction == "none":
        return values.reshape(target.shape)
    if reduction == "sum":
        return values.sum()
    if reduction == "mean":
        return values.mean()
    raise ValueError(f"未知 reduction: {reduction}")


class HSPMFDecoder(nn.Module):
    """
    HSPMF 解码器：将频域特征解码为值域的后验分布和点估计。
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
        learn_beta: bool = False,
    ):
        super().__init__()
        self.n_fourier = n_fourier
        self.period = period
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.grid_size = grid_size
        self.tau = tau
        self.score_mode = score_mode
        self.learn_beta = bool(learn_beta)

        beta0 = max(float(beta), 1e-6)
        if self.learn_beta:
            self.log_beta = nn.Parameter(torch.tensor(math.log(beta0), dtype=torch.float32))
        else:
            self.register_buffer("_fixed_beta", torch.tensor(beta0, dtype=torch.float32))

        omegas = make_symmetric_omegas(n_fourier, period)
        self.register_buffer("omegas", omegas)

        x_grid = torch.linspace(x_lo, x_hi, grid_size, dtype=torch.float32)
        self.register_buffer("x_grid", x_grid)

        n = torch.arange(-n_fourier, n_fourier + 1)
        if hier_levels is None:
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

        masks = []
        prev = -1
        for n_max in levels:
            mask = (n.abs() <= n_max) & (n.abs() > prev)
            masks.append(mask)
            prev = n_max

        self.n_levels = len(levels)
        for i, mask in enumerate(masks):
            self.register_buffer(f"mask_lvl{i}", mask.nonzero(as_tuple=True)[0])
        self.levels = levels

    @property
    def beta_value(self) -> torch.Tensor:
        if self.learn_beta:
            return torch.exp(self.log_beta)
        return self._fixed_beta

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解码复数频域特征 y 为值域后验分布和点估计。
        """
        if y.shape[-1] != self.omegas.numel():
            raise ValueError(
                f"HSPMFDecoder 输入频点数不匹配：期望 {self.omegas.numel()}，实际 {y.shape[-1]}"
            )

        orig_shape = y.shape[:-1]
        y = y.reshape(-1, y.shape[-1])
        bsz = y.shape[0]
        grid_size = self.grid_size
        device = y.device
        u = self.x_grid - self.x_lo

        target_dtype = y.dtype
        log_prior = torch.zeros(bsz, grid_size, dtype=torch.float32, device=device)
        beta = self.beta_value.to(device=device, dtype=torch.float32)
        posterior = None

        for lvl in range(self.n_levels):
            mask = getattr(self, f"mask_lvl{lvl}")
            omegas_sub = self.omegas[mask]
            y_sub = y[:, mask]
            ksub = float(omegas_sub.numel())

            phi_conj = torch.exp((1j) * omegas_sub[:, None] * u[None, :])
            if phi_conj.dtype != target_dtype:
                phi_conj = phi_conj.to(target_dtype)

            ip = y_sub @ phi_conj
            if self.score_mode == "abs2":
                score = (ip.abs() ** 2) / ksub
            else:
                score = ip.real / ksub

            logits = log_prior + (beta * score) / self.tau
            logits = logits - logits.max(dim=1, keepdim=True).values
            posterior = F.softmax(logits, dim=1)
            log_prior = torch.log(posterior + 1e-12)

        if posterior is None:
            raise RuntimeError("HSPMFDecoder 未产生有效后验；请检查 hier_levels 配置")

        x_hat = (posterior * self.x_grid.unsqueeze(0)).sum(dim=1)
        x_hat = x_hat.reshape(orig_shape)
        p_final = posterior.reshape(*orig_shape, grid_size)
        return x_hat, p_final

    def extra_repr(self) -> str:
        beta = float(self.beta_value.detach().cpu().item())
        return (
            f"n_fourier={self.n_fourier}, period={self.period:.3f}, "
            f"x_range=[{self.x_lo:.2f}, {self.x_hi:.2f}], grid_size={self.grid_size}, "
            f"levels={self.levels}, beta={beta:.4f}, learn_beta={self.learn_beta}, tau={self.tau}"
        )
