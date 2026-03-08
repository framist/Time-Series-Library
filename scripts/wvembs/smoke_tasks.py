"""
WVEmbs 多任务 smoke test：

目的：验证 WVEmbs + WV-Lift 接入后，至少在以下任务上“能前向 + 能反传 + 不崩”：
- long_term_forecast
- imputation
- anomaly_detection
- classification

说明：该脚本不依赖真实数据集，仅构造随机输入以覆盖代码路径。
"""

import argparse
import os
import sys
from argparse import Namespace

import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.timefeatures import time_features_dim


def _base_cfg(task_name: str, embed: str) -> Namespace:
    return Namespace(
        task_name=task_name,
        # embedding（启用 WVEmbs）
        embed=embed,
        freq="h",
        dropout=0.1,
        wv_base=10000.0,
        wv_mask_prob=0.0,
        wv_mask_type="none",
        wv_mask_phi_max=3.141592653589793 / 8,
        wv_mask_dlow_min=0,
        wv_extrap_mode="direct",
        wv_extrap_scale=1.0,
        wv_sampling="iss",
        wv_jss_std=1.0,
        # shapes
        seq_len=32,
        label_len=16,
        pred_len=16,
        enc_in=3,
        dec_in=3,
        c_out=3,
        # backbone（Transformer）
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        factor=3,
        activation="gelu",
        distil=True,
        # classification
        num_class=5,
    )


def _timeF_dim(freq: str) -> int:
    return int(time_features_dim(freq))


def _fake_forecast_batch(cfg: Namespace, batch_size: int):
    time_dim = _timeF_dim(cfg.freq)
    x_enc = torch.randn(batch_size, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.randn(batch_size, cfg.seq_len, time_dim)
    x_dec = torch.randn(batch_size, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.randn(batch_size, cfg.label_len + cfg.pred_len, time_dim)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def _run_backward(out: torch.Tensor):
    loss = out.mean()
    loss.backward()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--embed",
        type=str,
        default="wv",
        help="embedding 模式：timeF / linear / linear_timeF / wv / wv_timeF",
    )
    parser.add_argument("--wv_mask_prob", type=float, default=0.0)
    parser.add_argument(
        "--wv_mask_type",
        type=str,
        default="none",
        choices=["none", "zero", "arcsine", "phase_rotate"],
    )
    parser.add_argument("--wv_mask_phi_max", type=float, default=3.141592653589793 / 8)
    parser.add_argument("--wv_mask_dlow_min", type=int, default=0)
    parser.add_argument("--wv_extrap_mode", type=str, default="direct", choices=["direct", "scale"])
    parser.add_argument("--wv_extrap_scale", type=float, default=1.0)
    parser.add_argument("--wv_sampling", type=str, default="iss", choices=["iss", "jss"])
    parser.add_argument("--wv_jss_std", type=float, default=1.0)
    args = parser.parse_args()

    from models.Transformer import Model as Transformer

    def apply_mask_cfg(cfg: Namespace):
        cfg.wv_mask_prob = args.wv_mask_prob
        cfg.wv_mask_type = args.wv_mask_type
        cfg.wv_mask_phi_max = args.wv_mask_phi_max
        cfg.wv_mask_dlow_min = args.wv_mask_dlow_min
        cfg.wv_extrap_mode = args.wv_extrap_mode
        cfg.wv_extrap_scale = args.wv_extrap_scale
        cfg.wv_sampling = args.wv_sampling
        cfg.wv_jss_std = args.wv_jss_std
        return cfg

    # 1) forecasting
    cfg = apply_mask_cfg(_base_cfg("long_term_forecast", args.embed))
    model = Transformer(cfg).train()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _fake_forecast_batch(cfg, args.batch_size)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    _run_backward(out)
    print(f"[OK] long_term_forecast out={tuple(out.shape)} embed={cfg.embed} mask={cfg.wv_mask_type}")

    # 2) imputation
    cfg = apply_mask_cfg(_base_cfg("imputation", args.embed))
    model = Transformer(cfg).train()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _fake_forecast_batch(cfg, args.batch_size)
    mask = torch.ones(args.batch_size, cfg.seq_len, cfg.enc_in)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
    _run_backward(out)
    print(f"[OK] imputation out={tuple(out.shape)} embed={cfg.embed} mask={cfg.wv_mask_type}")

    # 3) anomaly detection
    cfg = apply_mask_cfg(_base_cfg("anomaly_detection", args.embed))
    model = Transformer(cfg).train()
    x_enc = torch.randn(args.batch_size, cfg.seq_len, cfg.enc_in)
    out = model(x_enc, None, None, None)
    _run_backward(out)
    print(f"[OK] anomaly_detection out={tuple(out.shape)} embed={cfg.embed} mask={cfg.wv_mask_type}")

    # 4) classification
    cfg = apply_mask_cfg(_base_cfg("classification", args.embed))
    model = Transformer(cfg).train()
    x_enc = torch.randn(args.batch_size, cfg.seq_len, cfg.enc_in)
    padding_mask = torch.ones(args.batch_size, cfg.seq_len)
    out = model(x_enc, padding_mask, None, None)
    _run_backward(out)
    print(f"[OK] classification out={tuple(out.shape)} embed={cfg.embed} mask={cfg.wv_mask_type}")


if __name__ == "__main__":
    main()
