"""
WVEmbs 最小 smoke test：

- 不依赖真实数据集
- 随机生成 (x_enc, x_mark_enc, x_dec, x_mark_dec)
- 对若干 backbone 做一次前向 + 反传，验证“接入 WVEmbs 后不崩、梯度可回传”
"""

import argparse
from argparse import Namespace

import os
import sys

import torch


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils.timefeatures import time_features_dim


def _make_common_configs(model_name: str, embed: str) -> Namespace:
    # 这套最小配置可覆盖多个主流预测 backbone（Transformer / Informer / TimesNet）。
    return Namespace(
        task_name="long_term_forecast",
        model=model_name,
        # 数据形状
        seq_len=32,
        label_len=16,
        pred_len=16,
        enc_in=3,
        dec_in=3,
        c_out=3,
        # embedding（启用 WVEmbs）
        embed=embed,
        freq="h",
        dropout=0.1,
        # WVEmbs masking（默认关闭；由 CLI 覆盖）
        wv_base=10000.0,
        wv_mask_prob=0.0,
        wv_mask_type="none",
        wv_mask_phi_max=3.141592653589793 / 8,
        wv_mask_dlow_min=0,
        # WVEmbs extrapolation / sampling
        wv_extrap_mode="direct",
        wv_extrap_scale=1.0,
        wv_sampling="iss",
        wv_jss_std=1.0,
        # backbone（Transformer/Informer/TimesNet）
        d_model=64,
        n_heads=4,
        e_layers=2,
        d_layers=1,
        d_ff=128,
        factor=3,
        activation="gelu",
        distil=True,
        # TimesNet
        top_k=5,
        num_kernels=6,
    )


def _make_fake_batch(cfg: Namespace, batch_size: int = 2):
    time_dim = int(time_features_dim(cfg.freq))
    x_enc = torch.randn(batch_size, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.randn(batch_size, cfg.seq_len, time_dim)
    x_dec = torch.randn(batch_size, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.randn(batch_size, cfg.label_len + cfg.pred_len, time_dim)
    return x_enc, x_mark_enc, x_dec, x_mark_dec


def _run_one(model_cls, cfg: Namespace):
    model = model_cls(cfg)
    model.train()
    x_enc, x_mark_enc, x_dec, x_mark_dec = _make_fake_batch(cfg)
    out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    loss = out.mean()
    loss.backward()
    return out.shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default="Transformer,Informer,TimesNet",
        help="Comma-separated model names to smoke-test: Transformer, Informer, TimesNet",
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="wv",
        help="embedding 模式：wv（统一时间入通道）或 wv_timeF（消融：时间仍用 TimeFeatureEmbedding）",
    )
    parser.add_argument("--wv_mask_prob", type=float, default=0.0, help="WVEmbs 掩码概率，0 关闭")
    parser.add_argument(
        "--wv_mask_type",
        type=str,
        default="none",
        choices=["none", "zero", "arcsine", "phase_rotate"],
        help="WVEmbs 掩码类型",
    )
    parser.add_argument("--wv_mask_phi_max", type=float, default=3.141592653589793 / 8, help="phase_rotate 扰动上限（弧度）")
    parser.add_argument("--wv_mask_dlow_min", type=int, default=0, help="dlow 下界（dlow_limited 变体）")
    parser.add_argument("--wv_extrap_mode", type=str, default="direct", choices=["direct", "scale"])
    parser.add_argument("--wv_extrap_scale", type=float, default=1.0)
    parser.add_argument("--wv_sampling", type=str, default="iss", choices=["iss", "jss"])
    parser.add_argument("--wv_jss_std", type=float, default=1.0)
    args = parser.parse_args()

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_names:
        raise SystemExit("No models specified.")

    for name in model_names:
        cfg = _make_common_configs(name, args.embed)
        cfg.wv_mask_prob = args.wv_mask_prob
        cfg.wv_mask_type = args.wv_mask_type
        cfg.wv_mask_phi_max = args.wv_mask_phi_max
        cfg.wv_mask_dlow_min = args.wv_mask_dlow_min
        cfg.wv_extrap_mode = args.wv_extrap_mode
        cfg.wv_extrap_scale = args.wv_extrap_scale
        cfg.wv_sampling = args.wv_sampling
        cfg.wv_jss_std = args.wv_jss_std
        if name == "Transformer":
            from models.Transformer import Model as ModelCls
        elif name == "Informer":
            from models.Informer import Model as ModelCls
        elif name == "TimesNet":
            from models.TimesNet import Model as ModelCls
        else:
            raise SystemExit(f"Unsupported model: {name}")

        shape = _run_one(ModelCls, cfg)
        print(f"[OK] {name}: out={tuple(shape)} embed={cfg.embed}")


if __name__ == "__main__":
    main()
