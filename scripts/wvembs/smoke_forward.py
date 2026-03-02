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


def _make_common_configs(model_name: str) -> Namespace:
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
        embed="wv_timeF",
        freq="h",
        dropout=0.1,
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
    # `timeF` 在 freq='h' 下生成 4 维时间特征（month/day/weekday/hour 的线性组合特征）。
    x_enc = torch.randn(batch_size, cfg.seq_len, cfg.enc_in)
    x_mark_enc = torch.randn(batch_size, cfg.seq_len, 4)
    x_dec = torch.randn(batch_size, cfg.label_len + cfg.pred_len, cfg.dec_in)
    x_mark_dec = torch.randn(batch_size, cfg.label_len + cfg.pred_len, 4)
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
    args = parser.parse_args()

    model_names = [x.strip() for x in args.models.split(",") if x.strip()]
    if not model_names:
        raise SystemExit("No models specified.")

    for name in model_names:
        cfg = _make_common_configs(name)
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
