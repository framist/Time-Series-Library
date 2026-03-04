"""
估计 `scale_mode=prior` 的初始先验尺度（仅作为起点，不等价于“物理先验”）。

动机：
- prior 模式要求提供 `--prior_scale`（标量或每通道一个值）
- 在没有明确物理量量纲/量程时，可先用训练集上的 `max(abs(x))` 做一个“宽松上界”初始化，再人工调整

注意：
- 该脚本输出的是“数据驱动的粗估”，论文叙述中仍应把最终 prior 参数解释为物理先验/工程先验
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def _load_csv(root_path: str, data_path: str) -> pd.DataFrame:
    fp = os.path.join(root_path, data_path)
    if not os.path.exists(fp):
        raise FileNotFoundError(f"找不到数据文件：{fp}")
    return pd.read_csv(fp)


def _select_df_data(df_raw: pd.DataFrame, features: str, target: str) -> pd.DataFrame:
    if features in ("M", "MS"):
        # 约定：第 0 列是 date，其余为数值列
        return df_raw[df_raw.columns[1:]]
    if features == "S":
        return df_raw[[target]]
    raise ValueError(f"未知 features={features!r}，可选 M/MS/S")


def _ett_train_slice_len(data: str) -> int:
    # 与 data_provider/data_loader.py 的 ETT 切分保持一致
    if data in ("ETTm1", "ETTm2"):
        return 12 * 30 * 24 * 4
    if data in ("ETTh1", "ETTh2"):
        return 12 * 30 * 24
    raise ValueError(f"该脚本当前仅支持 ETT*，但得到 data={data!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M", choices=["M", "MS", "S"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--slack", type=float, default=2.0, help="对 max(|x|) 乘以该系数作为宽松上界（默认 2.0）")
    args = parser.parse_args()

    df_raw = _load_csv(args.root_path, args.data_path)
    df_data = _select_df_data(df_raw, args.features, args.target)

    train_len = _ett_train_slice_len(args.data)
    train_vals = df_data.iloc[:train_len].to_numpy(dtype=np.float32)

    max_abs = np.max(np.abs(train_vals), axis=0)
    prior_scale = (max_abs * float(args.slack)).tolist()

    cols = list(df_data.columns)
    print(f"[INFO] data={args.data} features={args.features} target={args.target} train_len={train_len}")
    print("[INFO] columns:", cols)
    print("[STAT] max_abs:", " ".join(f"{x:.6g}" for x in max_abs.tolist()))
    print("[SUGGEST] prior_scale(slack={}):".format(args.slack), " ".join(f"{x:.6g}" for x in prior_scale))
    print("[CLI] --prior_scale", " ".join(f"{x:.6g}" for x in prior_scale))


if __name__ == "__main__":
    main()
