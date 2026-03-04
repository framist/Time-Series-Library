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
from huggingface_hub import hf_hub_download


HUGGINGFACE_REPO = "thuml/Time-Series-Library"


def _load_csv(root_path: str, data_path: str) -> pd.DataFrame:
    fp = os.path.join(root_path, data_path)
    if not os.path.exists(fp):
        # 与 data_provider/data_loader.py 一致：优先尝试从 HF 数据集仓库自动下载
        subdir = os.path.basename(os.path.normpath(root_path))
        candidates = [f"{subdir}/{data_path}", data_path]
        last_err = None
        for rel in candidates:
            try:
                fp = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=rel, repo_type="dataset")
                break
            except Exception as e:
                last_err = e
                continue
        else:
            raise FileNotFoundError(f"找不到数据文件：{os.path.join(root_path, data_path)}") from last_err
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
    raise ValueError(f"未知 ETT 数据集：{data!r}")


def _custom_train_slice_len(df_len: int) -> int:
    # 与 Dataset_Custom 保持一致：train/val/test = 0.7/0.1/0.2
    return int(df_len * 0.7)


def _reorder_custom_columns(df_raw: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    与 Dataset_Custom 对齐列顺序：['date'] + other_features + [target]

    重要性：
    - prior_scale 如果按“每通道”给定，则通道顺序必须与训练时一致；
    - Dataset_Custom 会把 target 移到最后一列；这里也同步这一行为。
    """
    cols = list(df_raw.columns)
    if "date" not in cols:
        raise ValueError("custom 数据集要求存在 'date' 列")
    if target not in cols:
        raise ValueError(f"custom 数据集找不到 target={target!r} 列；可用 --target 指定正确列名")

    cols.remove(target)
    cols.remove("date")
    return df_raw[["date"] + cols + [target]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom"])
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--features", type=str, default="M", choices=["M", "MS", "S"])
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--slack", type=float, default=2.0, help="对 max(|x|) 乘以该系数作为宽松上界（默认 2.0）")
    parser.add_argument(
        "--reduce",
        type=str,
        default="per_channel",
        choices=["per_channel", "global_max"],
        help="输出 prior_scale 的形式：per_channel（每通道）或 global_max（全局最大值标量）",
    )
    parser.add_argument(
        "--print_full",
        action="store_true",
        help="当通道数很大时仍打印完整向量（默认仅打印摘要，并始终给出标量建议）",
    )
    args = parser.parse_args()

    df_raw = _load_csv(args.root_path, args.data_path)
    if args.data == "custom":
        df_raw = _reorder_custom_columns(df_raw, target=args.target)
    df_data = _select_df_data(df_raw, args.features, args.target)

    if args.data == "custom":
        train_len = _custom_train_slice_len(len(df_raw))
    else:
        train_len = _ett_train_slice_len(args.data)
    train_vals = df_data.iloc[:train_len].to_numpy(dtype=np.float32)

    max_abs = np.max(np.abs(train_vals), axis=0)
    prior_scale_per_channel = (max_abs * float(args.slack)).tolist()
    prior_scale_scalar = float(np.max(max_abs) * float(args.slack))

    cols = list(df_data.columns)
    print(f"[INFO] data={args.data} features={args.features} target={args.target} train_len={train_len}")
    print("[INFO] feature_dim:", len(cols))
    print("[STAT] max_abs_global:", f"{float(np.max(max_abs)):.6g}")
    print("[SUGGEST] prior_scale_global_max(slack={}): {:.6g}".format(args.slack, prior_scale_scalar))
    print("[CLI] --prior_scale", f"{prior_scale_scalar:.6g}")

    if args.reduce == "global_max":
        return

    if len(cols) <= 64 or args.print_full:
        print("[INFO] columns:", cols)
        print("[STAT] max_abs:", " ".join(f"{x:.6g}" for x in max_abs.tolist()))
        print("[SUGGEST] prior_scale_per_channel(slack={}):".format(args.slack), " ".join(f"{x:.6g}" for x in prior_scale_per_channel))
        print("[CLI] --prior_scale", " ".join(f"{x:.6g}" for x in prior_scale_per_channel))
        return

    # 通道很多时，避免刷屏：只打印摘要（仍可通过 --print_full 打印完整向量）
    head_n = 8
    tail_n = 8
    max_abs_list = max_abs.tolist()
    print("[INFO] columns(head):", cols[:head_n], "...", cols[-tail_n:])
    print(
        "[STAT] max_abs(head..tail):",
        " ".join(f"{x:.6g}" for x in max_abs_list[:head_n]),
        "...",
        " ".join(f"{x:.6g}" for x in max_abs_list[-tail_n:]),
    )
    print("[HINT] 通道数较多：如需打印完整 per_channel 向量，请加 --print_full")


if __name__ == "__main__":
    main()
