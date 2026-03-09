#!/usr/bin/env python3
"""
导出 Forecast 任务的逐预测步误差曲线与摘要表。

读取 `results/long_term_forecast_*<DES>*/pred.npy` 与 `true.npy`，
按数据集、预测长度、输入层变体聚合，输出：

- 每个数据集一张多子图曲线图（横轴为预测步，纵轴为逐步误差）
- 一个 Markdown 摘要表，记录首步、末步、前段均值、后段均值与增长比例

用法：
    python scripts/wvembs/export_horizon_error_curves.py \
        --des NoPrepFairFull_20260309 \
        --metric mse \
        --outdir results/paper_visualizations/horizon_error_curves
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
VARIANT_ORDER = {"raw_timeF": 0, "timeF": 0, "linear": 1, "wv": 2}
VARIANT_LABEL = {
    "raw_timeF": "原始时间特征输入层",
    "timeF": "原始时间特征输入层",
    "linear": "线性统一输入层",
    "wv": "WVEmbs 统一输入层",
}
VARIANT_COLOR = {
    "raw_timeF": "#1f77b4",
    "timeF": "#1f77b4",
    "linear": "#ff7f0e",
    "wv": "#d62728",
}


def setup_fonts() -> None:
    """统一中文字体配置，避免无头环境下中文乱码。"""
    font_config = {
        "font.sans-serif": ["WenQuanYi Zen Hei", "SimHei", "DejaVu Sans", "Arial Unicode MS"],
        "axes.unicode_minus": False,
    }
    plt.rcParams.update(font_config)


def parse_setting(name: str) -> Dict[str, str]:
    match = re.match(
        r"long_term_forecast_NoPrepFair_(?P<dataset>[^_]+)_Transformer_(?P<variant>raw_timeF|linear|wv)_pl(?P<pred_len>\d+)_",
        name,
    )
    if not match:
        raise ValueError(f"无法解析 setting: {name}")
    return match.groupdict()


def collect_runs(des: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    pattern = f"results/long_term_forecast_*{des}*/pred.npy"
    for pred_path in sorted(ROOT.glob(pattern)):
        setting = pred_path.parent.name
        meta = parse_setting(setting)
        true_path = pred_path.parent / "true.npy"
        if not true_path.exists():
            continue
        rows.append(
            {
                "setting": setting,
                "dataset": meta["dataset"],
                "pred_len": int(meta["pred_len"]),
                "variant": meta["variant"],
                "variant_label": VARIANT_LABEL[meta["variant"]],
                "pred_path": pred_path,
                "true_path": true_path,
            }
        )
    return rows


def merge_runs(
    primary_rows: List[Dict[str, object]],
    reference_rows: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[tuple, Dict[str, object]] = {}
    primary_groups = {
        (str(row["dataset"]), int(row["pred_len"]))
        for row in primary_rows
    }
    for row in reference_rows:
        group_key = (str(row["dataset"]), int(row["pred_len"]))
        if group_key not in primary_groups:
            continue
        key = (str(row["dataset"]), int(row["pred_len"]), str(row["variant"]))
        merged[key] = dict(row)
    for row in primary_rows:
        key = (str(row["dataset"]), int(row["pred_len"]), str(row["variant"]))
        merged[key] = dict(row)
    return list(merged.values())


def compute_step_curve(pred_path: Path, true_path: Path, metric: str) -> np.ndarray:
    preds = np.load(pred_path).astype(np.float64, copy=False)
    trues = np.load(true_path).astype(np.float64, copy=False)
    diff = preds - trues
    if metric == "mse":
        values = np.square(diff)
    elif metric == "mae":
        values = np.abs(diff)
    else:
        raise ValueError(f"不支持的 metric: {metric}")

    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=(0, 2))
    sums = np.where(finite_mask, values, 0.0).sum(axis=(0, 2))
    curve = np.full(values.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    curve[valid] = sums[valid] / counts[valid]
    return curve


def summarize_curve(curve: np.ndarray) -> Dict[str, str]:
    finite = np.isfinite(curve)
    if not finite.any():
        return {
            "first": "nan",
            "last": "nan",
            "head_mean": "nan",
            "tail_mean": "nan",
            "tail_vs_head": "NaN/Inf",
        }

    clean = curve[finite]
    quarter = max(1, int(math.ceil(len(clean) * 0.25)))
    head_mean = float(np.mean(clean[:quarter]))
    tail_mean = float(np.mean(clean[-quarter:]))
    growth = "NaN/Inf" if head_mean == 0 else f"{(tail_mean - head_mean) / head_mean * 100.0:+.1f}%"
    return {
        "first": f"{float(clean[0]):.6f}",
        "last": f"{float(clean[-1]):.6f}",
        "head_mean": f"{head_mean:.6f}",
        "tail_mean": f"{tail_mean:.6f}",
        "tail_vs_head": growth,
    }


def markdown_table(headers: List[str], rows: List[Dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def plot_dataset(dataset: str, rows: List[Dict[str, object]], metric: str, outdir: Path) -> Path:
    pred_lens = sorted({int(row["pred_len"]) for row in rows})
    if not pred_lens:
        raise ValueError(f"{dataset} 没有可绘制的数据")

    ncols = 2
    nrows = int(math.ceil(len(pred_lens) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    handles = {}
    for ax, pred_len in zip(axes_flat, pred_lens):
        subset = [row for row in rows if int(row["pred_len"]) == pred_len]
        subset.sort(key=lambda row: VARIANT_ORDER.get(str(row["variant"]), 99))

        for row in subset:
            curve = row["curve"]
            variant = str(row["variant"])
            x = np.arange(1, len(curve) + 1)
            finite = np.isfinite(curve)
            if finite.any():
                (line,) = ax.plot(
                    x[finite],
                    curve[finite],
                    linewidth=2.0,
                    color=VARIANT_COLOR.get(variant, "#333333"),
                    label=str(row["variant_label"]),
                )
                handles[str(row["variant_label"])] = line
                if not finite.all():
                    ax.scatter(
                        x[~finite],
                        np.full((~finite).sum(), np.nanmin(curve[finite])),
                        marker="x",
                        color=VARIANT_COLOR.get(variant, "#333333"),
                        alpha=0.5,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{row['variant_label']}\n全为 NaN/Inf",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=VARIANT_COLOR.get(variant, "#333333"),
                )

        ax.set_title(f"{dataset}，预测长度 {pred_len}", fontsize=12)
        ax.set_xlabel("预测步")
        ax.set_ylabel(f"逐步 {metric.upper()}")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.8)

    for ax in axes_flat[len(pred_lens) :]:
        ax.axis("off")

    if handles:
        fig.legend(
            handles.values(),
            handles.keys(),
            loc="upper center",
            ncol=min(3, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(f"{dataset} 无预处理公平对照：误差随预测步变化", fontsize=14, y=1.05)
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{dataset}_{metric}_horizon_curve.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_rows_with_curves(rows: List[Dict[str, object]], metric: str) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in rows:
        curve = compute_step_curve(row["pred_path"], row["true_path"], metric)
        summary = summarize_curve(curve)
        enriched.append({**row, "curve": curve, **summary})
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="导出逐预测步误差曲线")
    parser.add_argument("--des", required=True, help="实验 DES，例如 NoPrepFairFull_20260309")
    parser.add_argument(
        "--reference-des",
        default="",
        help="可选参考实验描述字段；当当前 DES 只补跑部分 variant 时，可从参考结果中补齐 raw_timeF/linear 等曲线",
    )
    parser.add_argument("--metric", choices=["mse", "mae"], default="mse", help="曲线指标")
    parser.add_argument(
        "--outdir",
        default="results/paper_visualizations/horizon_error_curves",
        help="输出目录",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="可选，仅导出指定数据集，例如 ETTh1 ETTh2 Weather",
    )
    args = parser.parse_args()

    setup_fonts()

    runs = collect_runs(args.des)
    if args.reference_des:
        runs = merge_runs(runs, collect_runs(args.reference_des))
    rows = build_rows_with_curves(runs, args.metric)
    if args.datasets:
        wanted = set(args.datasets)
        rows = [row for row in rows if row["dataset"] in wanted]

    if not rows:
        print("未找到匹配的 Forecast 结果。")
        return

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)

    summary_rows: List[Dict[str, str]] = []
    for dataset in sorted(grouped):
        saved_path = plot_dataset(dataset, grouped[dataset], args.metric, outdir)
        print(f"[OK] 图已保存：{saved_path.relative_to(ROOT)}")
        for row in sorted(
            grouped[dataset],
            key=lambda item: (str(item["dataset"]), int(item["pred_len"]), VARIANT_ORDER.get(str(item["variant"]), 99)),
        ):
            summary_rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "pred_len": str(row["pred_len"]),
                    "variant_label": str(row["variant_label"]),
                    "first": str(row["first"]),
                    "last": str(row["last"]),
                    "head_mean": str(row["head_mean"]),
                    "tail_mean": str(row["tail_mean"]),
                    "tail_vs_head": str(row["tail_vs_head"]),
                }
            )

    summary_md = markdown_table(
        ["dataset", "pred_len", "variant_label", "first", "last", "head_mean", "tail_mean", "tail_vs_head"],
        summary_rows,
    )
    summary_path = outdir / f"{args.des}_{args.metric}_summary.md"
    summary_path.write_text(summary_md + "\n", encoding="utf-8")
    print(f"[OK] 摘要已保存：{summary_path.relative_to(ROOT)}")
    print(summary_md)


if __name__ == "__main__":
    main()
