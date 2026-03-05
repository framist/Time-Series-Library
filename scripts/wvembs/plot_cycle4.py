#!/usr/bin/env python3
"""Cycle 4 可视化：热力图 + 折线图

读取 result_long_term_forecast.txt，按 DES 标签过滤，生成：
1. 三因素联合扫描热力图（jss_std × scale_mode → MSE）
2. wv_base 灵敏度折线图
3. ETTh2 跨数据集验证小热力图

用法：
    python scripts/wvembs/plot_cycle4.py [--results result_long_term_forecast.txt] [--outdir results/]
"""

import argparse
import re
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def parse_results(filepath: str) -> list[dict]:
    """解析 result_long_term_forecast.txt，返回 [{header, mse, mae}, ...]"""
    records = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("long_term_forecast_"):
            header = line
            # 下一行是 mse/mae
            if i + 1 < len(lines):
                metrics_line = lines[i + 1].strip()
                m = re.match(
                    r"mse:([\d.eE+\-nan]+),\s*mae:([\d.eE+\-nan]+)", metrics_line
                )
                if m:
                    try:
                        mse = float(m.group(1))
                    except ValueError:
                        mse = float("nan")
                    try:
                        mae = float(m.group(2))
                    except ValueError:
                        mae = float("nan")
                    records.append({"header": header, "mse": mse, "mae": mae})
                i += 2
                continue
        i += 1

    return records


def filter_by_des(records: list[dict], des: str) -> list[dict]:
    """按 DES 标签过滤"""
    return [r for r in records if des in r["header"]]


def extract_tag_from_header(header: str, prefix: str) -> str:
    """从 header 中提取 cycle4 tag（model_id 部分）

    header 格式: long_term_forecast_{model_id}_{model}_{dataset}_ftM_..._{des}_0
    model_id 格式: WVEmbs_ETTh1_Transformer_cycle4_{tag}
    tag 示例: sm_standard_iss, sm_none_jss_0.25, base_100

    策略：匹配 cycle4_ 到 _Transformer/_TimeMixer 之间的内容
    """
    # 匹配 cycle4_ 后面直到遇到已知模型名的部分
    m = re.search(r"cycle4_(" + prefix + r"[\w.]+(?:_[\w.]+)*?)_(Transformer|TimeMixer|TimesNet|Autoformer|Nonstationary_Transformer|Informer)", header)
    if m:
        return m.group(1)
    # fallback：匹配到 _Transformer 之前
    m = re.search(r"cycle4_(" + prefix + r".*?)_(?:Transformer|TimeMixer|ETTh|ETTm|Weather)", header)
    if m:
        return m.group(1)
    return ""

def plot_3factor_heatmap(records: list[dict], outdir: str):
    """绘制三因素联合扫描热力图"""
    filtered = filter_by_des(records, "WVEmbsCycle4_3factor")
    if not filtered:
        print("[WARN] No WVEmbsCycle4_3factor results found, skipping heatmap")
        return

    # 解析 tag → (scale_mode, sampling_config)
    data = {}  # {(scale_mode, config_label): mse}
    for r in filtered:
        tag = extract_tag_from_header(r["header"], "sm_")
        if not tag:
            continue
        # tag 示例: sm_standard_iss, sm_none_jss_0.25, sm_prior_jss_1.0
        parts = tag.split("_")
        # sm_{mode}_{sampling}[_{std}]
        if len(parts) >= 3:
            mode = parts[1]  # standard/none/prior
            sampling = parts[2]  # iss/jss
            if sampling == "jss" and len(parts) >= 4:
                std = parts[3]
                config_label = f"jss(σ={std})"
            else:
                config_label = "iss"
            data[(mode, config_label)] = r["mse"]

    if not data:
        print("[WARN] Could not parse 3-factor tags, skipping heatmap")
        return

    # 构建矩阵
    scale_modes = ["standard", "none", "prior"]
    configs = [
        "iss",
        "jss(σ=0.05)",
        "jss(σ=0.1)",
        "jss(σ=0.25)",
        "jss(σ=0.5)",
        "jss(σ=1.0)",
        "jss(σ=2.0)",
    ]

    matrix = np.full((len(scale_modes), len(configs)), np.nan)
    for i, sm in enumerate(scale_modes):
        for j, cfg in enumerate(configs):
            if (sm, cfg) in data:
                matrix[i, j] = data[(sm, cfg)]

    # 绘制热力图
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # 对 NaN 使用灰色
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="lightgray")

    # 计算有效值范围（排除极端值用于更好的颜色映射）
    valid = matrix[~np.isnan(matrix)]
    if len(valid) > 0:
        vmin = np.percentile(valid, 5)
        vmax = np.percentile(valid, 95)
        # 至少保留一定范围
        if vmax - vmin < 1.0:
            vmin = valid.min()
            vmax = valid.max()
    else:
        vmin, vmax = 0, 1

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # 标注数值
    for i in range(len(scale_modes)):
        for j in range(len(configs)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "NaN"
                color = "red"
            else:
                text = f"{val:.2f}"
                color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(scale_modes)))
    ax.set_yticklabels(scale_modes, fontsize=10)
    ax.set_xlabel("Sampling Configuration", fontsize=11)
    ax.set_ylabel("scale_mode", fontsize=11)
    ax.set_title(
        "Cycle 4: wv_sampling × wv_jss_std × scale_mode → MSE (ETTh1, inverse=True)",
        fontsize=12,
    )

    plt.colorbar(im, ax=ax, label="MSE", shrink=0.8)
    plt.tight_layout()

    outpath = os.path.join(outdir, "cycle4_heatmap.pdf")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Heatmap saved to {outpath}")

    # 同时保存 PNG 版本方便快速查看
    outpath_png = os.path.join(outdir, "cycle4_heatmap.png")
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 4))
    im2 = ax2.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(len(scale_modes)):
        for j in range(len(configs)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "NaN"
                color = "red"
            else:
                text = f"{val:.2f}"
                color = "white" if val > (vmin + vmax) / 2 else "black"
            ax2.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels(configs, rotation=30, ha="right", fontsize=9)
    ax2.set_yticks(range(len(scale_modes)))
    ax2.set_yticklabels(scale_modes, fontsize=10)
    ax2.set_xlabel("Sampling Configuration", fontsize=11)
    ax2.set_ylabel("scale_mode", fontsize=11)
    ax2.set_title(
        "Cycle 4: wv_sampling × wv_jss_std × scale_mode → MSE (ETTh1, inverse=True)",
        fontsize=12,
    )
    plt.colorbar(im2, ax=ax2, label="MSE", shrink=0.8)
    plt.tight_layout()
    fig2.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Heatmap PNG saved to {outpath_png}")


def plot_wvbase(records: list[dict], outdir: str):
    """绘制 wv_base 灵敏度折线图"""
    filtered = filter_by_des(records, "WVEmbsCycle4_wvbase")
    if not filtered:
        print("[WARN] No WVEmbsCycle4_wvbase results found, skipping wv_base plot")
        return

    data = {}  # {wv_base: mse}
    for r in filtered:
        # model_id: WVEmbs_ETTh1_Transformer_cycle4_base_{value}
        m = re.search(r"cycle4_base_([\d.]+)", r["header"])
        if m:
            base = float(m.group(1))
            data[base] = r["mse"]

    if not data:
        print("[WARN] Could not parse wv_base tags, skipping")
        return

    bases = sorted(data.keys())
    mses = [data[b] for b in bases]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(bases, mses, "o-", color="#2196F3", linewidth=2, markersize=8, label="MSE")

    # 标注最优点
    best_idx = np.nanargmin(mses)
    ax.annotate(
        f"Best: {mses[best_idx]:.2f}\n(base={bases[best_idx]:.0f})",
        xy=(bases[best_idx], mses[best_idx]),
        xytext=(30, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
        fontweight="bold",
    )

    ax.set_xscale("log")
    ax.set_xlabel("wv_base (log scale)", fontsize=11)
    ax.set_ylabel("MSE (inverse=True)", fontsize=11)
    ax.set_title(
        "Cycle 4: wv_base Sensitivity (ETTh1, embed=wv, none+iss)", fontsize=12
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    outpath = os.path.join(outdir, "cycle4_wvbase.pdf")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    outpath_png = os.path.join(outdir, "cycle4_wvbase.png")
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] wv_base plot saved to {outpath} / {outpath_png}")


def plot_etth2(records: list[dict], outdir: str):
    """绘制 ETTh2 跨数据集验证小热力图"""
    filtered = filter_by_des(records, "WVEmbsCycle4_ETTh2_jssstd")
    if not filtered:
        print("[WARN] No WVEmbsCycle4_ETTh2_jssstd results found, skipping ETTh2 plot")
        return

    data = {}
    for r in filtered:
        tag = extract_tag_from_header(r["header"], "sm_")
        if not tag:
            continue
        parts = tag.split("_")
        if len(parts) >= 4:
            mode = parts[1]  # none/prior
            std = parts[3]
            data[(mode, std)] = r["mse"]

    if not data:
        print("[WARN] Could not parse ETTh2 tags, skipping")
        return

    scale_modes = ["none", "prior"]
    stds = ["0.1", "0.25", "0.5"]

    matrix = np.full((len(scale_modes), len(stds)), np.nan)
    for i, sm in enumerate(scale_modes):
        for j, std in enumerate(stds):
            if (sm, std) in data:
                matrix[i, j] = data[(sm, std)]

    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="lightgray")

    valid = matrix[~np.isnan(matrix)]
    if len(valid) > 0:
        vmin, vmax = valid.min(), valid.max()
    else:
        vmin, vmax = 0, 1

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(len(scale_modes)):
        for j in range(len(stds)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "NaN"
                color = "red"
            else:
                text = f"{val:.2f}"
                color = "white" if val > (vmin + vmax) / 2 else "black"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
                fontweight="bold",
            )

    ax.set_xticks(range(len(stds)))
    ax.set_xticklabels([f"σ={s}" for s in stds], fontsize=10)
    ax.set_yticks(range(len(scale_modes)))
    ax.set_yticklabels(scale_modes, fontsize=10)
    ax.set_xlabel("wv_jss_std", fontsize=11)
    ax.set_ylabel("scale_mode", fontsize=11)
    ax.set_title(
        "Cycle 4: ETTh2 Cross-Dataset Validation (jss_std × scale_mode → MSE)",
        fontsize=12,
    )

    plt.colorbar(im, ax=ax, label="MSE", shrink=0.8)
    plt.tight_layout()

    outpath = os.path.join(outdir, "cycle4_etth2.pdf")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    outpath_png = os.path.join(outdir, "cycle4_etth2.png")
    fig.savefig(outpath_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] ETTh2 plot saved to {outpath} / {outpath_png}")


def print_summary_table(records: list[dict]):
    """打印所有 Cycle 4 结果的汇总表"""
    all_c4 = [r for r in records if "WVEmbsCycle4" in r["header"]]
    if not all_c4:
        print("[INFO] No Cycle 4 results found yet.")
        return

    print("\n" + "=" * 80)
    print("Cycle 4 Results Summary")
    print("=" * 80)

    for des_tag in [
        "WVEmbsCycle4_3factor",
        "WVEmbsCycle4_wvbase",
        "WVEmbsCycle4_ETTh2_jssstd",
    ]:
        subset = filter_by_des(all_c4, des_tag)
        if not subset:
            continue
        print(f"\n--- {des_tag} ({len(subset)} experiments) ---")
        print(f"{'Tag':<35} {'MSE':>12} {'MAE':>12}")
        print("-" * 60)
        for r in sorted(
            subset, key=lambda x: x["mse"] if not np.isnan(x["mse"]) else 1e10
        ):
            tag = extract_tag_from_header(
                r["header"], "sm_"
            ) or extract_tag_from_header(r["header"], "base_")
            if not tag:
                # 尝试 base_ 前缀
                m = re.search(r"cycle4_(base_[\d.]+)", r["header"])
                tag = m.group(1) if m else r["header"][:35]
            mse_str = f"{r['mse']:.6f}" if not np.isnan(r["mse"]) else "NaN"
            mae_str = f"{r['mae']:.6f}" if not np.isnan(r["mae"]) else "NaN"
            print(f"{tag:<35} {mse_str:>12} {mae_str:>12}")


def main():
    parser = argparse.ArgumentParser(description="Cycle 4 可视化")
    parser.add_argument(
        "--results",
        type=str,
        default="result_long_term_forecast.txt",
        help="结果文件路径",
    )
    parser.add_argument("--outdir", type=str, default="results/", help="输出目录")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"[ERROR] Results file not found: {args.results}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    records = parse_results(args.results)
    print(f"[INFO] Parsed {len(records)} total experiment records")

    # 打印汇总表
    print_summary_table(records)

    # 生成可视化
    plot_3factor_heatmap(records, args.outdir)
    plot_wvbase(records, args.outdir)
    plot_etth2(records, args.outdir)

    print("\n[DONE] All Cycle 4 plots generated.")


if __name__ == "__main__":
    main()
