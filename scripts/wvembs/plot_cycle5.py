#!/usr/bin/env python3
"""Cycle 5 可视化：掩码消融热力图 + 外推柱状图

读取 result_long_term_forecast.txt，按 DES 标签过滤，生成：
1. 掩码消融热力图（mask_type × mask_prob → MSE，dlow_min=0 和 dlow_min=4 各一张）
2. 外推模式柱状图（direct vs scale(1.5/2.0/5.0) → MSE）
3. 汇总表打印到 stdout

用法：
    python scripts/wvembs/plot_cycle5.py [--results result_long_term_forecast.txt] [--outdir results/]
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


def parse_mask_tag(header: str) -> dict | None:
    """从 header 中解析掩码消融实验的参数

    实际 header 格式: ...cycle5_mt_{mask_type}_mp_{prob}_dl_{dlow_min}_...
    例如: cycle5_mt_zero_mp_0.1_dl_0, cycle5_mt_phase_rotate_mp_0.3_dl_4
    返回: {"mask_type": str, "prob": str, "dlow_min": str} 或 None
    """
    # 匹配 cycle5_mt_{type}_mp_{prob}_dl_{dlow}
    m = re.search(
        r"cycle5_mt_([a-zA-Z_]+?)_mp_([\d.]+)_dl_(\d+)",
        header,
    )
    if m:
        mask_type = m.group(1)  # zero / arcsine / phase_rotate
        prob = m.group(2)  # 0.1 / 0.3 / 0.5
        dlow_min = m.group(3)  # 0 / 4
        return {"mask_type": mask_type, "prob": prob, "dlow_min": dlow_min}
    return None


def parse_extrap_tag(header: str) -> dict | None:
    """从 header 中解析外推实验的参数

    实际 header 格式: ...cycle5_extrap_scale_{scale}_... 或 ...cycle5_extrap_direct_...
    例如: cycle5_extrap_direct, cycle5_extrap_scale_1.5
    返回: {"mode": str, "scale": str|None} 或 None
    """
    # 匹配 scale 模式：cycle5_extrap_scale_{scale}
    m_scale = re.search(r"cycle5_extrap_scale_([\d.]+)", header)
    if m_scale:
        return {"mode": "scale", "scale": m_scale.group(1)}

    # 匹配 direct 模式：cycle5_extrap_direct
    m_direct = re.search(r"cycle5_extrap_direct", header)
    if m_direct:
        return {"mode": "direct", "scale": None}

    return None


def plot_mask_heatmap(records: list[dict], outdir: str):
    """绘制掩码消融热力图（左：dlow_min=0，右：dlow_min=4）"""
    filtered = filter_by_des(records, "WVEmbsCycle5_mask")
    if not filtered:
        print("[WARN] No WVEmbsCycle5_mask results found, skipping mask heatmap")
        return

    # 解析所有掩码实验
    # data[dlow_min][(mask_type, prob)] = mse
    data = defaultdict(dict)
    for r in filtered:
        tag = parse_mask_tag(r["header"])
        if tag:
            key = (tag["mask_type"], tag["prob"])
            data[tag["dlow_min"]][key] = r["mse"]

    if not data:
        print("[WARN] Could not parse mask tags, skipping heatmap")
        return

    mask_types = ["zero", "arcsine", "phase_rotate"]
    mask_probs = ["0.1", "0.3", "0.5"]
    dlow_labels = ["0", "4"]

    # 构建 2 个矩阵（dlow_min=0 和 dlow_min=4）
    matrices = {}
    for dlow in dlow_labels:
        mat = np.full((len(mask_types), len(mask_probs)), np.nan)
        for i, mt in enumerate(mask_types):
            for j, mp in enumerate(mask_probs):
                if (mt, mp) in data.get(dlow, {}):
                    mat[i, j] = data[dlow][(mt, mp)]
        matrices[dlow] = mat

    # colormap 设置（绿色=低 MSE=更优）
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="lightgray")

    # 计算全局 vmin/vmax（两张图共用，便于比较）
    all_valid = []
    for mat in matrices.values():
        valid = mat[~np.isnan(mat)]
        all_valid.extend(valid.tolist())
    if all_valid:
        vmin = np.percentile(all_valid, 5)
        vmax = np.percentile(all_valid, 95)
        if vmax - vmin < 0.5:
            vmin = min(all_valid)
            vmax = max(all_valid)
    else:
        vmin, vmax = 0, 1

    # 找到全局最优（MSE 最低，排除 NaN）
    global_best_val = float("inf")
    global_best_loc = None  # (dlow_idx, row, col)
    for di, dlow in enumerate(dlow_labels):
        mat = matrices[dlow]
        for i in range(len(mask_types)):
            for j in range(len(mask_probs)):
                if not np.isnan(mat[i, j]) and mat[i, j] < global_best_val:
                    global_best_val = mat[i, j]
                    global_best_loc = (di, i, j)

    # 绘图：两列子图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Cycle 5: Mask Ablation (ETTh1, embed=wv, none+iss, inverse=True)",
        fontsize=13,
        fontweight="bold",
    )

    # 基线 MSE（无掩码，从extrap direct结果获取，即 none+iss 无掩码基线）
    baseline_mse = None
    # 尝试从外推实验的 direct 模式获取基线
    extrap_records = filter_by_des(records, "WVEmbsCycle5_extrap")
    for r in extrap_records:
        if "extrap_direct" in r["header"]:
            baseline_mse = r["mse"]
            break
    # 如果没有找到，尝试从 Cycle 4 结果获取 none+iss 基线
    if baseline_mse is None:
        c4_records = filter_by_des(records, "WVEmbsCycle4_3factor")
        for r in c4_records:
            if "sm_none_iss" in r["header"]:
                baseline_mse = r["mse"]
                break

    ims = []
    for di, (ax, dlow) in enumerate(zip(axes, dlow_labels)):
        mat = matrices[dlow]
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ims.append(im)

        # 标注数值，加粗
        for i in range(len(mask_types)):
            for j in range(len(mask_probs)):
                val = mat[i, j]
                if np.isnan(val):
                    text = "NaN"
                    color = "red"
                    fw = "normal"
                else:
                    text = f"{val:.3f}"
                    color = "white" if val > (vmin + vmax) / 2 else "black"
                    fw = "bold"

                # 全局最优单元格：加星号标记
                if global_best_loc and global_best_loc == (di, i, j):
                    text = f"★{text}"
                    color = "gold"
                    fw = "bold"

                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight=fw,
                )

        ax.set_xticks(range(len(mask_probs)))
        ax.set_xticklabels([f"p={p}" for p in mask_probs], fontsize=10)
        ax.set_yticks(range(len(mask_types)))
        ax.set_yticklabels(mask_types, fontsize=10)
        ax.set_xlabel("mask_prob", fontsize=11)
        ax.set_ylabel("mask_type", fontsize=11)
        ax.set_title(f"dlow_min={dlow}", fontsize=11)

    # 添加颜色条（共用最右侧子图）
    plt.colorbar(ims[-1], ax=axes, label="MSE", shrink=0.8, pad=0.02)

    # 若有基线则用红色虚线注释（在两张图的标题区添加文字）
    if baseline_mse is not None and not np.isnan(baseline_mse):
        fig.text(
            0.5,
            0.01,
            f"No-mask baseline MSE = {baseline_mse:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="red",
            style="italic",
        )

    plt.tight_layout()

    # 保存 PDF + PNG
    for ext in ["pdf", "png"]:
        outpath = os.path.join(outdir, f"cycle5_mask_heatmap.{ext}")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"[OK] Mask heatmap ({ext}) saved to {outpath}")

    plt.close()


def plot_extrap_bar(records: list[dict], outdir: str):
    """绘制外推模式柱状图"""
    filtered = filter_by_des(records, "WVEmbsCycle5_extrap")
    if not filtered:
        print("[WARN] No WVEmbsCycle5_extrap results found, skipping extrap bar chart")
        return

    # 解析外推实验
    # bar_data: label → mse
    bar_data = {}
    baseline_mse = None

    for r in filtered:
        tag = parse_extrap_tag(r["header"])
        if tag is None:
            continue

        if tag["mode"] == "direct":
            label = "direct"
            # direct 模式就是无外推基线
            baseline_mse = r["mse"]
        else:
            label = f"scale({tag['scale']})"

        bar_data[label] = r["mse"]

        if tag["mode"] == "direct":
            label = "direct"
        else:
            label = f"scale({tag['scale']})"

        bar_data[label] = r["mse"]

    if not bar_data:
        print("[WARN] Could not parse extrap tags, skipping bar chart")
        return

    # 排列顺序：direct, scale(1.5), scale(2.0), scale(5.0)
    preferred_order = ["direct", "scale(1.5)", "scale(2.0)", "scale(5.0)"]
    labels = [lbl for lbl in preferred_order if lbl in bar_data]
    # 补充任何未在 preferred_order 中但存在的 key
    for lbl in sorted(bar_data.keys()):
        if lbl not in labels:
            labels.append(lbl)

    mses = [bar_data[lbl] for lbl in labels]

    # 颜色方案
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63", "#9C27B0", "#795548"]
    bar_colors = colors[: len(labels)]

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        mses,
        color=bar_colors,
        width=0.55,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )

    # 标注每根柱子的 MSE 值
    for bar, mse_val in zip(bars, mses):
        if np.isnan(mse_val):
            label_text = "NaN"
            color = "red"
        else:
            label_text = f"{mse_val:.4f}"
            color = "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003 * max([v for v in mses if not np.isnan(v)] or [1]),
            label_text,
            ha="center",
            va="bottom",
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    # 基线红色虚线（若有基线记录则用该值，否则留空）
    if baseline_mse is not None and not np.isnan(baseline_mse):
        ax.axhline(
            y=baseline_mse,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"no-mask baseline ({baseline_mse:.4f})",
        )
        ax.legend(fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("Extrapolation Mode", fontsize=12)
    ax.set_ylabel("MSE (inverse=True)", fontsize=12)
    ax.set_title(
        "Cycle 5: Extrapolation Mode Comparison (ETTh1, embed=wv, none+iss)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # 保存 PDF + PNG
    for ext in ["pdf", "png"]:
        outpath = os.path.join(outdir, f"cycle5_extrap_bar.{ext}")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"[OK] Extrap bar chart ({ext}) saved to {outpath}")

    plt.close()


def print_summary_table(records: list[dict]):
    """打印所有 Cycle 5 结果的汇总表（按 MSE 升序）"""
    all_c5 = [r for r in records if "WVEmbsCycle5" in r["header"]]
    if not all_c5:
        print("[INFO] No Cycle 5 results found yet.")
        return

    print("\n" + "=" * 90)
    print("Cycle 5 Results Summary (sorted by MSE ascending)")
    print("=" * 90)

    # 按子组分类打印
    for des_tag in [
        "WVEmbsCycle5_mask",
        "WVEmbsCycle5_extrap",
        "WVEmbsCycle5_crossval",
        "WVEmbsCycle5_baseline",
    ]:
        subset = filter_by_des(all_c5, des_tag)
        if not subset:
            continue
        print(f"\n--- {des_tag} ({len(subset)} experiments) ---")
        print(f"{'Tag':<50} {'MSE':>12} {'MAE':>12}")
        print("-" * 75)
        for r in sorted(
            subset, key=lambda x: x["mse"] if not np.isnan(x["mse"]) else 1e10
        ):
            # 尝试提取短标签
            m_mask = re.search(r"cycle5_(mt_[\w.]+_mp_[\d.]+_dl_\d+)", r["header"])
            m_extrap = re.search(r"cycle5_(extrap(?:_[\w.]+)*)", r["header"])
            if m_mask:
                tag_str = m_mask.group(1)
            elif m_extrap:
                tag_str = m_extrap.group(1)
            else:
                tag_str = r["header"][:50]

            mse_str = f"{r['mse']:.6f}" if not np.isnan(r["mse"]) else "NaN"
            mae_str = f"{r['mae']:.6f}" if not np.isnan(r["mae"]) else "NaN"
            print(f"{tag_str:<50} {mse_str:>12} {mae_str:>12}")

    # 未归类的 Cycle5 记录
    classified_tags = [
        "WVEmbsCycle5_mask",
        "WVEmbsCycle5_extrap",
        "WVEmbsCycle5_crossval",
        "WVEmbsCycle5_baseline",
    ]
    unclassified = [
        r for r in all_c5 if not any(t in r["header"] for t in classified_tags)
    ]
    if unclassified:
        print(f"\n--- Other WVEmbsCycle5 ({len(unclassified)} experiments) ---")
        print(f"{'Tag':<50} {'MSE':>12} {'MAE':>12}")
        print("-" * 75)
        for r in sorted(
            unclassified, key=lambda x: x["mse"] if not np.isnan(x["mse"]) else 1e10
        ):
            tag_str = r["header"][:50]
            mse_str = f"{r['mse']:.6f}" if not np.isnan(r["mse"]) else "NaN"
            mae_str = f"{r['mae']:.6f}" if not np.isnan(r["mae"]) else "NaN"
            print(f"{tag_str:<50} {mse_str:>12} {mae_str:>12}")


def main():
    """主函数：解析参数，读取结果文件，生成可视化"""
    parser = argparse.ArgumentParser(description="Cycle 5 可视化：掩码消融 + 外推实验")
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
    plot_mask_heatmap(records, args.outdir)
    plot_extrap_bar(records, args.outdir)

    print("\n[DONE] All Cycle 5 plots generated.")


if __name__ == "__main__":
    main()
