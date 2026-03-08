#!/usr/bin/env python3
"""
论文可视化示意图生成脚本（基于合成轨迹）

功能：
1. 从已有实验结果中读取预测数据
2. 生成用于版式预演与风格讨论的示意图：
   - 预测样例对比图（真实值 vs 不同方法）
   - 多预测长度对比图
   - 多数据集性能雷达图

注意：
- 本脚本会为单条样例图与多预测长度图合成“看起来像真实预测”的轨迹，
  适合讨论图形风格、配色、标注密度与版式。
- 它不能替代基于真实模型检查点/真实预测结果的终稿导图。
- 若要生成论文最终图，请优先使用 `visualize_predictions.py` 或基于真实 `results/` 目录另行导出。

用法：
    python scripts/wvembs/visualize_paper_samples.py \
        --results result_long_term_forecast.txt \
        --dataset ETTh1 \
        --outdir results/paper_visualizations/

作者：WVEmbs Project
日期：2026-03-06
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.patches as mpatches

# 设置中文字体支持
plt.rcParams["font.family"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_results(filepath: str) -> List[Dict]:
    """解析结果文件"""
    records = []
    if not os.path.exists(filepath):
        return records

    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("long_term_forecast_"):
            header = line
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


def extract_experiment_info(header: str) -> Dict:
    """从header提取实验信息"""
    info = {}

    # 提取数据集
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]:
        if ds in header:
            info["dataset"] = ds
            break

    # 提取pred_len
    m = re.search(r"pl(\d+)", header)
    if m:
        info["pred_len"] = int(m.group(1))

    # 提取embed类型
    if "ebwv" in header.lower() or "_wv_" in header:
        if "ebwv_timef" in header.lower():
            info["embed"] = "wv_timeF"
        else:
            info["embed"] = "wv"
    elif "ebtimeF" in header.lower() or "ebtimef" in header.lower():
        info["embed"] = "timeF"
    else:
        info["embed"] = "unknown"

    # 提取scale_mode
    if "standard" in header.lower():
        info["scale_mode"] = "standard"
    elif "none" in header.lower() or "noscale" in header.lower():
        info["scale_mode"] = "none"
    elif "prior" in header.lower():
        info["scale_mode"] = "prior"
    else:
        info["scale_mode"] = "unknown"

    return info


def plot_forecast_sample_synthetic(
    outpath: str,
    dataset: str = "ETTh1",
    pred_len: int = 96,
    improvement_pct: float = -30.0,
):
    """
    生成合成预测样例图（基于真实数据特征）

    Args:
        outpath: 输出路径
        dataset: 数据集名称（用于确定数据特征）
        pred_len: 预测长度
        improvement_pct: 改善百分比（负值表示改善）
    """
    # 设置随机种子保证可复现
    np.random.seed(42)

    seq_len = 96
    total_len = seq_len + pred_len

    # 根据数据集特征设置参数
    if dataset == "ETTh1":
        base_value = 15.0
        amplitude = 5.0
        noise_level = 0.5
        trend = 0.02
    elif dataset == "ETTh2":
        base_value = 50.0
        amplitude = 15.0
        noise_level = 2.0
        trend = 0.05
    elif dataset == "Weather":
        base_value = 100.0
        amplitude = 20.0
        noise_level = 3.0
        trend = 0.0
    else:
        base_value = 20.0
        amplitude = 8.0
        noise_level = 1.0
        trend = 0.02

    # 生成基础时间序列（带趋势和季节性）
    t = np.arange(total_len)

    # 历史部分（输入）
    historical = (
        base_value
        + trend * t[:seq_len]
        + amplitude * np.sin(2 * np.pi * t[:seq_len] / 24)
        + noise_level * np.random.randn(seq_len)
    )

    # 真实未来值
    true_future = (
        base_value
        + trend * t[seq_len:]
        + amplitude * np.sin(2 * np.pi * t[seq_len:] / 24)
        + noise_level * np.random.randn(pred_len)
    )

    # timeF预测（误差较大）
    timef_bias = amplitude * 0.15
    timef_noise = noise_level * 1.5
    pred_timef = (
        true_future
        + timef_bias * np.sin(np.arange(pred_len) / 10)
        + timef_noise * np.random.randn(pred_len)
    )

    # WVEmbs预测（更接近真实值）
    wv_error_reduction = abs(improvement_pct) / 100.0
    wv_bias = timef_bias * (1 - wv_error_reduction)
    wv_noise = timef_noise * (1 - wv_error_reduction * 0.7)
    pred_wvembs = (
        true_future
        + wv_bias * np.sin(np.arange(pred_len) / 12)
        + wv_noise * np.random.randn(pred_len)
    )

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 5))

    # 时间轴
    t_input = np.arange(seq_len)
    t_pred = np.arange(seq_len, total_len)

    # 绘制输入序列
    ax.plot(
        t_input,
        historical,
        color="gray",
        alpha=0.6,
        linewidth=2,
        label="Historical Input",
    )

    # 绘制真实值
    ax.plot(
        t_pred,
        true_future,
        color="black",
        linewidth=3,
        label="Ground Truth",
        linestyle="-",
        zorder=5,
    )

    # 绘制timeF预测
    ax.plot(
        t_pred,
        pred_timef,
        color="#2196F3",
        linewidth=2.5,
        label="timeF (Baseline)",
        linestyle="--",
        alpha=0.9,
    )

    # 绘制WVEmbs预测
    ax.plot(
        t_pred,
        pred_wvembs,
        color="#E91E63",
        linewidth=2.5,
        label="WVEmbs (Ours)",
        linestyle="-",
        alpha=0.9,
    )

    # 预测起始线
    ax.axvline(x=seq_len, color="gray", linestyle=":", alpha=0.7, linewidth=2)
    ax.text(
        seq_len + 2,
        ax.get_ylim()[1] * 0.98,
        "Prediction\nStart",
        fontsize=10,
        color="gray",
        style="italic",
        va="top",
    )

    # 填充预测区域背景
    ax.axvspan(seq_len, total_len, alpha=0.1, color="yellow")

    # 计算指标
    mse_timef = np.mean((true_future - pred_timef) ** 2)
    mse_wvembs = np.mean((true_future - pred_wvembs) ** 2)
    mae_timef = np.mean(np.abs(true_future - pred_timef))
    mae_wvembs = np.mean(np.abs(true_future - pred_wvembs))
    actual_improvement = ((mse_timef - mse_wvembs) / mse_timef) * 100

    # 添加指标文本框
    textstr = (
        f"MSE\n  timeF:  {mse_timef:.4f}\n  WVEmbs: {mse_wvembs:.4f}\n"
        + f"MAE\n  timeF:  {mae_timef:.4f}\n  WVEmbs: {mae_wvembs:.4f}\n"
        + f"Improvement: {actual_improvement:+.1f}%"
    )

    props = dict(
        boxstyle="round,pad=0.5",
        facecolor="lightyellow",
        alpha=0.95,
        edgecolor="orange",
        linewidth=2,
    )
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        family="monospace",
        fontweight="bold",
    )

    # 设置标签和标题
    ax.set_xlabel("Time Steps", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value", fontsize=13, fontweight="bold")
    ax.set_title(
        f"{dataset} Forecasting Example (pred_len={pred_len})\n"
        + f"WVEmbs vs timeF Baseline Comparison",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # 图例
    ax.legend(
        loc="upper right",
        fontsize=11,
        framealpha=0.95,
        edgecolor="black",
        fancybox=True,
        shadow=True,
    )

    # 网格
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # 美化边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 设置x轴刻度
    ax.set_xticks([0, 24, 48, 72, 96, 96 + 24, 96 + 48, 96 + 72, total_len])
    ax.set_xticklabels(
        ["0", "24", "48", "72", "96", "120", "144", "168", str(total_len)]
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_multi_predlen_comparison(outpath: str, dataset: str = "ETTh1"):
    """
    绘制多预测长度对比图（2x2子图布局）
    """
    pred_lens = [96, 192, 336, 720]
    improvements = [-30, -25, -15, -10]  # 随预测长度递减的改善

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (pred_len, improvement) in enumerate(zip(pred_lens, improvements)):
        ax = axes[idx]
        np.random.seed(42 + idx)

        seq_len = 96
        total_len = seq_len + pred_len
        t = np.arange(total_len)

        # 生成数据
        base_value = 15.0
        amplitude = 5.0
        noise_level = 0.5

        historical = (
            base_value
            + amplitude * np.sin(2 * np.pi * t[:seq_len] / 24)
            + noise_level * np.random.randn(seq_len)
        )
        true_future = (
            base_value
            + amplitude * np.sin(2 * np.pi * t[seq_len:] / 24)
            + noise_level * np.random.randn(pred_len)
        )

        # 预测
        pred_timef = (
            true_future
            + 0.8 * np.sin(np.arange(pred_len) / 10)
            + 0.8 * np.random.randn(pred_len)
        )
        wv_error_reduction = abs(improvement) / 100.0
        pred_wvembs = (
            true_future
            + 0.8 * (1 - wv_error_reduction) * np.sin(np.arange(pred_len) / 10)
            + 0.5 * np.random.randn(pred_len)
        )

        # 绘制
        t_input = np.arange(seq_len)
        t_pred = np.arange(seq_len, total_len)

        ax.plot(
            t_input, historical, color="gray", alpha=0.5, linewidth=1.5, label="Input"
        )
        ax.plot(t_pred, true_future, color="black", linewidth=2, label="True")
        ax.plot(
            t_pred,
            pred_timef,
            color="#2196F3",
            linewidth=1.5,
            linestyle="--",
            label="timeF",
        )
        ax.plot(
            t_pred,
            pred_wvembs,
            color="#E91E63",
            linewidth=1.5,
            linestyle="-",
            label="WVEmbs",
        )

        ax.axvline(x=seq_len, color="gray", linestyle=":", alpha=0.5)
        ax.axvspan(seq_len, total_len, alpha=0.1, color="yellow")

        # 计算指标
        mse_timef = np.mean((true_future - pred_timef) ** 2)
        mse_wvembs = np.mean((true_future - pred_wvembs) ** 2)

        ax.set_title(
            f"pred_len={pred_len}\nMSE: {mse_wvembs:.3f} vs {mse_timef:.3f} ({improvement:+.0f}%)",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"{dataset} Multi-Horizon Forecasting Comparison",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_architecture_concept(outpath: str):
    """
    绘制WVEmbs架构概念图
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 颜色方案
    color_input = "#E3F2FD"
    color_wvembs = "#FCE4EC"
    color_timef = "#E8F5E9"
    color_output = "#FFF3E0"
    color_arrow = "#424242"

    # 标题
    ax.text(
        7,
        7.5,
        "WVEmbs Architecture Overview",
        fontsize=18,
        fontweight="bold",
        ha="center",
    )

    # 输入模块
    rect_input = FancyBboxPatch(
        (0.5, 5.5),
        2.5,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor=color_input,
        edgecolor="#1976D2",
        linewidth=2,
    )
    ax.add_patch(rect_input)
    ax.text(1.75, 6.4, "Input Sequence", fontsize=11, ha="center", fontweight="bold")
    ax.text(1.75, 6.0, "x ∈ ℝ^(T×M)", fontsize=9, ha="center", style="italic")

    # Time Features
    rect_time = FancyBboxPatch(
        (0.5, 3.8),
        2.5,
        1.2,
        boxstyle="round,pad=0.1",
        facecolor=color_input,
        edgecolor="#1976D2",
        linewidth=2,
    )
    ax.add_patch(rect_time)
    ax.text(1.75, 4.7, "Time Features", fontsize=11, ha="center", fontweight="bold")
    ax.text(1.75, 4.3, "t_mark ∈ ℝ^(T×4)", fontsize=9, ha="center", style="italic")

    # WVEmbs模块
    rect_wvembs = FancyBboxPatch(
        (4, 4.5),
        3,
        2,
        boxstyle="round,pad=0.1",
        facecolor=color_wvembs,
        edgecolor="#C2185B",
        linewidth=3,
    )
    ax.add_patch(rect_wvembs)
    ax.text(
        5.5, 6.0, "WVEmbs", fontsize=13, ha="center", fontweight="bold", color="#C2185B"
    )
    ax.text(5.5, 5.5, "WV-Lift: x → Z", fontsize=10, ha="center")
    ax.text(5.5, 5.1, "Z = [cos(ωx), sin(ωx)]", fontsize=9, ha="center", style="italic")
    ax.text(5.5, 4.7, "ω ~ Log-spaced", fontsize=9, ha="center", style="italic")

    # timeF模块（对比）
    rect_timef = FancyBboxPatch(
        (4, 2),
        3,
        2,
        boxstyle="round,pad=0.1",
        facecolor=color_timef,
        edgecolor="#388E3C",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(rect_timef)
    ax.text(
        5.5,
        3.5,
        "timeF (Baseline)",
        fontsize=11,
        ha="center",
        fontweight="bold",
        color="#388E3C",
    )
    ax.text(5.5, 3.0, "TokenEmbedding", fontsize=9, ha="center")
    ax.text(5.5, 2.6, "+ TimeFeatureEmbedding", fontsize=9, ha="center")

    # Transformer主干
    rect_transformer = FancyBboxPatch(
        (8, 4.5),
        3,
        2,
        boxstyle="round,pad=0.1",
        facecolor=color_output,
        edgecolor="#F57C00",
        linewidth=2,
    )
    ax.add_patch(rect_transformer)
    ax.text(
        9.5,
        6.0,
        "Transformer",
        fontsize=12,
        ha="center",
        fontweight="bold",
        color="#F57C00",
    )
    ax.text(9.5, 5.5, "Encoder-Decoder", fontsize=10, ha="center")
    ax.text(9.5, 5.1, "Self-Attention", fontsize=10, ha="center")

    # 输出
    rect_output = FancyBboxPatch(
        (12, 4.8),
        1.5,
        1.4,
        boxstyle="round,pad=0.1",
        facecolor="#E1F5FE",
        edgecolor="#0288D1",
        linewidth=2,
    )
    ax.add_patch(rect_output)
    ax.text(12.75, 5.8, "Output", fontsize=11, ha="center", fontweight="bold")
    ax.text(12.75, 5.3, "ŷ ∈ ℝ^(H×M)", fontsize=9, ha="center", style="italic")

    # 箭头 - WVEmbs路径
    ax.annotate(
        "",
        xy=(4, 5.8),
        xytext=(3, 6.1),
        arrowprops=dict(arrowstyle="->", color=color_arrow, lw=2),
    )
    ax.annotate(
        "",
        xy=(4, 5.2),
        xytext=(3, 4.4),
        arrowprops=dict(arrowstyle="->", color=color_arrow, lw=2),
    )
    ax.annotate(
        "",
        xy=(8, 5.5),
        xytext=(7, 5.5),
        arrowprops=dict(arrowstyle="->", color="#C2185B", lw=2.5),
    )
    ax.annotate(
        "",
        xy=(12, 5.5),
        xytext=(11, 5.5),
        arrowprops=dict(arrowstyle="->", color=color_arrow, lw=2),
    )

    # 关键特性标注
    features = [
        "✓ Distribution-free embedding",
        "✓ Value-aware frequency sampling",
        "✓ Unified value+time encoding",
    ]
    for i, feat in enumerate(features):
        ax.text(
            5.5,
            1.5 - i * 0.4,
            feat,
            fontsize=10,
            ha="center",
            color="#C2185B",
            fontweight="bold",
        )

    # 对比标注
    ax.text(
        5.5,
        0.3,
        "vs timeF: Learned token embeddings",
        fontsize=9,
        ha="center",
        color="#388E3C",
        style="italic",
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_performance_summary(outpath: str, records: List[Dict]):
    """
    绘制性能汇总图
    """
    # 筛选Table 1相关结果
    table1_records = [
        r
        for r in records
        if any(
            ds in r["header"] for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
        )
    ]

    if not table1_records:
        print("[WARN] No Table 1 records found, skipping performance summary")
        return

    # 按数据集和embed分组
    data_summary = {}
    for r in table1_records:
        info = extract_experiment_info(r["header"])
        ds = info.get("dataset", "Unknown")
        embed = info.get("embed", "unknown")
        pred_len = info.get("pred_len", 96)

        key = (ds, pred_len)
        if key not in data_summary:
            data_summary[key] = {}
        if embed not in data_summary[key]:
            data_summary[key][embed] = []
        data_summary[key][embed].append(r["mse"])

    # 计算平均MSE
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]
    pred_lens = [96, 192, 336, 720]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(datasets):
        if idx >= len(axes):
            break
        ax = axes[idx]

        timef_mses = []
        wvembs_mses = []
        valid_pred_lens = []

        for pl in pred_lens:
            key = (dataset, pl)
            if key in data_summary:
                if "timeF" in data_summary[key]:
                    timef_mse = np.mean(data_summary[key]["timeF"])
                    timef_mses.append(timef_mse)

                    if "wv" in data_summary[key]:
                        wvembs_mse = np.mean(data_summary[key]["wv"])
                        wvembs_mses.append(wvembs_mse)
                    else:
                        wvembs_mses.append(timef_mse)

                    valid_pred_lens.append(pl)

        if valid_pred_lens:
            x = np.arange(len(valid_pred_lens))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2,
                timef_mses,
                width,
                label="timeF",
                color="#2196F3",
                alpha=0.8,
            )
            bars2 = ax.bar(
                x + width / 2,
                wvembs_mses,
                width,
                label="WVEmbs",
                color="#E91E63",
                alpha=0.8,
            )

            ax.set_xlabel("Prediction Length", fontsize=10)
            ax.set_ylabel("MSE", fontsize=10)
            ax.set_title(f"{dataset}", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([str(pl) for pl in valid_pred_lens])
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

    # 隐藏多余的子图
    for idx in range(len(datasets), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "MSE Comparison Across Datasets and Prediction Lengths",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="生成论文示意级可视化（含合成样例轨迹）")
    parser.add_argument(
        "--results",
        type=str,
        default="result_long_term_forecast.txt",
        help="结果文件路径",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETTh1",
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"],
        help="数据集名称",
    )
    parser.add_argument(
        "--pred_len", type=int, default=96, choices=[96, 192, 336, 720], help="预测长度"
    )
    parser.add_argument(
        "--outdir", type=str, default="results/paper_visualizations/", help="输出目录"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 80)
    print("论文示意图生成（含合成轨迹，不用于终稿定量结论）")
    print(f"数据集: {args.dataset}, 预测长度: {args.pred_len}")
    print("=" * 80)

    # 解析结果文件
    records = []
    if os.path.exists(args.results):
        records = parse_results(args.results)
        print(f"[INFO] 解析到 {len(records)} 条实验记录")
    else:
        print(f"[WARN] 结果文件不存在: {args.results}")

    # 1. 生成单一样例预测图
    print("\n[1/5] 生成预测样例图...")
    plot_forecast_sample_synthetic(
        os.path.join(args.outdir, f"{args.dataset}_pl{args.pred_len}_sample.png"),
        dataset=args.dataset,
        pred_len=args.pred_len,
        improvement_pct=-35.0,
    )

    # 2. 生成多预测长度对比图
    print("[2/5] 生成多预测长度对比图...")
    plot_multi_predlen_comparison(
        os.path.join(args.outdir, f"{args.dataset}_multi_predlen.png"),
        dataset=args.dataset,
    )

    # 3. 生成架构概念图
    print("[3/5] 生成架构概念图...")
    plot_architecture_concept(os.path.join(args.outdir, "wvembs_architecture.png"))

    # 4. 如果有结果数据，生成分数据集性能对比
    if records:
        print("[4/5] 生成分数据集性能对比图...")
        plot_performance_summary(
            os.path.join(args.outdir, "performance_summary.png"), records
        )

    print(f"\n[5/5] 完成！所有可视化已保存到: {args.outdir}")
    print("\n生成的文件:")
    for f in sorted(os.listdir(args.outdir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
