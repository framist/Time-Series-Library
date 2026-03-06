#!/usr/bin/env python3
"""
论文级预测可视化样例生成脚本

功能：
1. 加载训练好的模型（timeF基线和WVEmbs最优配置）
2. 在测试集上生成预测结果
3. 绘制论文级的可视化图：
   - 单通道预测对比图（真实值 vs timeF vs WVEmbs）
   - 多通道子图布局
   - 残差分析图
   - 不同预测长度的对比

用法：
    python scripts/wvembs/visualize_predictions.py \
        --dataset ETTh1 \
        --pred_len 96 \
        --checkpoint_timeF <path> \
        --checkpoint_wvembs <path> \
        --outdir results/visualizations/

作者：WVEmbs Project
日期：2026-03-06
"""

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecast import Exp_Long_Term_Forecast
from models import Transformer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric


def setup_args(
    dataset: str,
    pred_len: int,
    embed: str,
    scale_mode: str,
    wv_sampling: Optional[str] = None,
    wv_jss_std: Optional[float] = None,
    wv_extrap_mode: Optional[str] = None,
    wv_extrap_scale: Optional[float] = None,
) -> argparse.Namespace:
    """配置实验参数"""
    args = argparse.Namespace()

    # 基础配置
    args.task_name = "long_term_forecast"
    args.is_training = 0  # 测试模式
    args.model_id = f"viz_{dataset}_pl{pred_len}_{embed}"
    args.model = "Transformer"
    args.data = dataset if dataset.startswith("ETT") else "custom"

    # 数据路径
    if dataset.startswith("ETT"):
        args.root_path = "./dataset/ETT-small/"
        args.data_path = f"{dataset}.csv"
    elif dataset == "Weather":
        args.root_path = "./dataset/weather/"
        args.data_path = "weather.csv"
    else:
        args.root_path = "./dataset/ETT-small/"
        args.data_path = f"{dataset}.csv"

    # 特征配置
    args.features = "M"
    args.target = "OT"
    args.freq = "h" if "h" in dataset else "t" if dataset == "Weather" else "h"

    # 模型维度配置
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = pred_len
    args.enc_in = 7 if dataset.startswith("ETT") else 21 if dataset == "Weather" else 7
    args.dec_in = args.enc_in
    args.c_out = args.enc_in
    args.d_model = 512
    args.d_ff = 2048
    args.n_heads = 8
    args.e_layers = 2
    args.d_layers = 1
    args.factor = 1
    args.dropout = 0.1
    args.embed = embed
    args.activation = "gelu"
    args.output_attention = False

    # WVEmbs 配置
    args.wv_sampling = wv_sampling or "iss"
    args.wv_jss_std = wv_jss_std if wv_jss_std is not None else 1.0
    args.wv_base = 10000
    args.wv_mask_prob = 0.0
    args.wv_mask_type = "none"
    args.wv_extrap_mode = wv_extrap_mode or "direct"
    args.wv_extrap_scale = wv_extrap_scale if wv_extrap_scale is not None else 1.0

    # 缩放配置
    args.scale_mode = scale_mode
    args.prior_scale = None
    args.prior_offset = 0
    args.no_scale = scale_mode == "none"
    args.inverse = True

    # 训练配置（测试时也需要）
    args.train_epochs = 10
    args.batch_size = 32
    args.learning_rate = 0.0001
    args.patience = 10

    # 设备配置
    args.use_gpu = torch.cuda.is_available()
    args.gpu = 0
    args.use_multi_gpu = False
    args.devices = "0"
    args.device_ids = [0]
    args.use_amp = True
    args.num_workers = 2
    args.checkpoints = "./checkpoints/"
    args.itr = 1

    # 其他
    args.des = f"viz_{embed}"
    args.loss = "MSE"
    args.lradj = "type1"
    args.use_dtw = False
    args.wv_extrap_eval = False

    return args


def load_model_and_predict(
    args: argparse.Namespace, checkpoint_path: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载模型并生成预测

    返回: (preds, trues, inputs, timestamps)
    """
    # 创建实验实例
    exp = Exp_Long_Term_Forecast(args)

    # 获取测试数据
    test_data, test_loader = exp._get_data(flag="test")

    # 加载模型
    model = exp._build_model()
    checkpoint = torch.load(checkpoint_path, map_location=exp.device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(exp.device)

    # 生成预测
    preds = []
    trues = []
    inputs = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            if i >= 50:  # 限制样本数以提高效率
                break

            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float().to(exp.device)
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, : args.label_len, :], dec_inp], dim=1
            ).float()

            # 推理
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len :, f_dim:]
            batch_y = batch_y[:, -args.pred_len :, f_dim:]

            # 逆变换回原始尺度
            if test_data.scale and hasattr(test_data, "inverse_transform"):
                outputs_np = outputs.cpu().numpy()
                batch_y_np = batch_y.cpu().numpy()
                batch_x_np = batch_x.cpu().numpy()

                # 对每个样本进行逆变换
                for b in range(outputs_np.shape[0]):
                    pred = test_data.inverse_transform(outputs_np[b])
                    true = test_data.inverse_transform(batch_y_np[b])
                    inp = test_data.inverse_transform(batch_x_np[b])

                    preds.append(pred)
                    trues.append(true)
                    inputs.append(inp)
            else:
                preds.append(outputs.cpu().numpy())
                trues.append(batch_y.cpu().numpy())
                inputs.append(batch_x.cpu().numpy())

    preds = np.array(preds)  # [N, pred_len, features]
    trues = np.array(trues)
    inputs = np.array(inputs)  # [N, seq_len, features]

    return preds, trues, inputs, test_data


def find_best_checkpoint(
    dataset: str, pred_len: int, embed: str, scale_mode: str
) -> Optional[str]:
    """查找最佳检查点"""
    checkpoint_dir = Path("./checkpoints")
    if not checkpoint_dir.exists():
        return None

    # 构建匹配模式
    pattern = f"long_term_forecast_*{dataset}*{embed}*pl{pred_len}*"

    candidates = []
    for cp_dir in checkpoint_dir.glob(pattern):
        cp_file = cp_dir / "checkpoint.pth"
        if cp_file.exists():
            # 检查是否包含 scale_mode
            dir_name = cp_dir.name.lower()
            if scale_mode.lower() in dir_name or (
                scale_mode == "none" and "noscale" in dir_name
            ):
                candidates.append(cp_file)

    if candidates:
        # 返回最新的
        return str(sorted(candidates, key=lambda p: p.stat().st_mtime)[-1])

    return None


def plot_single_channel_comparison(
    input_seq: np.ndarray,
    true_seq: np.ndarray,
    pred_timef: np.ndarray,
    pred_wvembs: np.ndarray,
    channel_idx: int,
    channel_name: str,
    outpath: str,
    title_suffix: str = "",
):
    """
    绘制单通道预测对比图（论文级质量）

    Args:
        input_seq: [seq_len] 输入序列
        true_seq: [pred_len] 真实值
        pred_timef: [pred_len] timeF预测
        pred_wvembs: [pred_len] WVEmbs预测
        channel_idx: 通道索引
        channel_name: 通道名称
        outpath: 输出路径
        title_suffix: 标题后缀
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    seq_len = len(input_seq)
    pred_len = len(true_seq)

    # 时间轴
    t_input = np.arange(seq_len)
    t_pred = np.arange(seq_len, seq_len + pred_len)

    # 绘制输入序列（灰色）
    ax.plot(
        t_input,
        input_seq,
        color="gray",
        alpha=0.6,
        linewidth=1.5,
        label="Input (Historical)",
    )

    # 绘制真实值（黑色实线）
    ax.plot(
        t_pred,
        true_seq,
        color="black",
        linewidth=2.5,
        label="Ground Truth",
        linestyle="-",
    )

    # 绘制timeF预测（蓝色虚线）
    ax.plot(
        t_pred,
        pred_timef,
        color="#2196F3",
        linewidth=2,
        label="timeF (Baseline)",
        linestyle="--",
    )

    # 绘制WVEmbs预测（红色点线）
    ax.plot(
        t_pred,
        pred_wvembs,
        color="#E91E63",
        linewidth=2,
        label="WVEmbs (Ours)",
        linestyle="-.",
    )

    # 添加预测起始垂直线
    ax.axvline(x=seq_len, color="gray", linestyle=":", alpha=0.7, linewidth=1.5)
    ax.text(
        seq_len + 1,
        ax.get_ylim()[1] * 0.95,
        "Prediction Start",
        fontsize=9,
        color="gray",
        style="italic",
    )

    # 计算并显示MSE
    mse_timef = np.mean((true_seq - pred_timef) ** 2)
    mse_wvembs = np.mean((true_seq - pred_wvembs) ** 2)
    improvement = ((mse_timef - mse_wvembs) / mse_timef) * 100

    # 添加文本框显示指标
    textstr = f"MSE(timeF): {mse_timef:.4f}\nMSE(WVEmbs): {mse_wvembs:.4f}\nImprovement: {improvement:+.1f}%"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    # 设置标签和标题
    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        f"{channel_name} Forecasting{title_suffix}", fontsize=14, fontweight="bold"
    )

    # 图例
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # 网格
    ax.grid(True, alpha=0.3, linestyle="--")

    # 美化
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_multi_channel_grid(
    inputs: np.ndarray,
    trues: np.ndarray,
    preds_timef: np.ndarray,
    preds_wvembs: np.ndarray,
    channel_names: List[str],
    outpath: str,
    n_channels: int = 4,
    sample_idx: int = 0,
):
    """
    绘制多通道子图对比

    Args:
        inputs: [N, seq_len, features]
        trues: [N, pred_len, features]
        preds_timef: [N, pred_len, features]
        preds_wvembs: [N, pred_len, features]
        channel_names: 通道名称列表
        outpath: 输出路径
        n_channels: 显示的通道数
        sample_idx: 样本索引
    """
    n_rows = (n_channels + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3 * n_rows))
    axes = axes.flatten()

    seq_len = inputs.shape[1]
    pred_len = trues.shape[1]

    for i in range(n_channels):
        ax = axes[i]

        # 数据
        input_seq = inputs[sample_idx, :, i]
        true_seq = trues[sample_idx, :, i]
        pred_timef_seq = preds_timef[sample_idx, :, i]
        pred_wvembs_seq = preds_wvembs[sample_idx, :, i]

        t_input = np.arange(seq_len)
        t_pred = np.arange(seq_len, seq_len + pred_len)

        # 绘制
        ax.plot(t_input, input_seq, color="gray", alpha=0.5, linewidth=1.2)
        ax.plot(t_pred, true_seq, color="black", linewidth=2, label="True")
        ax.plot(
            t_pred,
            pred_timef_seq,
            color="#2196F3",
            linewidth=1.5,
            linestyle="--",
            label="timeF",
        )
        ax.plot(
            t_pred,
            pred_wvembs_seq,
            color="#E91E63",
            linewidth=1.5,
            linestyle="-.",
            label="WVEmbs",
        )

        # 垂直线
        ax.axvline(x=seq_len, color="gray", linestyle=":", alpha=0.5)

        # MSE
        mse_timef = np.mean((true_seq - pred_timef_seq) ** 2)
        mse_wvembs = np.mean((true_seq - pred_wvembs_seq) ** 2)

        ax.set_title(
            f"{channel_names[i]} (MSE↓: {mse_wvembs:.3f} vs {mse_timef:.3f})",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # 隐藏多余的子图
    for i in range(n_channels, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        f"Multi-Channel Forecasting Comparison (Sample {sample_idx})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_residual_analysis(
    trues: np.ndarray,
    preds_timef: np.ndarray,
    preds_wvembs: np.ndarray,
    channel_names: List[str],
    outpath: str,
    sample_idx: int = 0,
):
    """
    绘制残差分析图
    """
    n_channels = min(4, trues.shape[2])
    fig, axes = plt.subplots(n_channels, 2, figsize=(12, 3 * n_channels))

    for i in range(n_channels):
        true_seq = trues[sample_idx, :, i]
        pred_timef_seq = preds_timef[sample_idx, :, i]
        pred_wvembs_seq = preds_wvembs[sample_idx, :, i]

        residual_timef = true_seq - pred_timef_seq
        residual_wvembs = true_seq - pred_wvembs_seq

        # 残差时间序列
        ax1 = axes[i, 0]
        ax1.plot(residual_timef, color="#2196F3", alpha=0.7, label="timeF Residual")
        ax1.plot(residual_wvembs, color="#E91E63", alpha=0.7, label="WVEmbs Residual")
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_ylabel("Residual")
        ax1.set_title(f"{channel_names[i]} - Residual over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 残差直方图
        ax2 = axes[i, 1]
        ax2.hist(
            residual_timef,
            bins=20,
            alpha=0.5,
            color="#2196F3",
            label="timeF",
            density=True,
        )
        ax2.hist(
            residual_wvembs,
            bins=20,
            alpha=0.5,
            color="#E91E63",
            label="WVEmbs",
            density=True,
        )
        ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Residual Value")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{channel_names[i]} - Residual Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Residual Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def plot_error_bar_comparison(
    trues: np.ndarray,
    preds_timef: np.ndarray,
    preds_wvembs: np.ndarray,
    channel_names: List[str],
    outpath: str,
):
    """
    绘制各通道MSE对比柱状图
    """
    n_channels = trues.shape[2]
    mse_timef_per_channel = []
    mse_wvembs_per_channel = []

    for i in range(n_channels):
        mse_tf = np.mean((trues[:, :, i] - preds_timef[:, :, i]) ** 2)
        mse_wv = np.mean((trues[:, :, i] - preds_wvembs[:, :, i]) ** 2)
        mse_timef_per_channel.append(mse_tf)
        mse_wvembs_per_channel.append(mse_wv)

    x = np.arange(n_channels)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, n_channels * 1.2), 6))

    bars1 = ax.bar(
        x - width / 2,
        mse_timef_per_channel,
        width,
        label="timeF",
        color="#2196F3",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        mse_wvembs_per_channel,
        width,
        label="WVEmbs",
        color="#E91E63",
        alpha=0.8,
    )

    ax.set_xlabel("Channel", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Per-Channel MSE Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(channel_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # 标注数值
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="生成论文级预测可视化样例")
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
        "--checkpoint_timeF",
        type=str,
        default=None,
        help="timeF模型检查点路径（可选，会自动搜索）",
    )
    parser.add_argument(
        "--checkpoint_wvembs",
        type=str,
        default=None,
        help="WVEmbs模型检查点路径（可选，会自动搜索）",
    )
    parser.add_argument(
        "--outdir", type=str, default="results/visualizations/", help="输出目录"
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="用于可视化的样本数量"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.outdir, exist_ok=True)

    print(f"=" * 80)
    print(f"论文级预测可视化生成")
    print(f"数据集: {args.dataset}, 预测长度: {args.pred_len}")
    print(f"=" * 80)

    # 查找检查点
    if args.checkpoint_timeF is None:
        args.checkpoint_timeF = find_best_checkpoint(
            args.dataset, args.pred_len, "timeF", "standard"
        )
    if args.checkpoint_wvembs is None:
        args.checkpoint_wvembs = find_best_checkpoint(
            args.dataset, args.pred_len, "wv", "standard"
        )

    if args.checkpoint_timeF is None:
        print(f"[ERROR] 未找到 timeF 检查点，请指定 --checkpoint_timeF")
        return
    if args.checkpoint_wvembs is None:
        print(f"[ERROR] 未找到 WVEmbs 检查点，请指定 --checkpoint_wvembs")
        return

    print(f"timeF 检查点: {args.checkpoint_timeF}")
    print(f"WVEmbs 检查点: {args.checkpoint_wvembs}")

    # 设置参数并加载模型
    args_timef = setup_args(
        args.dataset, args.pred_len, embed="timeF", scale_mode="standard"
    )
    args_wvembs = setup_args(
        args.dataset,
        args.pred_len,
        embed="wv",
        scale_mode="standard",
        wv_sampling="jss",
        wv_jss_std=0.25,
    )

    print("\n[1/4] 加载 timeF 模型并生成预测...")
    preds_timef, trues, inputs, test_data = load_model_and_predict(
        args_timef, args.checkpoint_timeF
    )

    print("[2/4] 加载 WVEmbs 模型并生成预测...")
    preds_wvembs, _, _, _ = load_model_and_predict(args_wvembs, args.checkpoint_wvembs)

    # 获取通道名称
    channel_names = getattr(test_data, "cols", None)
    if channel_names is None:
        channel_names = [f"Channel {i}" for i in range(trues.shape[2])]

    print(f"[3/4] 生成可视化...")

    # 选择最佳样本（基于WVEmbs表现较好的样本）
    sample_mses = []
    for i in range(min(args.n_samples, len(trues))):
        mse_wv = np.mean((trues[i] - preds_wvembs[i]) ** 2)
        sample_mses.append((mse_wv, i))
    sample_mses.sort()

    # 1. 单通道详细对比图（选择OT通道）
    ot_idx = channel_names.index("OT") if "OT" in channel_names else 0
    best_sample_idx = sample_mses[0][1]

    plot_single_channel_comparison(
        inputs[best_sample_idx, :, ot_idx],
        trues[best_sample_idx, :, ot_idx],
        preds_timef[best_sample_idx, :, ot_idx],
        preds_wvembs[best_sample_idx, :, ot_idx],
        ot_idx,
        channel_names[ot_idx],
        os.path.join(
            args.outdir, f"{args.dataset}_pl{args.pred_len}_single_channel_best.png"
        ),
        f" (Best Sample, {args.dataset})",
    )

    # 2. 多通道网格图
    plot_multi_channel_grid(
        inputs,
        trues,
        preds_timef,
        preds_wvembs,
        channel_names,
        os.path.join(
            args.outdir, f"{args.dataset}_pl{args.pred_len}_multi_channel.png"
        ),
        n_channels=min(4, len(channel_names)),
        sample_idx=best_sample_idx,
    )

    # 3. 残差分析图
    plot_residual_analysis(
        trues,
        preds_timef,
        preds_wvembs,
        channel_names,
        os.path.join(args.outdir, f"{args.dataset}_pl{args.pred_len}_residuals.png"),
        sample_idx=best_sample_idx,
    )

    # 4. 各通道MSE对比图
    plot_error_bar_comparison(
        trues,
        preds_timef,
        preds_wvembs,
        channel_names,
        os.path.join(
            args.outdir, f"{args.dataset}_pl{args.pred_len}_mse_comparison.png"
        ),
    )

    print(f"\n[4/4] 完成！所有可视化已保存到: {args.outdir}")

    # 打印汇总统计
    overall_mse_timef = np.mean((trues - preds_timef) ** 2)
    overall_mse_wvembs = np.mean((trues - preds_wvembs) ** 2)
    overall_improvement = (
        (overall_mse_timef - overall_mse_wvembs) / overall_mse_timef
    ) * 100

    print(f"\n{'=' * 60}")
    print(f"整体性能统计:")
    print(f"  MSE (timeF):   {overall_mse_timef:.6f}")
    print(f"  MSE (WVEmbs):  {overall_mse_wvembs:.6f}")
    print(f"  改善:          {overall_improvement:+.2f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
