#!/usr/bin/env python3
"""
汇总无预处理公平对照实验结果。

读取：
- `results/<setting>/metrics.npy`
- `results/<setting>/result_classification.txt`
- 根目录下的 `result_anomaly_detection.txt`

输出：
- 终端 Markdown 表格
- 可选保存到文件
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VARIANT_ORDER = {"raw_timeF": 0, "timeF": 0, "linear": 1, "wv": 2}


def _parse_forecast_setting(name: str) -> Dict[str, str]:
    m = re.match(
        r"long_term_forecast_NoPrepFair_(?P<dataset>[^_]+)_Transformer_(?P<variant>raw_timeF|linear|wv)_pl(?P<pred_len>\d+)_",
        name,
    )
    if not m:
        raise ValueError(f"无法解析 forecast setting: {name}")
    return m.groupdict()


def _parse_imputation_setting(name: str) -> Dict[str, str]:
    m = re.match(
        r"imputation_NoPrepFair_(?P<dataset>[^_]+)_[^_]+_(?P<variant>raw_timeF|linear|wv)_",
        name,
    )
    if not m:
        raise ValueError(f"无法解析 imputation setting: {name}")
    return m.groupdict()


def _parse_anomaly_block(text: str) -> List[Dict[str, str]]:
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    rows = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        setting = lines[0]
        metrics = lines[1]
        m = re.search(
            r"Accuracy : (?P<acc>[\d.]+), Precision : (?P<prec>[\d.]+), Recall : (?P<rec>[\d.]+), F-score : (?P<f1>[\d.]+)",
            metrics,
        )
        if not m:
            continue
        s = re.match(
            r"anomaly_detection_NoPrepFair_(?P<dataset>[^_]+)_[^_]+_(?P<variant>raw_timeF|linear|wv)_",
            setting,
        )
        if not s:
            continue
        row = s.groupdict()
        row.update(m.groupdict())
        row["setting"] = setting
        rows.append(row)
    return rows


def _variant_label(variant: str) -> str:
    return {
        "raw_timeF": "原始时间特征输入层",
        "timeF": "原始时间特征输入层",
        "linear": "线性统一输入层",
        "wv": "WVEmbs 统一输入层",
    }.get(variant, variant)


def _variant_rank(variant: str) -> int:
    return VARIANT_ORDER.get(variant, 99)


def collect_forecast(des: str) -> List[Dict[str, str]]:
    rows = []
    for metrics_path in sorted(ROOT.glob(f"results/long_term_forecast_*{des}*/metrics.npy")):
        setting = metrics_path.parent.name
        meta = _parse_forecast_setting(setting)
        mae, mse, *_ = np.load(metrics_path)
        rows.append(
            {
                "dataset": meta["dataset"],
                "pred_len": meta["pred_len"],
                "variant": meta["variant"],
                "variant_label": _variant_label(meta["variant"]),
                "mae": f"{float(mae):.6f}",
                "mse": f"{float(mse):.6f}",
                "setting": setting,
            }
        )
    return rows


def collect_imputation(des: str) -> List[Dict[str, str]]:
    rows = []
    for metrics_path in sorted(ROOT.glob(f"results/imputation_*{des}*/metrics.npy")):
        setting = metrics_path.parent.name
        meta = _parse_imputation_setting(setting)
        mae, mse, *_ = np.load(metrics_path)
        rows.append(
            {
                "dataset": meta["dataset"],
                "variant": meta["variant"],
                "variant_label": _variant_label(meta["variant"]),
                "mae": f"{float(mae):.6f}",
                "mse": f"{float(mse):.6f}",
                "setting": setting,
            }
        )
    return rows


def collect_classification(des: str) -> List[Dict[str, str]]:
    rows = []
    for result_path in sorted(ROOT.glob(f"results/classification_*{des}*/result_classification.txt")):
        setting = result_path.parent.name
        m = re.match(r"classification_(?P<dataset>[^_]+)_[^_]+_[^_]+_.*_(?P<variant>raw_timeF|linear|wv)_[^_]+$", setting)
        if not m:
            continue
        text = result_path.read_text()
        acc = re.search(r"accuracy:([\d.]+)", text)
        if not acc:
            continue
        rows.append(
            {
                "dataset": m.group("dataset"),
                "variant": m.group("variant"),
                "variant_label": _variant_label(m.group("variant")),
                "accuracy": f"{float(acc.group(1)):.6f}",
                "setting": setting,
            }
        )
    return rows


def collect_anomaly(des: str) -> List[Dict[str, str]]:
    txt = ROOT / "result_anomaly_detection.txt"
    if not txt.exists():
        return []
    rows = []
    for row in _parse_anomaly_block(txt.read_text()):
        if des not in row["setting"]:
            continue
        row["variant_label"] = _variant_label(row["variant"])
        rows.append(row)
    return rows


def markdown_table(headers: List[str], rows: List[Dict[str, str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _format_delta(current: float, baseline: float, lower_is_better: bool = True) -> str:
    if not math.isfinite(current) or not math.isfinite(baseline):
        return "NaN/Inf"
    if baseline == 0:
        return ""
    ratio = (current - baseline) / baseline * 100.0
    if lower_is_better:
        return f"{ratio:+.1f}%"
    return f"{-ratio:+.1f}%"


def attach_delta(
    rows: List[Dict[str, str]],
    group_keys: List[str],
    metric_key: str,
    out_key: str,
    lower_is_better: bool = True,
    baseline_variants: Sequence[str] | None = None,
) -> None:
    grouped: Dict[tuple, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped[key][row["variant"]] = row

    if baseline_variants is None:
        baseline_variants = ("raw_timeF", "timeF")

    for variants in grouped.values():
        baseline = None
        for variant_name in baseline_variants:
            baseline = variants.get(variant_name)
            if baseline is not None:
                break
        if baseline is None:
            continue
        baseline_value = float(baseline[metric_key])
        for row in variants.values():
            row[out_key] = _format_delta(
                float(row[metric_key]),
                baseline_value,
                lower_is_better=lower_is_better,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总无预处理公平对照结果")
    parser.add_argument("--des", required=True, help="实验描述字段，如 NoPrepFairFull_20260309")
    parser.add_argument("--out", type=str, default="", help="可选输出 Markdown 文件")
    args = parser.parse_args()

    sections = []

    forecast_rows = collect_forecast(args.des)
    if forecast_rows:
        forecast_rows.sort(key=lambda x: (x["dataset"], int(x["pred_len"]), _variant_rank(x["variant"])))
        attach_delta(forecast_rows, ["dataset", "pred_len"], "mse", "vs_raw_timeF")
        attach_delta(forecast_rows, ["dataset", "pred_len"], "mse", "vs_linear", baseline_variants=("linear",))
        sections.append(
            "## Forecast\n"
            + markdown_table(
                ["dataset", "pred_len", "variant_label", "mse", "mae", "vs_raw_timeF", "vs_linear"],
                forecast_rows,
            )
        )

    imputation_rows = collect_imputation(args.des)
    if imputation_rows:
        imputation_rows.sort(key=lambda x: (x["dataset"], _variant_rank(x["variant"])))
        attach_delta(imputation_rows, ["dataset"], "mse", "vs_raw_timeF")
        attach_delta(imputation_rows, ["dataset"], "mse", "vs_linear", baseline_variants=("linear",))
        sections.append(
            "## Imputation\n"
            + markdown_table(["dataset", "variant_label", "mse", "mae", "vs_raw_timeF", "vs_linear"], imputation_rows)
        )

    anomaly_rows = collect_anomaly(args.des)
    if anomaly_rows:
        anomaly_rows.sort(key=lambda x: (x["dataset"], _variant_rank(x["variant"])))
        attach_delta(anomaly_rows, ["dataset"], "f1", "vs_raw_timeF", lower_is_better=False)
        attach_delta(
            anomaly_rows,
            ["dataset"],
            "f1",
            "vs_linear",
            lower_is_better=False,
            baseline_variants=("linear",),
        )
        sections.append(
            "## Anomaly\n"
            + markdown_table(
                ["dataset", "variant_label", "acc", "prec", "rec", "f1", "vs_raw_timeF", "vs_linear"],
                anomaly_rows,
            )
        )

    classification_rows = collect_classification(args.des)
    if classification_rows:
        classification_rows.sort(key=lambda x: (x["dataset"], _variant_rank(x["variant"])))
        attach_delta(classification_rows, ["dataset"], "accuracy", "vs_raw_timeF", lower_is_better=False)
        attach_delta(
            classification_rows,
            ["dataset"],
            "accuracy",
            "vs_linear",
            lower_is_better=False,
            baseline_variants=("linear",),
        )
        sections.append(
            "## Classification\n"
            + markdown_table(["dataset", "variant_label", "accuracy", "vs_raw_timeF", "vs_linear"], classification_rows)
        )

    output = "\n\n".join(sections) if sections else "未找到匹配结果。"
    print(output)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output)


if __name__ == "__main__":
    main()
