"""
从 Hugging Face 数据集仓库下载并“落盘”到本仓库的 `./dataset/` 目录。

为什么需要这个脚本？
- 本仓库的数据加载器支持 `hf_hub_download` 自动下载，但默认读取的是 HF 缓存路径；
- 论文/实验复现时，更希望关键数据文件真实存在于 `./dataset/...`，便于打包/迁移/离线运行；
- Cycle 0 需要补齐 ETTh2/ETTm2、Weather，以及若干 UEA 分类数据集的文件。

使用示例
- 下载 Cycle 0 需要的全部数据：
  conda run -n radio python scripts/wvembs/download_datasets.py --all
- 只下载 ETT：
  conda run -n radio python scripts/wvembs/download_datasets.py --ett
- 只下载 UEA 中指定数据集：
  conda run -n radio python scripts/wvembs/download_datasets.py --uea EthanolConcentration FaceDetection
"""

from __future__ import annotations

import argparse
import os
import shutil

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


HUGGINGFACE_REPO = "thuml/Time-Series-Library"


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _download_and_copy(rel_path_in_repo: str, dst_path: str, *, force: bool, strict: bool) -> bool:
    """
    下载 `rel_path_in_repo`（相对 HF dataset repo 根目录），并拷贝到 `dst_path`。
    """
    if (not force) and os.path.exists(dst_path):
        print(f"[SKIP] 已存在：{dst_path}")
        return True

    try:
        cached_fp = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename=rel_path_in_repo, repo_type="dataset")
    except EntryNotFoundError as e:
        if strict:
            raise
        print(f"[WARN] HF 仓库缺失：{rel_path_in_repo}（跳过；如需强制失败请加 --strict）")
        return False
    _ensure_parent_dir(dst_path)
    shutil.copyfile(cached_fp, dst_path)
    print(f"[OK] {rel_path_in_repo} -> {dst_path}")
    return True


def _download_ett(*, force: bool, strict: bool) -> None:
    # 仅补齐 Cycle 0 中缺失的两个文件
    items = [
        ("ETT-small/ETTh2.csv", "dataset/ETT-small/ETTh2.csv"),
        ("ETT-small/ETTm2.csv", "dataset/ETT-small/ETTm2.csv"),
    ]
    for rel, dst in items:
        _download_and_copy(rel, dst, force=force, strict=strict)


def _download_weather(*, force: bool, strict: bool) -> None:
    _download_and_copy("weather/weather.csv", "dataset/weather/weather.csv", force=force, strict=strict)


def _download_uea(datasets: list[str], *, force: bool, strict: bool) -> None:
    if not datasets:
        raise ValueError("UEA 数据集列表为空；请在 --uea 后面给出 1 个或多个数据集名。")

    for name in datasets:
        for split in ["TRAIN", "TEST"]:
            rel = f"{name}/{name}_{split}.ts"
            dst = f"dataset/{name}/{name}_{split}.ts"
            _download_and_copy(rel, dst, force=force, strict=strict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="覆盖已存在的本地文件")
    parser.add_argument("--strict", action="store_true", help="下载失败时直接报错退出（默认跳过缺失文件）")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="下载 Cycle 0 所需的全部数据")
    group.add_argument("--ett", action="store_true", help="下载/补齐 ETTh2、ETTm2")
    group.add_argument("--weather", action="store_true", help="下载/补齐 Weather")
    group.add_argument("--uea", nargs="+", help="下载 1 个或多个 UEA 数据集（给出名称列表）")

    args = parser.parse_args()

    if args.all:
        _download_ett(force=args.force, strict=args.strict)
        _download_weather(force=args.force, strict=args.strict)
        _download_uea(
            ["EthanolConcentration", "FaceDetection", "HandMovementDirection"],
            force=args.force,
            strict=args.strict,
        )
        return

    if args.ett:
        _download_ett(force=args.force, strict=args.strict)
        return

    if args.weather:
        _download_weather(force=args.force, strict=args.strict)
        return

    if args.uea is not None:
        _download_uea(list(args.uea), force=args.force, strict=args.strict)
        return

    raise RuntimeError("未命中任何下载分支；请检查参数。")


if __name__ == "__main__":
    main()
