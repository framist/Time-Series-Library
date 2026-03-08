def parse_embed_arg(embed: str):
    """
    解析统一的 `--embed` 参数。

    兼容原仓库含义：
    - `timeF` / `fixed` / `learned`：原始 TSLib 路径，值嵌入为 TokenEmbedding。

    为 WVEmbs 实验扩展：
    - `wv`：统一模式（论文核心配置）：value 与 time 特征统一视为“物理量通道”，在 embedding 内部做通道拼接后一起进入 WV-Lift
    - `wv_<time_embed>`：启用 WVEmbs 作为 value embedding，同时保持时间特征编码方式为 `<time_embed>`。
    - `linear`：统一线性模式；与 `wv` 一样将 value 与 `timeF` 时间特征拼接，但前端改为线性投影，用于“只替换输入层”的公平对照
    - `linear_<time_embed>`：值用线性投影，时间特征仍走 `<time_embed>` 对应的原始时间嵌入

    返回
    ----
    (time_embed_type, value_embed_type)
    - `time_embed_type`：通常为 `timeF/fixed/learned`
    - `value_embed_type`：`token`（原 TokenEmbedding）、`linear`（线性值嵌入）或 `wv`（WVEmbs + WV-Lift）
    """
    if embed is None:
        return "timeF", "token"

    if not isinstance(embed, str):
        embed = str(embed)

    embed = embed.strip()
    if embed == "":
        return "timeF", "token"

    if embed == "wv":
        # 注意：统一模式下，时间特征仍建议走 timeF（连续浮点），以便作为物理量通道输入 WVEmbs
        return "timeF", "wv"

    if embed == "linear":
        # 统一线性模式与 wv 对齐：把 timeF 作为连续通道直接拼到输入层
        return "timeF", "linear"

    if embed.startswith("wv_"):
        time_embed = embed[len("wv_") :].strip()
        return (time_embed or "timeF"), "wv"

    if embed.startswith("linear_"):
        time_embed = embed[len("linear_") :].strip()
        return (time_embed or "timeF"), "linear"

    return embed, "token"


def is_unified_embed(embed: str) -> bool:
    """
    判断是否启用“统一输入层”模式。

    约定：
    - `--embed wv`：值与 `timeF` 时间特征拼接后送入 WVEmbs
    - `--embed linear`：值与 `timeF` 时间特征拼接后送入线性投影
    """
    if embed is None:
        return False
    return str(embed).strip() in {"wv", "linear"}


def is_wv_unified(embed: str) -> bool:
    """
    判断是否启用 WVEmbs 的“统一时间嵌入”模式。

    约定：
    - `--embed wv` 表示统一模式
    - `--embed wv_timeF|wv_fixed|wv_learned` 表示“值用 WVEmbs、时间仍用传统嵌入”的消融对照
    """
    if embed is None:
        return False
    return str(embed).strip() == "wv"
