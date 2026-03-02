def parse_embed_arg(embed: str):
    """
    解析统一的 `--embed` 参数。

    兼容原仓库含义：
    - `timeF` / `fixed` / `learned`：仅控制时间特征编码方式 + `timeenc`（数据侧时间特征生成）。

    为 WVEmbs 实验扩展：
    - `wv`：等价于 `wv_timeF`
    - `wv_<time_embed>`：启用 WVEmbs 作为 value embedding，同时保持时间特征编码方式为 `<time_embed>`。

    返回
    ----
    (time_embed_type, value_embed_type)
    - `time_embed_type`：通常为 `timeF/fixed/learned`
    - `value_embed_type`：`token`（原 TokenEmbedding）或 `wv`（WVEmbs + WV-Lift）
    """
    if embed is None:
        return "timeF", "token"

    if not isinstance(embed, str):
        embed = str(embed)

    embed = embed.strip()
    if embed == "":
        return "timeF", "token"

    if embed == "wv":
        return "timeF", "wv"

    if embed.startswith("wv_"):
        time_embed = embed[len("wv_") :].strip()
        return (time_embed or "timeF"), "wv"

    return embed, "token"
