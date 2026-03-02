def parse_embed_arg(embed: str):
    """
    Parse the unified `--embed` argument.

    Legacy (upstream) meanings:
      - timeF / fixed / learned: only controls temporal embedding + timeenc.

    Extensions for WVEmbs experiments in this repo:
      - wv: alias of wv_timeF
      - wv_<time_embed>: enable WVEmbs value embedding, while keeping temporal embedding type.

    Returns
    -------
    (time_embed_type, value_embed_type)
      - time_embed_type: str, typically one of {timeF, fixed, learned}
      - value_embed_type: str, one of {token, wv}
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

