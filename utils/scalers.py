import numpy as np
from sklearn.preprocessing import StandardScaler


class IdentityScaler:
    """不做任何缩放；用于 `scale_mode=none`。"""

    def fit(self, data):
        return self

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class PriorScaler:
    """
    物理先验尺度无量纲化（不依赖训练集统计量）。

    约定：
    - 输入/输出均为 numpy 数组
    - 变换：x' = (x - offset) / scale
    - 逆变换：x = x' * scale + offset

    注意：这里的 scale/offset 应由“物理量先验”给定（可为标量或每通道一个值）。
    """

    def __init__(self, scale, offset=None, eps: float = 1e-12):
        self.eps = float(eps)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.offset = np.asarray(0.0 if offset is None else offset, dtype=np.float32)

    @staticmethod
    def _broadcast_param(param: np.ndarray, feature_dim: int, name: str) -> np.ndarray:
        if param.ndim == 0:
            return np.full((feature_dim,), float(param), dtype=np.float32)
        if param.ndim != 1:
            raise ValueError(f"{name} 只能是标量或 1D 向量，但得到 shape={tuple(param.shape)}")
        if int(param.shape[0]) == 1:
            return np.full((feature_dim,), float(param[0]), dtype=np.float32)
        if int(param.shape[0]) != int(feature_dim):
            raise ValueError(f"{name} 维度不匹配：期望 {feature_dim}，但得到 {int(param.shape[0])}")
        return param.astype(np.float32, copy=False)

    @classmethod
    def from_feature_dim(cls, feature_dim: int, scale, offset=None) -> "PriorScaler":
        scaler = cls(scale=scale, offset=offset)
        scaler.scale = cls._broadcast_param(scaler.scale, feature_dim, "prior_scale")
        scaler.offset = cls._broadcast_param(scaler.offset, feature_dim, "prior_offset")
        if np.any(np.abs(scaler.scale) <= scaler.eps):
            raise ValueError("prior_scale 不能为 0（或过小）")
        return scaler

    def fit(self, data):
        return self

    def transform(self, data):
        return (data - self.offset) / (self.scale + self.eps)

    def inverse_transform(self, data):
        return data * (self.scale + self.eps) + self.offset


def build_series_scaler(
    scale_mode: str,
    train_data: np.ndarray,
    feature_dim: int,
    prior_scale=None,
    prior_offset=None,
):
    """
    构建并返回一个具有 `transform / inverse_transform` 接口的 scaler。

    参数
    - scale_mode: standard|prior|none
    - train_data: 用于 standard 模式拟合的训练集数据（形状 [N, M]）
    - feature_dim: 通道数 M（用于 prior 参数广播与校验）
    """
    mode = str(scale_mode or "standard").strip().lower()
    if mode == "standard":
        scaler = StandardScaler()
        scaler.fit(train_data)
        return scaler
    if mode == "none":
        return IdentityScaler()
    if mode == "prior":
        if prior_scale is None:
            raise ValueError("scale_mode=prior 需要提供 --prior_scale（标量或每通道一个值）")
        return PriorScaler.from_feature_dim(feature_dim, scale=prior_scale, offset=prior_offset)
    raise ValueError(f"未知 scale_mode={scale_mode!r}，可选 standard/prior/none")

