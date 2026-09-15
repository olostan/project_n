"""
Project N: Sensory Feature Standardizer.
Computes and applies per-feature z-score standardization across sensory dimensions.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx


class FeatureStandardizer:
    """
    Standardizes feature arrays by subtracting the mean and dividing by std.
    Fitted strictly on training split and serialized alongside model checkpoints.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean: mx.array | None = None
        self.std: mx.array | None = None

    def fit(self, x: mx.array) -> FeatureStandardizer:
        """
        Computes mean and standard deviation over all axes except the last (feature) dimension.

        Args:
            x: Input array of shape (N, D) or (N, ..., D).

        Returns:
            Self instance with fitted statistics.
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x_flat = x.reshape(-1, x.shape[-1])
        mean = mx.mean(x_flat, axis=0)
        var = mx.var(x_flat, axis=0)
        std = mx.sqrt(var + self.eps)
        mx.eval(mean, std)
        self.mean = mean
        self.std = std
        return self

    def transform(self, x: mx.array) -> mx.array:
        """
        Transforms input array using fitted mean and standard deviation.

        Args:
            x: Input array with matching feature dimension D.

        Returns:
            Standardized array.
        """
        if self.mean is None or self.std is None:
            raise ValueError("FeatureStandardizer must be fitted before transforming.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: mx.array) -> mx.array:
        """Fits standardizer and returns transformed array."""
        return self.fit(x).transform(x)

    def save(self, path: str | Path) -> None:
        """Serializes mean and std statistics to npz file."""
        if self.mean is None or self.std is None:
            raise ValueError("Cannot save unfitted FeatureStandardizer.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        mx.savez(str(p), mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path, eps: float = 1e-8) -> FeatureStandardizer:
        """Loads fitted FeatureStandardizer from npz file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Standardizer file not found: {path}")
        data = mx.load(str(p))
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict from {path}, got {type(data)}")
        instance = cls(eps=eps)
        instance.mean = data["mean"]
        instance.std = data["std"]
        return instance
