"""
Project N: Metal Attention Pooling Module.
Learned temporal attention pooling condensing arbitrary sequence lengths into
a fixed-dimensional latent representation.
"""

import mlx.core as mx
import mlx.nn as nn


class AttentionPool(nn.Module):
    """
    Learned temporal attention pooling condensing arbitrary sequence lengths
    into a unified latent summary representation.

    Unified Mask Polarity:
    - Boolean mask: True = valid observation, False = masked/ignored.
    - Floating mask: 0.0 = valid, 1.0 = masked/ignored (for additive bias).
    """

    def __init__(self, d_in: int) -> None:
        super().__init__()
        self.d_in = d_in
        self.attn_vector = nn.Linear(d_in, 1, bias=False)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        """
        Pool sequence x of shape (B, T, D) -> (B, D).

        Args:
            x: Input tensor of shape (B, T, D).
            mask: Optional mask of shape (B, T) or (B, T, 1).
                  If boolean: True is valid, False is masked.
                  If numeric: 1 is masked, 0 is valid.

        Returns:
            Pooled tensor of shape (B, D).
        """
        # x: (B, T, D)
        scores = self.attn_vector(x)  # (B, T, 1)

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, :, None]  # Expand (B, T) -> (B, T, 1)

            if mask.dtype == mx.bool_:
                # True = valid, False = masked -> apply -1e9 where False
                scores = mx.where(mask, scores, mx.array(-1e9, dtype=scores.dtype))
            else:
                # Numeric: 1 = masked, 0 = valid
                scores = scores + (mask * -1e9)

        weights = mx.softmax(scores, axis=1)  # (B, T, 1)
        return mx.sum(x * weights, axis=1)
