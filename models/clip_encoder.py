"""
Project N: Clip-Level Sequence Attention Pooling.
Hierarchical temporal sequence pooling condensing multi-window sequences (30-120s)
into a canonical clip representation while strictly decoupling positional magnitude.
"""

import math

import mlx.core as mx
import mlx.nn as nn

from models.contracts import METRIC_EMBEDDING_D


def generate_sinusoidal_positional_encoding(length: int, d_model: int) -> mx.array:
    """
    Generates standard sinusoidal positional encodings of shape (length, d_model).

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    position = mx.arange(0, length, dtype=mx.float32)[:, None]
    div_term = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32) * (-math.log(10000.0) / d_model))[
        None, :
    ]

    sin_term = mx.sin(position * div_term)
    cos_term = mx.cos(position * div_term)

    # Interleave sin and cos
    pe = mx.zeros((length, d_model), dtype=mx.float32)
    pe = mx.concatenate([sin_term[:, :, None], cos_term[:, :, None]], axis=-1)
    pe = mx.reshape(pe, (length, d_model))
    return pe


class ClipSequenceAttentionPool(nn.Module):
    """
    Macro-trajectory sequence attention pooling combining W window vectors z^(w)
    into a clip-level embedding z_clip.

    IMPORTANT (Positional Encoding Magnitude Decoupling Invariant):
    Window vectors z^(w) are L2-normalized (||z^(w)|| = 1). Standard sinusoidal
    positional encodings have norm ||p_w|| = sqrt(128/2) = 8. Adding p_w directly
    to the pooled vectors would cause the positional component to overpower the
    sensory content by 8:1. Therefore, p_w is used STRICTLY for computing attention
    weights alpha_w, while the value vectors being pooled are the pure sensory
    representations z^(w):
        alpha_w = Softmax( (z^(w) + p_w) * q_clip / sqrt(128) )
        z_clip = L2Norm( sum_w alpha_w * z^(w) )
    """

    def __init__(self, d_metric: int = METRIC_EMBEDDING_D) -> None:
        super().__init__()
        self.d_metric = d_metric
        # Learned clip query vector
        self.query = mx.random.normal((d_metric,)) * 0.02
        self.scale = 1.0 / math.sqrt(d_metric)

    def __call__(self, z_windows: mx.array, p_encodings: mx.array | None = None) -> mx.array:
        """
        Pool sequence of window embeddings into clip-level representation.

        Args:
            z_windows: Window embeddings of shape (B, W, 128), L2-normalized.
            p_encodings: Optional sinusoidal positional encodings of shape (W, 128).
                         If None, generated dynamically based on W.

        Returns:
            z_clip: L2-normalized clip embedding of shape (B, 128).
        """
        # z_windows: (B, W, 128)
        w_count = z_windows.shape[1]

        if p_encodings is None:
            p_encodings = generate_sinusoidal_positional_encoding(w_count, self.d_metric)

        # Use (z + p) for attention key/scoring only:
        keys = z_windows + p_encodings[None, :, :]  # (B, W, 128)
        scores = (
            mx.sum(keys * self.query[None, None, :], axis=-1, keepdims=True) * self.scale
        )  # (B, W, 1)
        weights = mx.softmax(scores, axis=1)  # (B, W, 1)

        # Pool pure sensory window representations z^(w) (WITHOUT p_w):
        pooled = mx.sum(z_windows * weights, axis=1)  # (B, 128)
        norm = mx.sqrt(mx.sum(mx.square(pooled), axis=-1, keepdims=True) + 1e-8)
        return pooled / norm
