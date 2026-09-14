"""
Project N: Multimodal Metric Projection Head.
Maps acoustic, kinematic, and optional physiological inputs into an L2-normalized
128-dimensional metric space with graceful degradation for missing modalities.
"""

from typing import cast

import mlx.core as mx
import mlx.nn as nn

from models.attention import AttentionPool
from models.contracts import (
    ACOUSTIC_LATENT_D,
    KINEMATIC_LATENT_D,
    METRIC_EMBEDDING_D,
    PHYSIOLOGY_LATENT_D,
)


class MetricProjectionHead(nn.Module):
    """
    Multimodal projection head supporting graceful degradation when
    optional physiological sensors are absent. Maps inputs into an
    L2-normalized 128-dimensional metric space.
    """

    def __init__(
        self,
        d_audio: int = ACOUSTIC_LATENT_D,
        d_kinematic: int = KINEMATIC_LATENT_D,
        d_physio: int = PHYSIOLOGY_LATENT_D,
        d_hidden: int = 512,
        d_metric: int = METRIC_EMBEDDING_D,
    ) -> None:
        super().__init__()
        self.d_audio = d_audio
        self.d_kinematic = d_kinematic
        self.d_physio = d_physio
        self.d_hidden = d_hidden
        self.d_metric = d_metric

        # Modal attention pools
        self.pool_audio = AttentionPool(d_audio)
        self.pool_kinematic = AttentionPool(d_kinematic)
        self.pool_physio = AttentionPool(d_physio)

        # Dimension alignment projections
        self.proj_audio = nn.Linear(d_audio, d_hidden)
        self.proj_kinematic = nn.Linear(d_kinematic, d_hidden)
        self.proj_physio = nn.Linear(d_physio, d_hidden)

        # Learned null-modality embedding for missing physiological telemetry
        self.null_physio = mx.random.normal((1, 1, d_physio)) * 0.02

        # Multimodal fusion layers
        self.fusion_norm = nn.LayerNorm(d_hidden * 3)
        self.fc1 = nn.Linear(d_hidden * 3, d_hidden)
        self.fc_metric = nn.Linear(d_hidden, d_metric, bias=False)

    def __call__(
        self,
        x_audio: mx.array,
        x_kinematic: mx.array,
        x_physio: mx.array | None = None,
        mask_audio: mx.array | None = None,
        mask_kinematic: mx.array | None = None,
        mask_physio: mx.array | None = None,
    ) -> mx.array:
        """
        Forward pass mapping multimodal features to L2-normalized metric embedding.

        Args:
            x_audio: Acoustic features of shape (B, T_a, 512).
            x_kinematic: Kinematic features of shape (B, T_k, 512).
            x_physio: Optional physiological features of shape (B, T_p, 64).
            mask_audio: Optional attention mask for audio.
            mask_kinematic: Optional attention mask for kinematics.
            mask_physio: Optional attention mask for physiology.

        Returns:
            z_metric: L2-normalized metric embeddings of shape (B, 128).
        """
        batch_size = x_audio.shape[0]

        # 1. Pool and project acoustic and kinematic features
        h_a = nn.gelu(self.proj_audio(self.pool_audio(x_audio, mask=mask_audio)))
        h_k = nn.gelu(self.proj_kinematic(self.pool_kinematic(x_kinematic, mask=mask_kinematic)))

        # 2. Graceful degradation: substitute null embedding if physiology is missing
        if x_physio is None:
            broadcast_null = mx.broadcast_to(self.null_physio, (batch_size, 1, self.d_physio))
            h_p = nn.gelu(self.proj_physio(self.pool_physio(broadcast_null)))
        else:
            h_p = nn.gelu(self.proj_physio(self.pool_physio(x_physio, mask=mask_physio)))

        # 3. Concatenation and metric projection
        fused = mx.concatenate([h_a, h_k, h_p], axis=-1)
        fused = self.fusion_norm(fused)
        hidden = nn.gelu(self.fc1(fused))
        raw_metric = self.fc_metric(hidden)

        # 4. Explicit L2-normalization for cosine distance metric geometry
        norm = mx.sqrt(mx.sum(mx.square(raw_metric), axis=-1, keepdims=True) + 1e-8)
        z_metric = raw_metric / norm
        return cast(mx.array, z_metric)
