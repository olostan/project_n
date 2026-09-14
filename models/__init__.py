"""
Project N: Models Package (Tiers 0 & 2).
Canonical contracts, psychometrics, Apple MLX neural encoders, and triage models.
"""

from models.anomaly import AcuteDistressAnomalyDetector
from models.attention import AttentionPool
from models.baseline_profiler import PersonalBaselineProfiler
from models.clip_encoder import ClipSequenceAttentionPool, generate_sinusoidal_positional_encoding
from models.contracts import (
    ACOUSTIC_LATENT_D,
    AUDIO_SAMPLE_RATE_HZ,
    KINEMATIC_LATENT_D,
    METRIC_EMBEDDING_D,
    PHYSIOLOGY_LATENT_D,
    PITCH_FRAMES,
    RAW_AUDIO_SAMPLES,
    VIDEO_FPS,
    VIDEO_FRAMES,
    WINDOW_DURATION_SEC,
)
from models.matcher import EpisodicPrototypeMatcher
from models.nccpc import (
    CANONICAL_PV_ITEMS,
    CANONICAL_R_ITEMS,
    NCCPCScoreResult,
    score_nccpc_pv,
    score_nccpc_r,
)
from models.projection import MetricProjectionHead

__all__ = [
    "ACOUSTIC_LATENT_D",
    "AUDIO_SAMPLE_RATE_HZ",
    "KINEMATIC_LATENT_D",
    "METRIC_EMBEDDING_D",
    "PHYSIOLOGY_LATENT_D",
    "PITCH_FRAMES",
    "RAW_AUDIO_SAMPLES",
    "VIDEO_FPS",
    "VIDEO_FRAMES",
    "WINDOW_DURATION_SEC",
    "CANONICAL_PV_ITEMS",
    "CANONICAL_R_ITEMS",
    "AcuteDistressAnomalyDetector",
    "AttentionPool",
    "ClipSequenceAttentionPool",
    "EpisodicPrototypeMatcher",
    "MetricProjectionHead",
    "NCCPCScoreResult",
    "PersonalBaselineProfiler",
    "generate_sinusoidal_positional_encoding",
    "score_nccpc_pv",
    "score_nccpc_r",
]
