"""
Project N: Test Fixtures for Synthetic Episodes & Feature Triplets.
Used exclusively in testing suites to benchmark training pipelines and candidate evaluation.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx

from models.contracts import (
    ACOUSTIC_LATENT_D,
    ACOUSTIC_LATENT_T,
    KINEMATIC_LATENT_D,
    KINEMATIC_LATENT_T,
    METRIC_EMBEDDING_D,
)
from storage.episode_repo import EpisodeRepository


def seed_synthetic_episodes(repo: EpisodeRepository, count: int = 15) -> list[dict[str, Any]]:
    """
    Seeds the episode repository with deterministic confirmed episodes.
    Creates structured clusters so that same-antecedent episodes have correlated embeddings.
    """
    antecedents = [
        "post_school_transition",
        "mealtime",
        "bedtime_routine",
        "loud_environment",
    ]
    actions = [
        "quiet_refuge",
        "deep_pressure_proprioceptive",
        "sensory_break",
        "dimmed_lighting",
    ]

    episodes: list[dict[str, Any]] = []

    for i in range(count):
        ant_idx = i % len(antecedents)
        antecedent = antecedents[ant_idx]
        action = actions[ant_idx]

        # 80% settled, 20% escalated
        is_escalated = i % 5 == 4
        outcome = "escalated" if is_escalated else "settled_immediately"
        pain_breach = 1 if (is_escalated and i % 2 == 0) else -1

        # Generate cluster-correlated 128-dim metric embedding
        # Base angle determined by antecedent
        base_angle = (ant_idx * 2.0 * math.pi) / len(antecedents)
        raw_emb = [
            math.sin(base_angle + (j * 0.1) + (0.5 if is_escalated else 0.0))
            for j in range(METRIC_EMBEDDING_D)
        ]
        norm = math.sqrt(sum(x * x for x in raw_emb) + 1e-8)
        norm_emb = [round(x / norm, 6) for x in raw_emb]

        ep_id = f"ep_fixture_{i:03d}"
        ep_data = {
            "id": ep_id,
            "vault_uri": f"vault://clips/{ep_id}.enc",
            "encoder_version_id": "v1.0.0",
            "captured_at": f"2026-09-{(i % 28) + 1:02d}T12:00:00Z",
            "duration_ms": 15000,
            "windows_count": 3,
            "observed_f0_mean": 280.0 + (50.0 if is_escalated else 0.0),
            "observed_motion_rhythm_hz": 1.2,
            "antecedent_id": antecedent,
            "action_offered": action,
            "action_performed": action,
            "caregiver_decision": "accepted",
            "outcome_state": outcome,
            "settled_within_sec": None if is_escalated else 180,
            "child_response": "reach",
            "performance_status": "completed",
            "pain_cutoff_breached": pain_breach,
            "metric_embedding": norm_emb,
        }
        repo.insert_episode(ep_data)
        episodes.append(ep_data)

    return episodes


def generate_mock_feature_triplets(
    batch_size: int = 4,
    t_a: int = ACOUSTIC_LATENT_T,
    t_k: int = KINEMATIC_LATENT_T,
    d_a: int = ACOUSTIC_LATENT_D,
    d_k: int = KINEMATIC_LATENT_D,
    num_batches: int = 3,
) -> list[tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]]:
    """Generates synthetic feature batches for testing projection head training."""
    batches = []
    for _ in range(num_batches):
        a_a = mx.random.normal((batch_size, t_a, d_a))
        a_k = mx.random.normal((batch_size, t_k, d_k))
        # Positive has slight perturbation
        p_a = a_a + mx.random.normal(a_a.shape) * 0.05
        p_k = a_k + mx.random.normal(a_k.shape) * 0.05
        # Negative is distinct
        n_a = mx.random.normal(a_a.shape)
        n_k = mx.random.normal(a_k.shape)
        batches.append((a_a, a_k, p_a, p_k, n_a, n_k))
    return batches
