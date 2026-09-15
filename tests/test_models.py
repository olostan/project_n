"""
Project N: Unit Tests for Tier 2 Neural Models and Decision Triage.
Verifies AttentionPool, MetricProjectionHead, ClipSequenceAttentionPool,
AcuteDistressAnomalyDetector, EpisodicPrototypeMatcher, and PersonalBaselineProfiler.
"""

from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import pytest

from models.anomaly import AcuteDistressAnomalyDetector
from models.attention import AttentionPool
from models.baseline_profiler import PersonalBaselineProfiler
from models.clip_encoder import ClipSequenceAttentionPool, generate_sinusoidal_positional_encoding
from models.contracts import (
    ACOUSTIC_LATENT_D,
    ACOUSTIC_LATENT_T,
    KINEMATIC_LATENT_D,
    KINEMATIC_LATENT_T,
    METRIC_EMBEDDING_D,
    PHYSIOLOGY_LATENT_D,
    PHYSIOLOGY_STEPS,
)
from models.matcher import EpisodicPrototypeMatcher
from models.projection import MetricProjectionHead


def test_attention_pool_forward() -> None:
    batch_size = 2
    time_steps = 100
    d_model = 64

    pool = AttentionPool(d_in=d_model)
    x = mx.random.normal((batch_size, time_steps, d_model))
    out = pool(x)

    assert out.shape == (batch_size, d_model)


def test_attention_pool_masking() -> None:
    pool = AttentionPool(d_in=16)
    x = mx.ones((1, 4, 16))
    # Mask out the last two steps using boolean mask (True = valid, False = mask)
    mask = mx.array([[True, True, False, False]])
    out = pool(x, mask=mask)

    assert out.shape == (1, 16)
    assert not mx.isnan(out).any()


def test_metric_projection_head_l2_norm() -> None:
    head = MetricProjectionHead()

    batch_size = 3
    x_audio = mx.random.normal((batch_size, ACOUSTIC_LATENT_T, ACOUSTIC_LATENT_D))
    x_kinematic = mx.random.normal((batch_size, KINEMATIC_LATENT_T, KINEMATIC_LATENT_D))
    x_physio = mx.random.normal((batch_size, PHYSIOLOGY_STEPS, PHYSIOLOGY_LATENT_D))

    z_metric = head(x_audio, x_kinematic, x_physio)
    assert z_metric.shape == (batch_size, METRIC_EMBEDDING_D)

    # Invariant: Each row is L2-normalized: sum(z^2) == 1.0
    sq_sums = mx.sum(mx.square(z_metric), axis=-1)
    for s in cast(list[float], sq_sums.tolist()):
        assert pytest.approx(s, rel=1e-4) == 1.0


def test_metric_projection_head_missing_modality() -> None:
    head = MetricProjectionHead()

    batch_size = 2
    x_audio = mx.random.normal((batch_size, ACOUSTIC_LATENT_T, ACOUSTIC_LATENT_D))
    x_kinematic = mx.random.normal((batch_size, KINEMATIC_LATENT_T, KINEMATIC_LATENT_D))

    # Test graceful degradation with x_physio = None
    z_degraded = head(x_audio, x_kinematic, x_physio=None)
    assert z_degraded.shape == (batch_size, METRIC_EMBEDDING_D)

    sq_sums = mx.sum(mx.square(z_degraded), axis=-1)
    for s in cast(list[float], sq_sums.tolist()):
        assert pytest.approx(s, rel=1e-4) == 1.0


def test_clip_encoder_positional_magnitude_decoupling() -> None:
    clip_pool = ClipSequenceAttentionPool(d_metric=METRIC_EMBEDDING_D)

    batch_size = 2
    windows_count = 12
    # Generate random L2-normalized window vectors
    raw = mx.random.normal((batch_size, windows_count, METRIC_EMBEDDING_D))
    norm = mx.sqrt(mx.sum(mx.square(raw), axis=-1, keepdims=True) + 1e-8)
    z_windows = raw / norm

    pe = generate_sinusoidal_positional_encoding(windows_count, METRIC_EMBEDDING_D)
    assert pe.shape == (windows_count, METRIC_EMBEDDING_D)

    z_clip = clip_pool(z_windows, p_encodings=pe)
    assert z_clip.shape == (batch_size, METRIC_EMBEDDING_D)

    # Invariant: Output is L2-normalized
    sq_sums = mx.sum(mx.square(z_clip), axis=-1)
    for s in cast(list[float], sq_sums.tolist()):
        assert pytest.approx(s, rel=1e-4) == 1.0


def test_anomaly_detector_cold_start() -> None:
    # Test uncalibrated baseline (<10 clean episodes)
    features = {"f0_mean_hz": 480.0, "cpp_db": 3.0}
    incomplete_stats = {"calibrated_episodes_count": 5.0, "calibrated_days_count": 2.0}

    result = AcuteDistressAnomalyDetector.evaluate(features, baseline_stats=incomplete_stats)
    assert not result["screener_available"]
    assert result["screener_status"] == "uncalibrated_collecting_baseline"
    assert not result["distress_anomaly"]


def test_anomaly_detector_calibrated() -> None:
    calibrated_stats = {
        "calibrated_episodes_count": 15.0,
        "calibrated_days_count": 7.0,
        "f0_upper_limit_hz": 400.0,
        "cpp_lower_limit_db": 5.0,
    }

    # Case 1: Normal within limits
    normal_res = AcuteDistressAnomalyDetector.evaluate(
        {"f0_mean_hz": 280.0, "cpp_db": 8.0, "acute_guarding_detected": False},
        baseline_stats=calibrated_stats,
    )
    assert normal_res["screener_available"]
    assert not normal_res["distress_anomaly"]
    assert "No configured signal deviation was detected" in normal_res["recommendation"]

    # Case 2: Acute pitch excursion anomaly
    spike_res = AcuteDistressAnomalyDetector.evaluate(
        {"f0_mean_hz": 490.0, "cpp_db": 8.0, "acute_guarding_detected": False},
        baseline_stats=calibrated_stats,
    )
    assert spike_res["distress_anomaly"]
    assert spike_res["indicators"]["f0_spike"]
    assert "PHYSICAL COMFORT CHECK SUGGESTED" in spike_res["recommendation"]


class MockCollection:
    def __init__(
        self, count_val: int, distances: list[float], metadatas: list[dict[str, Any]]
    ) -> None:
        self._count = count_val
        self.distances = distances
        self.metadatas = metadatas

    def count(self) -> int:
        return self._count

    def query(self, **_kwargs: Any) -> dict[str, Any]:
        return {"distances": [self.distances], "metadatas": [self.metadatas]}


def test_prototype_matcher_lifecycle_and_abstention() -> None:
    query = mx.random.normal((1, METRIC_EMBEDDING_D))

    # 1. Collecting state (< 20 episodes)
    mock_coll = MockCollection(count_val=15, distances=[0.2], metadatas=[{}])
    matcher = EpisodicPrototypeMatcher(tau_abstain=0.35, is_calibrated=True)
    res = matcher.match(query, mock_coll)
    assert res["abstained"]
    assert res["matcher_state"] == "collecting"

    # 2. Awaiting calibration state (>= 20 episodes, is_calibrated=False)
    matcher_uncal = EpisodicPrototypeMatcher(tau_abstain=0.35, is_calibrated=False)
    mock_coll_20 = MockCollection(count_val=25, distances=[0.2], metadatas=[{}])
    res2 = matcher_uncal.match(query, mock_coll_20)
    assert res2["abstained"]
    assert res2["matcher_state"] == "awaiting_calibration"

    # 3. Active state with familiar episode (distance <= tau_abstain)
    matcher_active = EpisodicPrototypeMatcher(tau_abstain=0.35, is_calibrated=True)
    mock_coll_active = MockCollection(
        count_val=30,
        distances=[0.15],
        metadatas=[{"episode_id": "ep_1", "action_offered": "deep_pressure_proprioceptive"}],
    )
    res3 = matcher_active.match(query, mock_coll_active)
    assert not res3["abstained"]
    assert res3["matcher_state"] == "active"
    assert len(res3["candidates"]) == 1
    assert res3["candidates"][0]["action_offered"] == "deep_pressure_proprioceptive"

    # 4. Active state with unfamiliar episode (distance > tau_abstain -> Abstention)
    mock_coll_novel = MockCollection(
        count_val=30, distances=[0.55], metadatas=[{"episode_id": "ep_2"}]
    )
    res4 = matcher_active.match(query, mock_coll_novel)
    assert res4["abstained"]
    assert res4["matcher_state"] == "active_abstaining"


def test_personal_baseline_profiler() -> None:
    episodes = [
        {"f0_mean_hz": 260.0, "cpp_db": 8.0, "captured_date": f"2026-09-0{i % 7 + 1}"}
        for i in range(15)
    ]
    stats = PersonalBaselineProfiler.compute_baseline(episodes)
    assert stats["calibrated_episodes_count"] == 15.0
    assert stats["calibrated_days_count"] == 7.0
    assert stats["f0_mean_hz"] == 260.0
    assert stats["f0_upper_limit_hz"] >= 400.0
    assert stats["cpp_mean_db"] == 8.0
    assert stats["cpp_lower_limit_db"] <= 8.0


def test_feature_standardizer(tmp_path: Path) -> None:
    from models.standardizer import FeatureStandardizer

    # 1. Unfitted transform/save raises ValueError
    std_scaler = FeatureStandardizer()
    dummy = mx.ones((2, 10))
    with pytest.raises(ValueError, match="must be fitted"):
        std_scaler.transform(dummy)
    with pytest.raises(ValueError, match="Cannot save unfitted"):
        std_scaler.save(tmp_path / "scaler.npz")

    # 2. Fit 3D tensor (B, T, D)
    B, T, D = 4, 20, 16
    raw_data = mx.random.normal((B, T, D)) * 5.0 + 10.0
    std_scaler.fit(raw_data)
    assert std_scaler.mean is not None and std_scaler.mean.shape == (D,)
    assert std_scaler.std is not None and std_scaler.std.shape == (D,)

    # 3. Transform and verify zero mean, unit variance
    z = std_scaler.transform(raw_data)
    assert z.shape == (B, T, D)
    z_flat = z.reshape(-1, D)
    mean_z = mx.mean(z_flat, axis=0)
    var_z = mx.var(z_flat, axis=0)
    for m in cast(list[float], mean_z.tolist()):
        assert pytest.approx(m, abs=1e-4) == 0.0
    for v in cast(list[float], var_z.tolist()):
        assert pytest.approx(v, rel=1e-3) == 1.0

    # 4. Save and load parity
    scaler_file = tmp_path / "scaler.npz"
    std_scaler.save(scaler_file)
    assert scaler_file.exists()

    loaded = FeatureStandardizer.load(scaler_file)
    z_loaded = loaded.transform(raw_data)
    diff = mx.max(mx.abs(z - z_loaded))
    assert float(diff) == 0.0
