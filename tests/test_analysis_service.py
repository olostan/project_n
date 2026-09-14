"""
Project N: Integration Tests for Analysis Pipeline Service (Phase 2).
Verifies end-to-end multimodal extraction, metric projection, anomaly screening,
and 4-layer output structuring.
"""

import numpy as np
import pytest

from models.contracts import METRIC_EMBEDDING_D, RAW_AUDIO_SAMPLES
from rag.vector_store import VectorStore
from server.services.analysis_service import AnalysisService
from server.sse_bus import SSEBus
from storage.db_schema import init_db
from storage.episode_repo import EpisodeRepository
from storage.vault import MockKeychainProvider, VaultManager


@pytest.fixture
def analysis_service() -> AnalysisService:
    db = init_db(":memory:")
    repo = EpisodeRepository(db=db)
    vault = VaultManager(keychain=MockKeychainProvider())
    v_store = VectorStore(persist_path=None)
    sse = SSEBus()
    return AnalysisService(
        vault=vault,
        episode_repo=repo,
        vector_store=v_store,
        sse_bus=sse,
    )


def test_analysis_service_end_to_end_cold_start(analysis_service: AnalysisService) -> None:
    # 1. Synthetic 5.0s audio (240,000 samples)
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False)
    synthetic_audio = (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    # 2. Synthetic 150 video frames
    synthetic_frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(150)]

    clip_id = "test_clip_001"
    res = analysis_service.analyze_sensory_clip(
        clip_id=clip_id,
        audio_pcm=synthetic_audio,
        video_frames=synthetic_frames,
        antecedent_id="mealtime",
        baseline_stats=None,  # Cold start / uncalibrated
    )

    # Verify return schema
    assert res["episode_id"] == clip_id
    assert res["encoder_version"] == "v1.0.0"
    assert len(res["metric_embedding"]) == METRIC_EMBEDDING_D

    # Verify L1 Sensory Observations
    l1 = res["layer1_sensory"]
    assert l1["windows_count"] == 1
    assert "observed_f0_mean_hz" in l1
    assert "observed_motion_rhythm_hz" in l1
    assert not l1["acute_guarding_detected"]

    # Verify L2 Hypotheses (Matcher in cold-start collecting state)
    l2 = res["layer2_hypotheses"]
    assert l2["status"] == "abstained"

    # Verify L4 Safety Triage
    l4 = res["layer4_safety"]
    assert not l4["distress_anomaly_detected"]
    assert "screener_recommendation" in l4
    assert "clinical_disclaimer" in l4

    # Verify Episode record in database
    ep_record = analysis_service.episode_repo.get_episode(clip_id)
    assert ep_record is not None
    assert ep_record["id"] == clip_id
    assert ep_record["antecedent_id"] == "mealtime"


def test_analysis_service_acute_distress_triage_precedence(
    analysis_service: AnalysisService,
) -> None:
    # 1. Synthetic high-pitch distress audio (>450 Hz)
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False)
    high_pitch_audio = (0.7 * np.sin(2 * np.pi * 500.0 * t)).astype(np.float32)
    synthetic_frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(150)]

    # Calibrated baseline where upper f0 limit is 400 Hz
    calibrated_baseline = {
        "calibrated_episodes_count": 12.0,
        "calibrated_days_count": 6.0,
        "f0_upper_limit_hz": 400.0,
        "cpp_lower_limit_db": 3.0,
    }

    clip_id = "test_clip_distress_002"
    res = analysis_service.analyze_sensory_clip(
        clip_id=clip_id,
        audio_pcm=high_pitch_audio,
        video_frames=synthetic_frames,
        antecedent_id="loud_environment",
        baseline_stats=calibrated_baseline,
    )

    # Invariant 8: Medical Triage Precedence
    l2 = res["layer2_hypotheses"]
    assert l2["status"] == "suppressed_due_to_anomaly"

    l3 = res["layer3_dyadic"]
    assert "hydration_water" in l3["suggested_actions"]

    l4 = res["layer4_safety"]
    assert l4["distress_anomaly_detected"]
    assert "PHYSICAL COMFORT CHECK SUGGESTED" in l4["screener_recommendation"]
