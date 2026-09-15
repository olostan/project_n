"""
Project N: Integration Tests for Analysis Pipeline Service (Phase 2).
Verifies end-to-end multimodal extraction, metric projection, anomaly screening,
and 4-layer output structuring.
"""

from pathlib import Path

import pytest

from models.contracts import METRIC_EMBEDDING_D
from rag.vector_store import VectorStore
from server.services.analysis_service import AnalysisService
from server.sse_bus import SSEBus
from storage.db_schema import init_db
from storage.episode_repo import EpisodeRepository
from storage.vault import MockKeychainProvider, VaultManager
from tests.fixtures.audio import generate_realistic_audio_clip
from tests.fixtures.video import generate_valid_mock_video_frames


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
    # 1. Realistic 5.0s burst audio (240,000 samples)
    realistic_audio = generate_realistic_audio_clip(duration_sec=5.0, f0_hz=260.0)

    # 2. Valid 150 mock video frames
    valid_frames = generate_valid_mock_video_frames(num_frames=150)

    clip_id = "test_clip_001"
    res = analysis_service.analyze_sensory_clip(
        clip_id=clip_id,
        audio_pcm=realistic_audio,
        video_frames=valid_frames,
        antecedent_id="mealtime",
        baseline_stats=None,  # Cold start / uncalibrated
    )

    # Verify return schema in uncalibrated cold start
    assert res["episode_id"] == clip_id
    assert res["encoder_version"] == "uncalibrated_v0"
    assert res["metric_embedding"] is None

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
    # 1. Realistic high-pitch distress burst audio (>450 Hz)
    high_pitch_audio = generate_realistic_audio_clip(duration_sec=5.0, f0_hz=500.0)
    valid_frames = generate_valid_mock_video_frames(num_frames=150)

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
        video_frames=valid_frames,
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


def test_analysis_service_with_trained_checkpoint(
    tmp_path: Path, analysis_service: AnalysisService
) -> None:
    # Save dummy checkpoint and load into projection head
    ckpt_file = tmp_path / "model.safetensors"
    analysis_service.projection_head.save_checkpoint(str(ckpt_file))
    ckpt_hash = analysis_service.projection_head.load_checkpoint(str(ckpt_file))
    assert analysis_service.projection_head.is_trained

    audio = generate_realistic_audio_clip(duration_sec=5.0, f0_hz=260.0)
    frames = generate_valid_mock_video_frames(num_frames=150)

    res = analysis_service.analyze_sensory_clip(
        clip_id="test_clip_trained",
        audio_pcm=audio,
        video_frames=frames,
    )

    assert res["encoder_version"] == f"ckpt_{ckpt_hash[:8]}"
    assert res["metric_embedding"] is not None
    assert len(res["metric_embedding"]) == METRIC_EMBEDDING_D
    assert analysis_service.get_clip_embedding("test_clip_trained") is not None
