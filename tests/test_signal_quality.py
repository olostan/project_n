"""
Project N: Sensory Signal Quality Abstention Gates Tests (Gate B).
Verifies that low SNR, high clipping, poor pose tracking, or excessive
camera ego-motion fail-closed and suppress interpretive layers.
"""

import numpy as np

from extraction.signal_quality import (
    compute_audio_clipping_ratio,
    compute_audio_snr_db,
    evaluate_signal_quality,
)
from server.deps import get_analysis_service, get_sse_bus
from tests.fixtures.audio import (
    generate_clipped_audio_clip,
    generate_realistic_audio_clip,
    generate_stationary_audio_clip,
)


def test_audio_snr_computation() -> None:
    # 1. Pure silence: SNR = 0
    silence = np.zeros(48000, dtype=np.float32)
    assert compute_audio_snr_db(silence) == 0.0

    # 2. Stationary tone: P90 ~= P10, so SNR < 12.0 dB
    stationary = generate_stationary_audio_clip(duration_sec=2.0)
    assert compute_audio_snr_db(stationary) < 12.0

    # 3. Realistic burst audio with background noise floor: SNR >= 12.0 dB
    realistic = generate_realistic_audio_clip(duration_sec=2.0)
    assert compute_audio_snr_db(realistic) >= 12.0


def test_audio_clipping_ratio() -> None:
    # No clipping
    clean = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    assert compute_audio_clipping_ratio(clean) == 0.0

    # 10% clipped
    clipped = np.copy(clean)
    clipped[:100] = 1.0  # 100 / 1000 = 10%
    assert abs(compute_audio_clipping_ratio(clipped) - 0.10) < 1e-4


def test_signal_quality_individual_gate_breaches() -> None:
    clean_audio = generate_realistic_audio_clip(duration_sec=2.0)

    # Gate 1 breach: Low SNR (< 12 dB)
    low_snr_audio = generate_stationary_audio_clip(duration_sec=2.0)
    rep1 = evaluate_signal_quality(
        audio_pcm=low_snr_audio,
        mean_pose_confidence=0.85,
        mean_flow_velocity=10.0,
    )
    assert not rep1.is_acceptable
    assert not rep1.audio_valid
    assert any("Acoustic SNR" in b for b in rep1.breaches)

    # Gate 2 breach: Audio clipping (> 5%)
    clipped_audio = generate_clipped_audio_clip(duration_sec=2.0, clip_ratio=0.10)
    rep2 = evaluate_signal_quality(
        audio_pcm=clipped_audio,
        mean_pose_confidence=0.85,
        mean_flow_velocity=10.0,
    )
    assert not rep2.is_acceptable
    assert not rep2.audio_valid
    assert any("clipping" in b for b in rep2.breaches)

    # Gate 3 breach: Low pose tracking confidence (< 0.40)
    rep3 = evaluate_signal_quality(
        audio_pcm=clean_audio,
        mean_pose_confidence=0.25,
        mean_flow_velocity=10.0,
    )
    assert not rep3.is_acceptable
    assert not rep3.video_valid
    assert any("pose tracking" in b for b in rep3.breaches)

    # Gate 4 breach: Excessive optical flow velocity (> 85 px/s)
    rep4 = evaluate_signal_quality(
        audio_pcm=clean_audio,
        mean_pose_confidence=0.85,
        mean_flow_velocity=95.0,
    )
    assert not rep4.is_acceptable
    assert not rep4.video_valid
    assert any("optical flow" in b for b in rep4.breaches)

    # All pass
    clean_report = evaluate_signal_quality(
        audio_pcm=clean_audio,
        mean_pose_confidence=0.85,
        mean_flow_velocity=12.0,
    )
    assert clean_report.is_acceptable is True
    assert clean_report.audio_valid is True
    assert clean_report.video_valid is True
    assert len(clean_report.breaches) == 0


def test_analysis_service_abstains_on_corrupted_sensory_streams() -> None:
    service = get_analysis_service()
    sse_bus = get_sse_bus()

    # Create heavily clipped audio (100% clipped)
    clipped_audio = np.ones(240000, dtype=np.float32)
    blank_frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(150)]

    res = service.analyze_sensory_clip(
        clip_id="test_clip_corrupted_quality",
        audio_pcm=clipped_audio,
        video_frames=blank_frames,
    )

    # Layer 1 must contain signal quality report
    l1 = res["L1_measured"]
    assert "signal_quality" in l1
    sq = l1["signal_quality"]
    assert sq["is_acceptable"] is False
    assert len(sq["breaches"]) > 0

    # Layer 2 must be abstained
    l2 = res["L2_historical"]
    assert l2["status"] == "abstained"
    assert "breach" in l2["explanation"].lower()
    assert len(l2["matches"]) == 0

    # Precedence rule: Safety triage reports not_assessable_low_signal_quality
    safety = res["safety_triage"]
    assert safety["screener_status"] == "not_assessable_low_signal_quality"
    assert not safety["distress_anomaly_detected"]

    # SSE event published
    events = sse_bus.get_recent_events(limit=5)
    event_names = [e["event"] for e in events]
    assert "signal_quality_abstained" in event_names
