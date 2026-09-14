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


def test_audio_snr_computation() -> None:
    # Pure silence: SNR = 0
    silence = np.zeros(48000, dtype=np.float32)
    assert compute_audio_snr_db(silence) == 0.0

    # High signal with low noise
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    noise = np.random.normal(0, 0.001, 48000).astype(np.float32)
    high_snr_audio = tone + noise
    assert compute_audio_snr_db(high_snr_audio) >= 12.0


def test_audio_clipping_ratio() -> None:
    # No clipping
    clean = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
    assert compute_audio_clipping_ratio(clean) == 0.0

    # 10% clipped
    clipped = np.copy(clean)
    clipped[:100] = 1.0  # 100 / 1000 = 10%
    assert abs(compute_audio_clipping_ratio(clipped) - 0.10) < 1e-4


def test_signal_quality_report_evaluation() -> None:
    audio = np.random.normal(0, 0.01, 48000).astype(np.float32)

    # 1. Low SNR and low pose confidence -> both fail
    report = evaluate_signal_quality(
        audio_pcm=audio,
        mean_pose_confidence=0.25,  # < 0.40
        mean_flow_velocity=95.0,  # > 85.0
    )
    assert report.all_modalities_failed is True
    assert len(report.breaches) >= 2

    # 2. High quality audio and video -> both pass
    t = np.linspace(0, 1.0, 48000, endpoint=False)
    clean_audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    clean_report = evaluate_signal_quality(
        audio_pcm=clean_audio,
        mean_pose_confidence=0.85,
        mean_flow_velocity=12.0,
    )
    assert clean_report.audio_valid is True
    assert clean_report.video_valid is True
    assert clean_report.all_modalities_failed is False
    assert len(clean_report.breaches) == 0


def test_analysis_service_abstains_on_corrupted_sensory_streams() -> None:
    service = get_analysis_service()
    sse_bus = get_sse_bus()

    # Create heavily clipped audio (100% clipped)
    clipped_audio = np.ones(240000, dtype=np.float32)
    # Blank video frames where pose confidence will be 0.0
    blank_frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(150)]

    res = service.analyze_sensory_clip(
        clip_id="test_clip_corrupted_quality",
        audio_pcm=clipped_audio,
        video_frames=blank_frames,
    )

    # Layer 1 must contain signal quality report
    l1 = res["layer1_sensory"]
    assert "signal_quality" in l1
    sq = l1["signal_quality"]
    assert sq["all_modalities_failed"] is True

    # Layer 2 must be abstained
    l2 = res["layer2_hypotheses"]
    assert l2["status"] == "abstained"
    assert "quality" in l2["explanation"].lower() and "breach" in l2["explanation"].lower()
    assert len(l2["matches"]) == 0

    # SSE event published
    events = sse_bus.get_recent_events(limit=5)
    event_names = [e["event"] for e in events]
    assert "signal_quality_abstained" in event_names
