"""
Project N: Invariant 6 Four-Layer Output Separation Verification.
Validates that analysis output strictly partitions into Layer 1 (Sensory),
Layer 2 (Hypotheses), Layer 3 (Dyadic Suggestions), Layer 4 (Safety/Clinical).
"""

import numpy as np

from server.deps import get_analysis_service


def test_four_layer_output_structure_normal() -> None:
    service = get_analysis_service()
    audio = np.zeros(240000, dtype=np.float32)
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(30)]

    out = service.analyze_sensory_clip(
        clip_id="test_schema_norm",
        audio_pcm=audio,
        video_frames=frames,
    )

    # 1. Layer Presence
    assert "layer1_sensory" in out
    assert "layer2_hypotheses" in out
    assert "layer3_dyadic" in out
    assert "layer4_safety" in out

    # 2. Layer 1: Objective observations only
    l1 = out["layer1_sensory"]
    assert "observed_f0_mean_hz" in l1
    assert "observed_cpp_db" in l1
    assert "observed_motion_rhythm_hz" in l1
    assert "acute_guarding_detected" in l1

    # 3. Layer 2: Interpretations (abstained or active)
    l2 = out["layer2_hypotheses"]
    assert "status" in l2
    assert "explanation" in l2
    assert "matches" in l2

    # 4. Layer 3: Caregiver actions
    l3 = out["layer3_dyadic"]
    assert "suggested_actions" in l3
    assert "rationale" in l3

    # 5. Layer 4: Disclaimer and Screener
    l4 = out["layer4_safety"]
    assert "clinical_disclaimer" in l4
    assert "distress_anomaly_detected" in l4
    assert l4["distress_anomaly_detected"] is False


def test_four_layer_output_suppression_on_distress() -> None:
    """When distress anomaly triggers, Layer 2 MUST be suppressed per Invariant 7 & 6."""
    service = get_analysis_service()
    audio = np.zeros(240000, dtype=np.float32)
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(30)]

    # Provide baseline where limits trigger distress anomaly
    baseline_stats = {
        "calibrated_episodes_count": 15,
        "calibrated_days_count": 7,
        "f0_upper_limit_hz": 50.0,  # Below measured or guard triggered
        "cpp_lower_limit_db": 50.0,  # Floor > measured triggers anomaly
    }

    out = service.analyze_sensory_clip(
        clip_id="test_schema_distress",
        audio_pcm=audio,
        video_frames=frames,
        baseline_stats=baseline_stats,
    )

    assert out["layer4_safety"]["distress_anomaly_detected"] is True
    l2 = out["layer2_hypotheses"]
    assert l2["status"] == "suppressed_due_to_anomaly"
    assert len(l2["matches"]) == 0
    assert "suppressed" in l2["explanation"].lower()
