"""
Project N: Invariant 6 Four-Layer Output Separation Verification (SPECS.md:454).
Validates that analysis output strictly emits canonical keys:
L1_measured, L2_historical, L3_context, L4_evidence, safety_triage,
dyadic_suggestions, and parent_view_text.
"""

from server.deps import get_analysis_service
from tests.fixtures.audio import generate_realistic_audio_clip
from tests.fixtures.video import generate_valid_mock_video_frames


def test_four_layer_output_structure_normal() -> None:
    service = get_analysis_service()
    audio = generate_realistic_audio_clip(duration_sec=5.0, f0_hz=220.0)
    frames = generate_valid_mock_video_frames(num_frames=30)

    out = service.analyze_sensory_clip(
        clip_id="test_schema_norm",
        audio_pcm=audio,
        video_frames=frames,
        antecedent_id="post_school_transition",
        antecedent_notes="Post-school fatigue, noisy transit",
        caregiver_hypothesis="Possible sensory overload",
        setting="living_room",
        observer="caregiver",
    )

    # 1. Canonical Key Presence (SPECS.md:454)
    assert "L1_measured" in out
    assert "L2_historical" in out
    assert "L3_context" in out
    assert "L4_evidence" in out
    assert "safety_triage" in out
    assert "dyadic_suggestions" in out
    assert "parent_view_text" in out

    # 2. Layer 1: Objective observations only
    l1 = out["L1_measured"]
    assert "observed_f0_mean_hz" in l1
    assert "observed_cpp_db" in l1
    assert "observed_motion_rhythm_hz" in l1
    assert "acute_guarding_detected" in l1
    assert "signal_quality" in l1

    # 3. Layer 2: Comparable history (abstained or active)
    l2 = out["L2_historical"]
    assert "status" in l2
    assert "explanation" in l2
    assert "matches" in l2

    # 4. Layer 3: Context & Antecedents
    l3 = out["L3_context"]
    assert l3["antecedent_id"] == "post_school_transition"
    assert l3["caregiver_notes"] == "Post-school fatigue, noisy transit"
    assert l3["caregiver_hypothesis"] == "Possible sensory overload"
    assert l3["setting"] == "living_room"
    assert l3["observer"] == "caregiver"

    # 5. Layer 4: Evidence Library
    l4 = out["L4_evidence"]
    assert "status" in l4
    assert "citations" in l4
    # When query matches, citations list contains valid DOIs
    if l4["status"] == "matched_citations":
        assert len(l4["citations"]) > 0
        assert "doi" in l4["citations"][0]
        assert "evidence_level" in l4["citations"][0]

    # 6. Dyadic Suggestions
    dyad = out["dyadic_suggestions"]
    assert "suggested_actions" in dyad
    assert "supporting_episode_ids" in dyad
    assert "rationale" in dyad

    # 7. Safety Triage
    safety = out["safety_triage"]
    assert "clinical_disclaimer" in safety
    assert "distress_anomaly_detected" in safety
    assert safety["distress_anomaly_detected"] is False

    # 8. Parent View Text
    pv = out["parent_view_text"]
    assert isinstance(pv, str)
    assert len(pv) > 20


def test_four_layer_output_suppression_on_distress() -> None:
    """When distress anomaly triggers, Layer 2 MUST be suppressed per Invariant 7 & 6."""
    service = get_analysis_service()
    high_pitch_audio = generate_realistic_audio_clip(duration_sec=5.0, f0_hz=500.0)
    frames = generate_valid_mock_video_frames(num_frames=30)

    # Provide calibrated baseline where upper f0 limit is 400 Hz (500 Hz triggers excursion)
    baseline_stats = {
        "calibrated_episodes_count": 15,
        "calibrated_days_count": 7,
        "f0_upper_limit_hz": 400.0,
        "cpp_lower_limit_db": 3.0,
    }

    out = service.analyze_sensory_clip(
        clip_id="test_schema_distress",
        audio_pcm=high_pitch_audio,
        video_frames=frames,
        baseline_stats=baseline_stats,
    )

    assert out["safety_triage"]["distress_anomaly_detected"] is True
    l2 = out["L2_historical"]
    assert l2["status"] == "suppressed_due_to_anomaly"
    assert len(l2["matches"]) == 0
    assert "suppressed" in l2["explanation"].lower()
    assert "elevated" in out["parent_view_text"].lower()
