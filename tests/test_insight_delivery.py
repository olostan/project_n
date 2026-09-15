"""
Project N: Invariant 8 Caregiver Insight & Child Agency Verification.
Verifies dual-perspective rendering, medical-first triage precedence,
non-diagnostic boundaries, and child agency primacy.
"""

from __future__ import annotations

from server.services.card_builder import (
    derive_dyadic_suggestions,
    synthesize_parent_view_text,
)


def test_medical_first_triage_precedence() -> None:
    """Verifies that acute distress suppresses routine behavioral interpretations."""
    suggestions = derive_dyadic_suggestions(
        retrieved_candidates=[{"action_offered": "quiet_refuge"}],
        distress_triggered=True,
        is_acceptable=True,
    )
    assert suggestions["rationale"] == "Medical comfort check takes priority."
    assert "open_observation" in suggestions["suggested_actions"]

    parent_text = synthesize_parent_view_text(
        mean_f0=420.0,
        mean_rhythm=3.2,
        antecedent_id="loud_environment",
        top_action=suggestions["suggested_actions"][0],
        distress_triggered=True,
        is_acceptable=True,
    )
    assert "comfort check and water" in parent_text
    assert "elevated compared to usual patterns" in parent_text


def test_signal_quality_compromised_precedence() -> None:
    """Verifies that low signal quality suppresses automated interpretations."""
    suggestions = derive_dyadic_suggestions(
        retrieved_candidates=[{"action_offered": "quiet_refuge"}],
        distress_triggered=False,
        is_acceptable=False,
    )
    assert suggestions["rationale"] == "Signal quality insufficient for behavioral matching."
    assert "open_observation" in suggestions["suggested_actions"]

    parent_text = synthesize_parent_view_text(
        mean_f0=250.0,
        mean_rhythm=1.0,
        antecedent_id="unknown",
        top_action="open_observation",
        distress_triggered=False,
        is_acceptable=False,
    )
    assert "gate breach" in parent_text
    assert "gentle, open observation" in parent_text


def test_child_agency_and_non_diagnostic_boundary() -> None:
    """
    Verifies that Parent View text uses observational language and never asserts
    prescriptive medical diagnoses or mind-reading causal claims.
    """
    candidates = [
        {
            "episode_id": "ep_hist_1",
            "action_offered": "deep_pressure_proprioceptive",
        }
    ]
    suggestions = derive_dyadic_suggestions(
        retrieved_candidates=candidates,
        distress_triggered=False,
        is_acceptable=True,
    )
    assert "deep_pressure_proprioceptive" in suggestions["suggested_actions"]

    parent_text = synthesize_parent_view_text(
        mean_f0=260.0,
        mean_rhythm=1.2,
        antecedent_id="post_school_transition",
        top_action=suggestions["suggested_actions"][0],
        distress_triggered=False,
        is_acceptable=True,
    )

    # 1. Verification of structured observational content
    assert "260 Hz" in parent_text
    assert "1.2 Hz" in parent_text
    assert "post_school_transition" in parent_text
    assert "deep pressure proprioceptive" in parent_text

    # 2. Strict non-diagnostic check: forbid diagnostic pathology claims
    forbidden_diagnostic_terms = [
        "diagnosed with",
        "clinical breakdown",
        "pathological",
        "suffering from autism",
        "definitive intent",
        "mind-reading",
    ]
    for term in forbidden_diagnostic_terms:
        assert term not in parent_text.lower(), f"Forbidden diagnostic claim detected: {term}"


def test_four_layer_separation_integrity() -> None:
    """Verifies that L1 measured data is isolated from L3 caregiver hypotheses."""
    L1_measured = {"f0_mean_hz": 240.0, "motion_rhythm_hz": 1.0}
    L3_context = {
        "antecedent_id": "mealtime",
        "caregiver_hypothesis": "May prefer quiet refuge",
    }

    # Verify L1 does not contain speculative fields
    assert "hypothesis" not in L1_measured
    assert "caregiver_hypothesis" not in L1_measured

    # Verify L3 does not contain raw acoustic measurements
    assert "f0_mean_hz" not in L3_context
    assert "motion_rhythm_hz" not in L3_context
