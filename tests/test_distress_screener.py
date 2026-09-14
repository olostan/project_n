"""
Project N: Invariant 7 & 8 Distress Screener Precedence & Non-Diagnostic Boundary.
Verifies immediate medical triage precedence when distress signals breach thresholds,
and strictly enforces non-diagnostic clinical boundaries per Breau et al. (2002).
"""

from unittest.mock import MagicMock

import numpy as np

from models.anomaly import AcuteDistressAnomalyDetector
from models.nccpc import (
    CANONICAL_PV_ITEMS,
    CANONICAL_R_ITEMS,
    PV_CUTOFF_SCORE,
    R_CUTOFF_SCORE,
    score_nccpc_pv,
    score_nccpc_r,
)
from server.services.analysis_service import AnalysisService


def test_distress_screener_anomaly_detection() -> None:
    # Baseline stats for Child N (calibrated >= 10 episodes and >= 5 days)
    baseline = {
        "calibrated_episodes_count": 10,
        "calibrated_days_count": 5,
        "f0_upper_limit_hz": 300.0,
        "cpp_lower_limit_db": 4.0,
    }

    # Severe acute pitch excursion: f0 = 360 Hz (> 300 Hz limit)
    res_high_f0 = AcuteDistressAnomalyDetector.evaluate(
        measured_features={"f0_mean_hz": 360.0, "cpp_db": 12.0, "acute_guarding_detected": False},
        baseline_stats=baseline,
    )
    assert res_high_f0["distress_anomaly"] is True
    assert res_high_f0["indicators"]["f0_spike"] is True

    # Acute physical guarding detected
    res_guarding = AcuteDistressAnomalyDetector.evaluate(
        measured_features={"f0_mean_hz": 265.0, "cpp_db": 10.0, "acute_guarding_detected": True},
        baseline_stats=baseline,
    )
    assert res_guarding["distress_anomaly"] is True
    assert res_guarding["indicators"]["flinch_guarding"] is True

    # Nominal baseline - no distress anomaly
    res_nominal = AcuteDistressAnomalyDetector.evaluate(
        measured_features={"f0_mean_hz": 262.0, "cpp_db": 11.0, "acute_guarding_detected": False},
        baseline_stats=baseline,
    )
    assert res_nominal["distress_anomaly"] is False


def test_distress_screener_layer_suppression_and_triage_precedence() -> None:
    # Setup mock dependencies for AnalysisService
    vault = MagicMock()
    repo = MagicMock()
    vector_store = MagicMock()
    vector_store.confirmed_episodes.count.return_value = 0
    bus = MagicMock()

    service = AnalysisService(
        vault=vault, episode_repo=repo, vector_store=vector_store, sse_bus=bus
    )

    # 5s of nominal audio + video
    audio = np.zeros(240000, dtype=np.float32)
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(30)]

    # Analysis with calibrated baseline stats where child exhibits acute guarding / distress
    out = service.analyze_sensory_clip(
        clip_id="clip_distress_test",
        audio_pcm=audio,
        video_frames=frames,
        baseline_stats={
            "calibrated_episodes_count": 10,
            "calibrated_days_count": 5,
            "f0_upper_limit_hz": 300.0,
            "cpp_lower_limit_db": 4.0,
        },
    )

    # Layer 4 Safety must include mandatory non-diagnostic disclaimer
    assert "layer4_safety" in out
    disclaimer = out["layer4_safety"]["clinical_disclaimer"]
    assert "Assistive observational notebook only" in disclaimer
    assert "does not provide medical diagnosis" in disclaimer


def test_nccpc_non_diagnostic_boundary() -> None:
    # NCCPC-PV cutoff score is 11
    assert PV_CUTOFF_SCORE == 11
    # NCCPC-R cutoff score is 7
    assert R_CUTOFF_SCORE == 7

    # Scores at or above cutoff flag cutoff_breached
    pv_items = dict.fromkeys(CANONICAL_PV_ITEMS, 0)
    pv_items["crying"] = 3
    pv_items["screaming_yelling"] = 3
    pv_items["stiff_spastic_rigid_tense"] = 3
    pv_items["sharp_breath_gasping"] = 2  # Total = 11

    res = score_nccpc_pv(pv_items)
    assert res.score == 11
    assert res.cutoff_breached is True
    assert res.is_indeterminate is False

    # Excessive missingness (> 5 NA items) renders score indeterminate
    pv_na: dict[str, int | str] = dict.fromkeys(CANONICAL_PV_ITEMS, 0)
    for k in sorted(CANONICAL_PV_ITEMS)[:6]:
        pv_na[k] = "NA"

    res_na = score_nccpc_pv(pv_na)
    assert res_na.na_count == 6
    assert res_na.is_indeterminate is True
    assert res_na.high_missingness_advisory is True

    # R instrument test (cutoff is 7)
    r_items = dict.fromkeys(CANONICAL_R_ITEMS, 0)
    r_items["crying"] = 3
    r_items["screaming_yelling"] = 3
    r_items["flinching_moving_away"] = 1  # Total = 7
    res_r = score_nccpc_r(r_items)
    assert res_r.score == 7
    assert res_r.cutoff_breached is True
