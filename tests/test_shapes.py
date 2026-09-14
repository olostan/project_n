"""
Project N: Automated Tensor Shapes, Contracts, and Clinical Instrument Assertion Test Suite.
Verifies mathematical dimensional integrity and psychometric fidelity across SPECS.md and DESIGN.md.
"""

# ruff: noqa: E402
import math
import sys
from pathlib import Path

import numpy as np

# Add repository root to python path for standalone execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.contracts import (
    ACOUSTIC_LATENT_D,
    ACOUSTIC_LATENT_T,
    CAREGIVER_DECISIONS,
    CONTROLLED_ACTIONS,
    CONTROLLED_ANTECEDENTS,
    CQT_BINS_PER_OCTAVE,
    CQT_F_MAX_BOUNDARY_HZ,
    CQT_F_MAX_CENTER_HZ,
    CQT_F_MIN_HZ,
    CQT_OCTAVES,
    CQT_TOTAL_BINS,
    FLOW_PRE_DOWNSCALE_HEIGHT,
    FLOW_PRE_DOWNSCALE_WIDTH,
    HAND_COORDS_PER_KEYPOINT,
    HAND_KEYPOINTS,
    KINEMATIC_LATENT_D,
    KINEMATIC_LATENT_T,
    MAX_CLIP_DURATION_SEC,
    MAX_CLIP_WINDOWS,
    METRIC_EMBEDDING_D,
    MIN_CLIP_DURATION_SEC,
    MIN_CLIP_WINDOWS,
    OPTICAL_FLOW_DIMS,
    OPTICAL_FLOW_GRID_X,
    OPTICAL_FLOW_GRID_Y,
    PERFORMANCE_STATUSES,
    PHYSIOLOGY_LATENT_D,
    PHYSIOLOGY_STEPS,
    PITCH_DIMENSIONS,
    PITCH_FRAMES,
    POSE_COORDS_PER_LANDMARK,
    POSE_LANDMARKS,
    RAW_AUDIO_SAMPLES,
    STFT_BIN_WIDTH_HZ,
    STFT_UNCENTERED_FRAMES,
    TOTAL_LANDMARK_COORDS,
    VIDEO_FPS,
    VIDEO_FRAMES,
    WINDOW_DURATION_SEC,
)
from models.nccpc import (
    CANONICAL_PV_ITEMS,
    CANONICAL_R_ITEMS,
    NCCPC_PV_SUBSCALES,
    PV_CUTOFF_SCORE,
    R_CUTOFF_SCORE,
    score_nccpc_pv,
    score_nccpc_r,
)


def test_audio_pipeline_shapes() -> None:
    assert RAW_AUDIO_SAMPLES == 240000
    assert STFT_UNCENTERED_FRAMES == 1488
    assert math.isclose(STFT_BIN_WIDTH_HZ, 23.4375)
    assert PITCH_FRAMES == 500
    assert PITCH_DIMENSIONS == 16
    assert CQT_TOTAL_BINS == 84
    assert CQT_OCTAVES == 7
    assert CQT_BINS_PER_OCTAVE == 12
    assert math.isclose(CQT_F_MIN_HZ, 32.703195, rel_tol=1e-4)
    assert math.isclose(CQT_F_MAX_CENTER_HZ, 3951.07, rel_tol=1e-3)
    assert math.isclose(CQT_F_MAX_BOUNDARY_HZ, 4186.01, rel_tol=1e-3)
    assert ACOUSTIC_LATENT_T == 500
    assert ACOUSTIC_LATENT_D == 512


def test_kinematic_pipeline_shapes() -> None:
    assert VIDEO_FRAMES == 150
    assert VIDEO_FPS == 30
    assert POSE_LANDMARKS == 33
    assert POSE_COORDS_PER_LANDMARK == 4
    assert HAND_KEYPOINTS == 42
    assert HAND_COORDS_PER_KEYPOINT == 3
    assert TOTAL_LANDMARK_COORDS == 258
    assert FLOW_PRE_DOWNSCALE_WIDTH == 320
    assert FLOW_PRE_DOWNSCALE_HEIGHT == 180
    assert OPTICAL_FLOW_GRID_X == 8
    assert OPTICAL_FLOW_GRID_Y == 8
    assert OPTICAL_FLOW_DIMS == 128
    assert KINEMATIC_LATENT_T == 150
    assert KINEMATIC_LATENT_D == 512


def test_clip_to_window_aggregation_shapes() -> None:
    assert MIN_CLIP_DURATION_SEC == 30.0
    assert MAX_CLIP_DURATION_SEC == 120.0
    assert WINDOW_DURATION_SEC == 5.0
    assert MIN_CLIP_WINDOWS == 6
    assert MAX_CLIP_WINDOWS == 24

    # Positional magnitude decoupling invariant:
    # L2-normalized window vectors have norm 1.0; sinusoidal encodings have norm sqrt(128/2) = 8.0.
    # Positional encodings must be used for attention scoring ONLY, not added into the pooled value.
    w_count = 12
    d = METRIC_EMBEDDING_D
    # Random normalized window vectors
    z_w = np.random.randn(w_count, d)
    z_w = z_w / np.linalg.norm(z_w, axis=-1, keepdims=True)
    assert np.allclose(np.linalg.norm(z_w, axis=-1), 1.0)

    # Standard sinusoidal positional encodings
    pos = np.arange(w_count)[:, None]
    div_term = np.exp(np.arange(0, d, 2) * -(np.log(10000.0) / d))
    p_w = np.zeros((w_count, d))
    p_w[:, 0::2] = np.sin(pos * div_term)
    p_w[:, 1::2] = np.cos(pos * div_term)

    # Magnitude assertion: ||p_w|| = sqrt(128/2) = 8.0
    p_norms = np.linalg.norm(p_w, axis=-1)
    assert np.allclose(p_norms, np.sqrt(d / 2.0))

    # Attention scoring with (z_w + p_w)
    q_clip = np.random.randn(d) * 0.02
    scores = np.dot(z_w + p_w, q_clip) / np.sqrt(d)
    alpha = np.exp(scores - np.max(scores))
    alpha = alpha / np.sum(alpha)

    # Value pooling aggregates pure sensory vectors z_w (without p_w):
    z_clip_raw = np.sum(alpha[:, None] * z_w, axis=0)
    z_clip = z_clip_raw / np.linalg.norm(z_clip_raw)
    assert z_clip.shape == (d,)
    assert np.isclose(np.linalg.norm(z_clip), 1.0)


def test_physiology_pipeline_shapes() -> None:
    assert PHYSIOLOGY_STEPS == 50
    assert PHYSIOLOGY_LATENT_D == 64


def test_metric_space_shapes() -> None:
    assert METRIC_EMBEDDING_D == 128


def test_controlled_vocabularies() -> None:
    assert len(CONTROLLED_ACTIONS) == 12
    assert "open_observation" in CONTROLLED_ACTIONS
    assert "deep_pressure_proprioceptive" in CONTROLLED_ACTIONS
    assert "other_custom" in CONTROLLED_ACTIONS

    assert len(CONTROLLED_ANTECEDENTS) == 8
    assert "post_school_transition" in CONTROLLED_ANTECEDENTS
    assert "loud_environment" in CONTROLLED_ANTECEDENTS
    assert "unknown" in CONTROLLED_ANTECEDENTS

    assert len(PERFORMANCE_STATUSES) == 4
    assert "completed" in PERFORMANCE_STATUSES
    assert "attempted_refused" in PERFORMANCE_STATUSES
    assert "aborted" in PERFORMANCE_STATUSES
    assert "not_attempted" in PERFORMANCE_STATUSES

    assert len(CAREGIVER_DECISIONS) == 4
    assert "accepted" in CAREGIVER_DECISIONS
    assert "modified" in CAREGIVER_DECISIONS
    assert "declined" in CAREGIVER_DECISIONS
    assert "open_observation" in CAREGIVER_DECISIONS


def test_nccpc_instrument_specifications() -> None:
    # Subscale item counts per Breau et al. (2002)
    assert len(NCCPC_PV_SUBSCALES["vocal"]) == 4
    assert len(NCCPC_PV_SUBSCALES["social"]) == 4
    assert len(NCCPC_PV_SUBSCALES["facial"]) == 5
    assert len(NCCPC_PV_SUBSCALES["activity"]) == 2
    assert len(NCCPC_PV_SUBSCALES["body_and_limbs"]) == 6
    assert len(NCCPC_PV_SUBSCALES["physiological"]) == 6

    assert len(CANONICAL_PV_ITEMS) == 27
    assert len(CANONICAL_R_ITEMS) == 30
    assert CANONICAL_R_ITEMS.issuperset(CANONICAL_PV_ITEMS)

    # Key clinical items that must be present
    assert "floppy" in CANONICAL_PV_ITEMS
    assert "lips_puckering_tight_pouting_quivering" in CANONICAL_PV_ITEMS
    assert "not_moving_less_active_quiet" in CANONICAL_PV_ITEMS
    assert "trembling_shaking" not in CANONICAL_PV_ITEMS  # confirmed absent in canonical PV

    # Test PV Scoring
    mock_pv_zeros = dict.fromkeys(CANONICAL_PV_ITEMS, 0)
    res_zero = score_nccpc_pv(mock_pv_zeros)
    assert res_zero.score == 0
    assert not res_zero.cutoff_breached

    mock_pv_cutoff = dict.fromkeys(CANONICAL_PV_ITEMS, 0)
    mock_pv_cutoff["crying"] = 3
    mock_pv_cutoff["screaming_yelling"] = 3
    mock_pv_cutoff["stiff_spastic_rigid_tense"] = 3
    mock_pv_cutoff["sharp_breath_gasping"] = 2  # Total = 11
    res_cutoff = score_nccpc_pv(mock_pv_cutoff)
    assert res_cutoff.score == 11
    assert res_cutoff.cutoff_threshold == PV_CUTOFF_SCORE
    assert res_cutoff.cutoff_breached

    # Test "NA" handling
    mock_pv_na: dict[str, int | str] = {
        k: "NA" if i % 2 == 0 else 1 for i, k in enumerate(sorted(CANONICAL_PV_ITEMS))
    }
    res_na = score_nccpc_pv(mock_pv_na)
    assert res_na.na_count == 14
    assert res_na.high_missingness_advisory is True

    # Test R Scoring
    mock_r_zeros = dict.fromkeys(CANONICAL_R_ITEMS, 0)
    res_r_zero = score_nccpc_r(mock_r_zeros)
    assert res_r_zero.score == 0
    assert not res_r_zero.cutoff_breached
    assert res_r_zero.cutoff_threshold == R_CUTOFF_SCORE


if __name__ == "__main__":
    test_audio_pipeline_shapes()
    test_kinematic_pipeline_shapes()
    test_clip_to_window_aggregation_shapes()
    test_physiology_pipeline_shapes()
    test_metric_space_shapes()
    test_controlled_vocabularies()
    test_nccpc_instrument_specifications()
    print("All shape, dimension, vocabulary, and clinical instrument assertions PASSED!")
