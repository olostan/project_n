"""
Project N: Automated Tensor Shapes and Contract Assertion Test Suite.
Verifies mathematical dimensional integrity across SPECS.md and DESIGN.md.
"""

import math

# Canonical Architecture Dimensions (SPECS.md Section 2 & 3)
AUDIO_SAMPLE_RATE_HZ = 48000
WINDOW_DURATION_SEC = 5.0
RAW_AUDIO_SAMPLES = int(AUDIO_SAMPLE_RATE_HZ * WINDOW_DURATION_SEC)  # 240,000

STFT_WINDOW_N = 2048
STFT_HOP_H = 160
STFT_UNCENTERED_FRAMES = 1 + (RAW_AUDIO_SAMPLES - STFT_WINDOW_N) // STFT_HOP_H  # 1488
STFT_BIN_WIDTH_HZ = AUDIO_SAMPLE_RATE_HZ / STFT_WINDOW_N  # 23.4375

PITCH_HOP_MS = 10
PITCH_FRAMES = int((WINDOW_DURATION_SEC * 1000) / PITCH_HOP_MS)  # 500
PITCH_DIMENSIONS = 16

CQT_OCTAVES = 7
CQT_BINS_PER_OCTAVE = 12
CQT_TOTAL_BINS = CQT_OCTAVES * CQT_BINS_PER_OCTAVE  # 84
CQT_F_MIN_HZ = 32.70319566257483  # C1
CQT_F_MAX_CENTER_HZ = CQT_F_MIN_HZ * (2 ** (83 / 12))  # ~3951.07 Hz (B7)
CQT_F_MAX_BOUNDARY_HZ = CQT_F_MIN_HZ * (2 ** (84 / 12))  # 4186.01 Hz (C8)

ACOUSTIC_LATENT_D = 512
ACOUSTIC_LATENT_T = 500

VIDEO_FPS = 30
VIDEO_FRAMES = int(WINDOW_DURATION_SEC * VIDEO_FPS)  # 150
POSE_LANDMARKS = 33
POSE_COORDS_PER_LANDMARK = 4  # x, y, z, visibility
HAND_KEYPOINTS = 42  # 21 per hand * 2
HAND_COORDS_PER_KEYPOINT = 3  # x, y, z
TOTAL_LANDMARK_COORDS = (POSE_LANDMARKS * POSE_COORDS_PER_LANDMARK) + (
    HAND_KEYPOINTS * HAND_COORDS_PER_KEYPOINT
)  # 132 + 126 = 258

OPTICAL_FLOW_GRID_X = 8
OPTICAL_FLOW_GRID_Y = 8
OPTICAL_FLOW_DIMS = OPTICAL_FLOW_GRID_X * OPTICAL_FLOW_GRID_Y * 2  # 128
KINEMATIC_LATENT_D = 512
KINEMATIC_LATENT_T = 150

PHYSIOLOGY_STEPS = 50
PHYSIOLOGY_LATENT_D = 64

METRIC_EMBEDDING_D = 128

CONTROLLED_ACTIONS = {
    "quiet_refuge",
    "dimmed_lighting",
    "hydration_water",
    "deep_pressure_proprioceptive",
    "vestibular_rocking",
    "preferred_comfort_object",
    "sensory_break",
    "motor_movement_break",
    "warm_compress",
    "aac_choice_board",
    "open_observation",
    "other_custom",
}

CANONICAL_PV_ITEMS = {
    "moaning_whining_whimpering",
    "crying",
    "screaming_yelling",
    "pain_sound_word",
    "not_cooperating_cranky",
    "less_interaction_withdrawn",
    "seeking_comfort_closeness",
    "difficult_to_comfort_please",
    "furrowed_brow",
    "change_in_eyes_squinching",
    "turning_mouth_down",
    "clenching_teeth_chewing",
    "not_moving_less_active",
    "jumping_agitated_fidgety",
    "less_active_quiet",
    "stiff_spastic_rigid_tense",
    "gesturing_touching_hurt_part",
    "protecting_favoring_guarding",
    "flinching_moving_away",
    "moving_body_specifically",
    "trembling_shaking",
    "shivering",
    "change_in_color_pallor",
    "sweating_perspiring",
    "tears",
    "sharp_breath_gasping",
    "breath_holding",
}

CANONICAL_R_EXTRA_ITEMS = {
    "eating_less_not_interested",
    "increased_sleep",
    "decreased_sleep",
}
CANONICAL_R_ITEMS = CANONICAL_PV_ITEMS | CANONICAL_R_EXTRA_ITEMS


def test_audio_pipeline_shapes() -> None:
    assert RAW_AUDIO_SAMPLES == 240000
    assert STFT_UNCENTERED_FRAMES == 1488
    assert math.isclose(STFT_BIN_WIDTH_HZ, 23.4375)
    assert PITCH_FRAMES == 500
    assert PITCH_DIMENSIONS == 16
    assert CQT_TOTAL_BINS == 84
    assert math.isclose(CQT_F_MAX_CENTER_HZ, 3951.07, rel_tol=1e-3)
    assert math.isclose(CQT_F_MAX_BOUNDARY_HZ, 4186.01, rel_tol=1e-3)
    assert ACOUSTIC_LATENT_T == 500
    assert ACOUSTIC_LATENT_D == 512


def test_kinematic_pipeline_shapes() -> None:
    assert VIDEO_FRAMES == 150
    assert TOTAL_LANDMARK_COORDS == 258
    assert OPTICAL_FLOW_DIMS == 128
    assert KINEMATIC_LATENT_T == 150
    assert KINEMATIC_LATENT_D == 512


def test_physiology_pipeline_shapes() -> None:
    assert PHYSIOLOGY_STEPS == 50
    assert PHYSIOLOGY_LATENT_D == 64


def test_metric_space_shapes() -> None:
    assert METRIC_EMBEDDING_D == 128


def test_controlled_actions_vocabulary() -> None:
    assert len(CONTROLLED_ACTIONS) == 12
    assert "open_observation" in CONTROLLED_ACTIONS
    assert "deep_pressure_proprioceptive" in CONTROLLED_ACTIONS
    assert "other_custom" in CONTROLLED_ACTIONS


def test_nccpc_instrument_specifications() -> None:
    assert len(CANONICAL_PV_ITEMS) == 27
    assert len(CANONICAL_R_ITEMS) == 30
    assert CANONICAL_R_ITEMS.issuperset(CANONICAL_PV_ITEMS)
    assert len(CANONICAL_R_EXTRA_ITEMS) == 3


if __name__ == "__main__":
    test_audio_pipeline_shapes()
    test_kinematic_pipeline_shapes()
    test_physiology_pipeline_shapes()
    test_metric_space_shapes()
    test_controlled_actions_vocabulary()
    test_nccpc_instrument_specifications()
    print("All shape, dimension, vocabulary, and instrument assertions PASSED!")
