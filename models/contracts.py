"""
Project N: Canonical Architectural Dimensions, Constants, and Contracts.
Single source of truth imported by models, feature extractors, and test suites.
"""

from typing import Final

# ==============================================================================
# 1. Acoustic Pipeline Dimensions (SPECS.md §2.1)
# ==============================================================================
AUDIO_SAMPLE_RATE_HZ: Final[int] = 48000
WINDOW_DURATION_SEC: Final[float] = 5.0
RAW_AUDIO_SAMPLES: Final[int] = int(AUDIO_SAMPLE_RATE_HZ * WINDOW_DURATION_SEC)  # 240,000

STFT_WINDOW_N: Final[int] = 2048
STFT_HOP_H: Final[int] = 160
STFT_UNCENTERED_FRAMES: Final[int] = (
    1 + (RAW_AUDIO_SAMPLES - STFT_WINDOW_N) // STFT_HOP_H
)  # 1488 frames
STFT_BIN_WIDTH_HZ: Final[float] = AUDIO_SAMPLE_RATE_HZ / STFT_WINDOW_N  # 23.4375 Hz

PITCH_HOP_MS: Final[int] = 10
PITCH_FRAMES: Final[int] = int((WINDOW_DURATION_SEC * 1000) / PITCH_HOP_MS)  # 500 frames
PITCH_DIMENSIONS: Final[int] = 16  # 16 enumerated acoustic pitch descriptors

CQT_OCTAVES: Final[int] = 7
CQT_BINS_PER_OCTAVE: Final[int] = 12
CQT_TOTAL_BINS: Final[int] = CQT_OCTAVES * CQT_BINS_PER_OCTAVE  # 84 bins
CQT_F_MIN_HZ: Final[float] = 32.70319566257483  # C1
CQT_F_MAX_CENTER_HZ: Final[float] = CQT_F_MIN_HZ * (2 ** (83 / 12))  # ~3951.07 Hz (B7)
CQT_F_MAX_BOUNDARY_HZ: Final[float] = CQT_F_MIN_HZ * (2 ** (84 / 12))  # 4186.01 Hz (C8)

ACOUSTIC_LATENT_D: Final[int] = 512
ACOUSTIC_LATENT_T: Final[int] = 500

# ==============================================================================
# 2. Kinematic Pipeline Dimensions (SPECS.md §2.2)
# ==============================================================================
VIDEO_FPS: Final[int] = 30
VIDEO_FRAMES: Final[int] = int(WINDOW_DURATION_SEC * VIDEO_FPS)  # 150 frames

POSE_LANDMARKS: Final[int] = 33
POSE_COORDS_PER_LANDMARK: Final[int] = 4  # x, y, z, visibility
HAND_KEYPOINTS: Final[int] = 42  # 21 per hand * 2 hands
HAND_COORDS_PER_KEYPOINT: Final[int] = 3  # x, y, z
TOTAL_LANDMARK_COORDS: Final[int] = (POSE_LANDMARKS * POSE_COORDS_PER_LANDMARK) + (
    HAND_KEYPOINTS * HAND_COORDS_PER_KEYPOINT
)  # 132 + 126 = 258

FLOW_PRE_DOWNSCALE_WIDTH: Final[int] = 320
FLOW_PRE_DOWNSCALE_HEIGHT: Final[int] = 180
OPTICAL_FLOW_GRID_X: Final[int] = 8
OPTICAL_FLOW_GRID_Y: Final[int] = 8
OPTICAL_FLOW_DIMS: Final[int] = OPTICAL_FLOW_GRID_X * OPTICAL_FLOW_GRID_Y * 2  # 128

KINEMATIC_LATENT_D: Final[int] = 512
KINEMATIC_LATENT_T: Final[int] = 150

# ==============================================================================
# 3. Clip-to-Window Temporal Aggregation (SPECS.md §2.4)
# ==============================================================================
MIN_CLIP_DURATION_SEC: Final[float] = 30.0
MAX_CLIP_DURATION_SEC: Final[float] = 120.0
MIN_CLIP_WINDOWS: Final[int] = int(MIN_CLIP_DURATION_SEC // WINDOW_DURATION_SEC)  # 6
MAX_CLIP_WINDOWS: Final[int] = int(MAX_CLIP_DURATION_SEC // WINDOW_DURATION_SEC)  # 24

# ==============================================================================
# 4. Physiological & Metric Embedding Dimensions (SPECS.md §2.3, §3.1)
# ==============================================================================
PHYSIOLOGY_STEPS: Final[int] = 50  # 10 Hz over 5.0 seconds
PHYSIOLOGY_LATENT_D: Final[int] = 64
METRIC_EMBEDDING_D: Final[int] = 128

# ==============================================================================
# 5. Controlled Vocabularies (SPECS.md §4.2, §4.3)
# ==============================================================================
CONTROLLED_ACTIONS: Final[set[str]] = {
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

CONTROLLED_ANTECEDENTS: Final[set[str]] = {
    "post_school_transition",
    "mealtime",
    "bedtime_routine",
    "loud_environment",
    "unfamiliar_setting",
    "physical_transition",
    "preferred_activity_ended",
    "unknown",
}

OUTCOME_STATES: Final[set[str]] = {
    "settled_immediately",
    "settled_delayed",
    "no_change",
    "escalated",
}

CHILD_RESPONSES: Final[set[str]] = {
    "reach",
    "gesture",
    "vocal_signal",
    "aac_selection",
    "none",
}

RESPONSE_CHANNELS: Final[set[str]] = {
    "motor",
    "vocal",
    "aac",
    "none",
}

RESPONSE_INDEPENDENCE: Final[set[str]] = {
    "independent",
    "prompted",
    "passive",
    "refusal",
    "none",
}

PERFORMANCE_STATUSES: Final[set[str]] = {
    "completed",
    "attempted_refused",
    "aborted",
    "not_attempted",
}

CAREGIVER_DECISIONS: Final[set[str]] = {
    "accepted",
    "modified",
    "declined",
    "open_observation",
}
