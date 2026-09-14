"""
Project N: Unified Kinematic Extraction Pipeline.
Fuses torso-normalized skeletal landmarks (258 coords) and pooled optical flow (128 dims)
into an aligned kinematic latent representation X_kinematic in R^(150, 512) and motion metrics.
"""

from typing import Any

import numpy as np

from extraction.optical_flow import compute_sequence_optical_flow
from extraction.pose_tracker import PoseTracker
from models.contracts import (
    KINEMATIC_LATENT_D,
    OPTICAL_FLOW_DIMS,
    TOTAL_LANDMARK_COORDS,
    VIDEO_FPS,
    VIDEO_FRAMES,
)


def extract_kinematic_latent(
    frames: list[np.ndarray],
    tracker: PoseTracker | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extracts unified kinematic latent representation X_kinematic and summary metrics.

    Args:
        frames: List of 150 video frames (30 fps @ 5.0s).
        tracker: Optional PoseTracker instance.

    Returns:
        tuple (x_kinematic, metrics):
            - x_kinematic: (150, 512) float32 numpy array.
            - metrics: dictionary containing measured motion indicators:
                       motion_rhythm_hz, acute_guarding_detected, mean_flow_velocity.
    """
    if tracker is None:
        tracker = PoseTracker()

    # 1. Pose landmarks & guarding detection (150, 258)
    landmarks_seq, acute_guarding, mean_confidence = tracker.process_frames(frames)

    # 2. Optical flow sequence (150, 128)
    flow_seq = compute_sequence_optical_flow(frames, target_frames=VIDEO_FRAMES)

    # 3. Concatenate features: 258 + 128 = 386
    combined_raw = np.concatenate([landmarks_seq, flow_seq], axis=-1)  # (150, 386)

    # 4. Project / pad to canonical KINEMATIC_LATENT_D = 512
    x_kinematic = np.zeros((VIDEO_FRAMES, KINEMATIC_LATENT_D), dtype=np.float32)
    x_kinematic[:, : TOTAL_LANDMARK_COORDS + OPTICAL_FLOW_DIMS] = combined_raw

    # 5. Compute motion rhythmicity via FFT over optical flow magnitude
    flow_mag = np.linalg.norm(flow_seq.reshape(VIDEO_FRAMES, -1, 2), axis=-1).mean(axis=-1)
    flow_mag_detrend = flow_mag - np.mean(flow_mag)
    spec = np.abs(np.fft.rfft(flow_mag_detrend))
    freqs = np.fft.rfftfreq(VIDEO_FRAMES, d=1.0 / VIDEO_FPS)

    # Look for rhythmicity peak in 1.0 - 8.0 Hz (covering typical 3-6 Hz stimming)
    stim_band = (freqs >= 1.0) & (freqs <= 8.0)
    if np.any(stim_band) and np.max(spec[stim_band]) > 0.1:
        peak_idx = np.argmax(spec[stim_band])
        rhythm_hz = float(freqs[stim_band][peak_idx])
    else:
        rhythm_hz = 0.0

    mean_velocity = float(np.mean(flow_mag))

    metrics = {
        "motion_rhythm_hz": round(rhythm_hz, 2),
        "acute_guarding_detected": acute_guarding,
        "mean_flow_velocity": round(mean_velocity, 2),
        "mean_pose_confidence": round(mean_confidence, 2),
    }

    return x_kinematic, metrics
