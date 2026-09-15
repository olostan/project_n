"""
Project N: Realistic Kinematic / Video Test Fixtures.
Provides synthetic skeletal landmark sequences and image frames with calibrated
confidence and optical flow properties for pipeline testing.
"""

import numpy as np

from models.contracts import (
    POSE_COORDS_PER_LANDMARK,
    POSE_LANDMARKS,
    TOTAL_LANDMARK_COORDS,
    VIDEO_FRAMES,
)


def generate_valid_mock_video_frames(
    num_frames: int = VIDEO_FRAMES,
    confidence: float = 0.90,
) -> list[np.ndarray]:
    """
    Generates video frames as mock landmark coordinates with high confidence (>= 0.40)
    and zero ego-motion, satisfying fail-closed signal quality gates.
    """
    frames: list[np.ndarray] = []
    for _ in range(num_frames):
        coords = np.zeros(TOTAL_LANDMARK_COORDS, dtype=np.float32)
        # Populate pose landmarks with coordinates and visibility
        for p in range(POSE_LANDMARKS):
            idx = p * POSE_COORDS_PER_LANDMARK
            coords[idx : idx + 3] = [0.5, 0.5, 0.0]
            coords[idx + 3] = confidence

        # Torso landmarks to prevent degenerate shoulder distance
        coords[11 * 4 : 11 * 4 + 2] = [0.4, 0.4]  # left shoulder
        coords[12 * 4 : 12 * 4 + 2] = [0.6, 0.4]  # right shoulder
        coords[23 * 4 : 23 * 4 + 2] = [0.45, 0.7]  # left hip
        coords[24 * 4 : 24 * 4 + 2] = [0.55, 0.7]  # right hip
        frames.append(coords)

    return frames
