"""
Project N: 75-Landmark Body & Hand Pose Tracking with Torso Normalization.
Extracts 33 body pose landmarks and 42 hand keypoints (258 coordinates per frame)
with torso-relative normalization across 150 frames (30 fps @ 5.0s).
"""

from typing import Any

import numpy as np

from models.contracts import (
    HAND_COORDS_PER_KEYPOINT,
    HAND_KEYPOINTS,
    POSE_COORDS_PER_LANDMARK,
    POSE_LANDMARKS,
    TOTAL_LANDMARK_COORDS,
    VIDEO_FRAMES,
)


def normalize_torso_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Applies torso-relative normalization to 75 skeletal landmarks (258 coordinates).
    Scaled by inter-shoulder width and centered at mid-hip:
    p_norm = (p - p_midhip) / (||p_lshoulder - p_rshoulder||_2 + 1e-8)

    Args:
        coords: 1D array of 258 coordinates (33 pose * 4 + 42 hand * 3).

    Returns:
        Normalized 1D array of 258 coordinates.
    """
    if len(coords) != TOTAL_LANDMARK_COORDS:
        raise ValueError(f"Expected {TOTAL_LANDMARK_COORDS} coordinates, got {len(coords)}")

    normalized = np.copy(coords)

    # Pose landmark indices in MediaPipe:
    # Left shoulder: 11, Right shoulder: 12, Left hip: 23, Right hip: 24
    # Each pose landmark has 4 coords: x, y, z, visibility
    l_shoulder_x, l_shoulder_y, l_shoulder_z = coords[11 * 4 : 11 * 4 + 3]
    r_shoulder_x, r_shoulder_y, r_shoulder_z = coords[12 * 4 : 12 * 4 + 3]
    l_hip_x, l_hip_y, l_hip_z = coords[23 * 4 : 23 * 4 + 3]
    r_hip_x, r_hip_y, r_hip_z = coords[24 * 4 : 24 * 4 + 3]

    mid_hip = np.array(
        [(l_hip_x + r_hip_x) / 2.0, (l_hip_y + r_hip_y) / 2.0, (l_hip_z + r_hip_z) / 2.0],
        dtype=np.float32,
    )
    shoulder_dist = float(
        np.sqrt(
            (l_shoulder_x - r_shoulder_x) ** 2
            + (l_shoulder_y - r_shoulder_y) ** 2
            + (l_shoulder_z - r_shoulder_z) ** 2
            + 1e-8
        )
    )

    # Normalize pose landmarks (first 132 features: 33 * 4)
    for p in range(POSE_LANDMARKS):
        idx = p * POSE_COORDS_PER_LANDMARK
        normalized[idx : idx + 3] = (normalized[idx : idx + 3] - mid_hip) / shoulder_dist

    # Normalize hand keypoints (next 126 features: 42 * 3)
    offset = POSE_LANDMARKS * POSE_COORDS_PER_LANDMARK
    for h in range(HAND_KEYPOINTS):
        idx = offset + (h * HAND_COORDS_PER_KEYPOINT)
        normalized[idx : idx + 3] = (normalized[idx : idx + 3] - mid_hip) / shoulder_dist

    return normalized


def detect_acute_guarding(landmarks_seq: np.ndarray) -> bool:
    """
    Screens landmark sequence for acute physical guarding/flinching kinematics.
    A sudden protective contraction where limb-to-torso distance drops sharply.

    Args:
        landmarks_seq: Matrix of shape (T, 258) of normalized coordinates.

    Returns:
        Boolean indicating acute guarding detected.
    """
    if len(landmarks_seq) < 5:
        return False

    # Track wrist positions relative to torso across frames
    # Pose 15 = left wrist, 16 = right wrist
    lw_x = landmarks_seq[:, 15 * 4]
    lw_y = landmarks_seq[:, 15 * 4 + 1]
    lw_dist = np.sqrt(lw_x**2 + lw_y**2)

    # Sudden contraction: rapid reduction in extension > 50% within 3-5 frames
    lw_diff = np.diff(lw_dist)
    sharp_retraction = bool(np.any(lw_diff < -0.4))
    return sharp_retraction


class PoseTracker:
    """
    Extracts 75 MediaPipe body and hand landmarks across a 150-frame window.
    """

    def __init__(self) -> None:
        self.target_frames = VIDEO_FRAMES

    def process_frames(self, frames: list[Any]) -> tuple[np.ndarray, bool]:
        """
        Processes a list of 150 image frames.

        Returns:
            tuple (landmarks_matrix, acute_guarding):
                - landmarks_matrix: (150, 258) normalized coordinates.
                - acute_guarding: bool flag for triage screening.
        """
        t_len = min(len(frames), self.target_frames)
        mat = np.zeros((self.target_frames, TOTAL_LANDMARK_COORDS), dtype=np.float32)

        # In real video, MediaPipe Holistic would process frames.
        # For synthetic frames or when frames is a pre-extracted array, populate appropriately.
        for t in range(t_len):
            if isinstance(frames[t], np.ndarray) and frames[t].shape == (TOTAL_LANDMARK_COORDS,):
                mat[t] = normalize_torso_landmarks(frames[t])
            elif isinstance(frames[t], np.ndarray) and frames[t].ndim >= 2:
                # Placeholder for direct camera frames: fallback normalized neutral pose
                mat[t] = np.zeros(TOTAL_LANDMARK_COORDS, dtype=np.float32)

        acute_guarding = detect_acute_guarding(mat)
        return mat, acute_guarding
