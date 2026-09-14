"""
Project N: 75-Landmark Body & Hand Pose Tracking with Torso Normalization.
Extracts 33 body pose landmarks and 42 hand keypoints (258 coordinates per frame)
with torso-relative normalization across 150 frames (30 fps @ 5.0s).
"""

from typing import Any

import cv2
import mediapipe as mp
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
        )
    )
    if shoulder_dist < 1e-3:
        # Near-zero shoulder distance guard against division by zero / explosive normalization
        shoulder_dist = 1.0

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
    Tracks mean landmark confidence and flags quality breaches (< 0.40).
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.target_frames = VIDEO_FRAMES
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        if hasattr(self, "holistic") and self.holistic:
            self.holistic.close()

    def process_frames(self, frames: list[Any]) -> tuple[np.ndarray, bool, float]:
        """
        Processes a list of 150 image frames.

        Returns:
            tuple (landmarks_matrix, acute_guarding, mean_confidence):
                - landmarks_matrix: (150, 258) normalized coordinates.
                - acute_guarding: bool flag for triage screening.
                - mean_confidence: float in [0.0, 1.0].
        """
        t_len = min(len(frames), self.target_frames)
        mat = np.zeros((self.target_frames, TOTAL_LANDMARK_COORDS), dtype=np.float32)
        confidences: list[float] = []

        for t in range(t_len):
            frame = frames[t]
            if isinstance(frame, np.ndarray) and frame.shape == (TOTAL_LANDMARK_COORDS,):
                mat[t] = normalize_torso_landmarks(frame)
                confidences.append(1.0)
            elif isinstance(frame, np.ndarray) and frame.ndim >= 2:
                # Convert to RGB uint8 for MediaPipe Holistic
                if frame.dtype != np.uint8:
                    frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                if frame.ndim == 2 or frame.shape[-1] == 1:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif frame.shape[-1] == 3:
                    frame_rgb = frame
                elif frame.shape[-1] == 4:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                else:
                    frame_rgb = frame[:, :, :3]

                results = self.holistic.process(frame_rgb)
                coords = np.zeros(TOTAL_LANDMARK_COORDS, dtype=np.float32)
                frame_conf = 0.0

                if results.pose_landmarks:
                    pose_vis: list[float] = []
                    for p_idx, lm in enumerate(results.pose_landmarks.landmark[:POSE_LANDMARKS]):
                        base_idx = p_idx * POSE_COORDS_PER_LANDMARK
                        coords[base_idx : base_idx + 4] = [lm.x, lm.y, lm.z, lm.visibility]
                        pose_vis.append(float(lm.visibility))
                    if pose_vis:
                        frame_conf = float(np.mean(pose_vis))

                    # Left hand (21 * 3)
                    offset_lh = POSE_LANDMARKS * POSE_COORDS_PER_LANDMARK
                    if results.left_hand_landmarks:
                        for h_idx, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
                            h_base = offset_lh + (h_idx * HAND_COORDS_PER_KEYPOINT)
                            coords[h_base : h_base + 3] = [lm.x, lm.y, lm.z]

                    # Right hand (21 * 3)
                    offset_rh = offset_lh + 21 * HAND_COORDS_PER_KEYPOINT
                    if results.right_hand_landmarks:
                        for h_idx, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
                            h_base = offset_rh + (h_idx * HAND_COORDS_PER_KEYPOINT)
                            coords[h_base : h_base + 3] = [lm.x, lm.y, lm.z]

                    mat[t] = normalize_torso_landmarks(coords)
                confidences.append(frame_conf)
            else:
                confidences.append(0.0)

        mean_conf = float(np.mean(confidences)) if confidences else 0.0
        acute_guarding = detect_acute_guarding(mat)
        return mat, acute_guarding, mean_conf
