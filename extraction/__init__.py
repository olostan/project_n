"""
Project N: Extraction Package (Tier 1).
Stateless signal processors for audio and kinematic feature extraction.
"""

from extraction.audio_cqt import compute_cqt_filterbank
from extraction.audio_pipeline import extract_acoustic_latent
from extraction.audio_pitch import extract_pitch_features
from extraction.audio_stft import compute_log_mel_spectrogram, compute_stft_uncentered
from extraction.kinematic_pipeline import extract_kinematic_latent
from extraction.optical_flow import compute_frame_pair_flow, compute_sequence_optical_flow
from extraction.pose_tracker import PoseTracker, detect_acute_guarding, normalize_torso_landmarks
from extraction.windowing import slice_audio_windows, slice_video_windows, validate_clip_duration

__all__ = [
    "compute_cqt_filterbank",
    "compute_frame_pair_flow",
    "compute_log_mel_spectrogram",
    "compute_sequence_optical_flow",
    "compute_stft_uncentered",
    "detect_acute_guarding",
    "extract_acoustic_latent",
    "extract_kinematic_latent",
    "extract_pitch_features",
    "normalize_torso_landmarks",
    "slice_audio_windows",
    "slice_video_windows",
    "validate_clip_duration",
    "PoseTracker",
]
