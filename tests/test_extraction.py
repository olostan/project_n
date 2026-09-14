"""
Project N: Unit Tests for Tier 1 Feature Extraction.
Verifies audio STFT/pitch/CQT pipelines, windowing, pose torso-normalization,
and Farnebäck optical flow using deterministic synthetic signals.
"""

import numpy as np
import pytest

from extraction.audio_cqt import compute_cqt_filterbank
from extraction.audio_pipeline import extract_acoustic_latent
from extraction.audio_pitch import extract_pitch_features
from extraction.audio_stft import compute_log_mel_spectrogram, compute_stft_uncentered
from extraction.kinematic_pipeline import extract_kinematic_latent
from extraction.optical_flow import compute_sequence_optical_flow
from extraction.pose_tracker import PoseTracker, detect_acute_guarding, normalize_torso_landmarks
from extraction.windowing import slice_audio_windows, slice_video_windows, validate_clip_duration
from models.contracts import (
    ACOUSTIC_LATENT_D,
    AUDIO_SAMPLE_RATE_HZ,
    CQT_TOTAL_BINS,
    KINEMATIC_LATENT_D,
    OPTICAL_FLOW_DIMS,
    PITCH_DIMENSIONS,
    PITCH_FRAMES,
    RAW_AUDIO_SAMPLES,
    STFT_UNCENTERED_FRAMES,
    TOTAL_LANDMARK_COORDS,
    VIDEO_FRAMES,
)


def test_windowing_audio_and_video() -> None:
    # 60.0s audio clip = 2,880,000 samples -> exactly 12 windows of 240,000
    duration_sec = 60.0
    w_count = validate_clip_duration(duration_sec)
    assert w_count == 12

    audio_60s = np.zeros(int(duration_sec * AUDIO_SAMPLE_RATE_HZ), dtype=np.float32)
    windows = slice_audio_windows(audio_60s)
    assert len(windows) == 12
    for w in windows:
        assert len(w) == RAW_AUDIO_SAMPLES

    # 60.0s video = 1800 frames -> 12 windows of 150 frames
    frames_60s = [np.zeros((180, 320), dtype=np.uint8) for _ in range(1800)]
    v_windows = slice_video_windows(frames_60s)
    assert len(v_windows) == 12
    for vw in v_windows:
        assert len(vw) == VIDEO_FRAMES


def test_audio_stft_and_log_mel() -> None:
    # Synthetic 440 Hz pure tone (5.0s @ 48 kHz)
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    stft_mag = compute_stft_uncentered(synthetic_audio)
    assert stft_mag.shape == (STFT_UNCENTERED_FRAMES, 1025)

    log_mel = compute_log_mel_spectrogram(synthetic_audio)
    assert log_mel.shape == (PITCH_FRAMES, 128)
    assert not np.isnan(log_mel).any()


def test_audio_pitch_features() -> None:
    # Synthetic 250 Hz tone with Gaussian noise
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False)
    synthetic_audio = (
        0.6 * np.sin(2 * np.pi * 250.0 * t) + 0.05 * np.random.randn(RAW_AUDIO_SAMPLES)
    ).astype(np.float32)

    pitch_mat = extract_pitch_features(synthetic_audio)
    assert pitch_mat.shape == (PITCH_FRAMES, PITCH_DIMENSIONS)
    assert not np.isnan(pitch_mat).any()

    # F0 should detect pitch around 250 Hz for voiced frames
    detected_f0 = pitch_mat[:, 0][pitch_mat[:, 0] > 0]
    assert len(detected_f0) > 0
    assert pytest.approx(np.mean(detected_f0), abs=15.0) == 250.0


def test_audio_cqt_filterbank() -> None:
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 500.0 * t).astype(np.float32)

    cqt_mat = compute_cqt_filterbank(synthetic_audio)
    assert cqt_mat.shape == (PITCH_FRAMES, CQT_TOTAL_BINS)
    assert not np.isnan(cqt_mat).any()


def test_audio_pipeline_extraction() -> None:
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False)
    synthetic_audio = 0.5 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)

    x_audio, metrics = extract_acoustic_latent(synthetic_audio)
    assert x_audio.shape == (PITCH_FRAMES, ACOUSTIC_LATENT_D)
    assert "f0_mean_hz" in metrics
    assert "cpp_db" in metrics
    assert not np.isnan(x_audio).any()


def test_pose_normalization_and_guarding() -> None:
    # Create synthetic 258 landmark coordinates
    coords = np.zeros(TOTAL_LANDMARK_COORDS, dtype=np.float32)
    # Set shoulders and hips to establish frame:
    # Left shoulder (11): x=-0.2, y=0.5, z=0
    # Right shoulder (12): x=0.2, y=0.5, z=0
    # Left hip (23): x=-0.15, y=0.0, z=0
    # Right hip (24): x=0.15, y=0.0, z=0
    coords[11 * 4 : 11 * 4 + 3] = [-0.2, 0.5, 0.0]
    coords[12 * 4 : 12 * 4 + 3] = [0.2, 0.5, 0.0]
    coords[23 * 4 : 23 * 4 + 3] = [-0.15, 0.0, 0.0]
    coords[24 * 4 : 24 * 4 + 3] = [0.15, 0.0, 0.0]

    normalized = normalize_torso_landmarks(coords)
    assert len(normalized) == TOTAL_LANDMARK_COORDS
    assert not np.isnan(normalized).any()

    # Guarding test: sequence where wrist retracts abruptly
    seq = np.zeros((10, TOTAL_LANDMARK_COORDS), dtype=np.float32)
    for i in range(5):
        seq[i, 15 * 4 : 15 * 4 + 2] = [0.8, 0.8]  # Extended arm
    for i in range(5, 10):
        seq[i, 15 * 4 : 15 * 4 + 2] = [0.1, 0.1]  # Sudden sharp contraction

    assert detect_acute_guarding(seq)


def test_optical_flow_sequence() -> None:
    # Generate 10 synthetic frames with moving horizontal bar
    frames: list[np.ndarray] = []
    for i in range(10):
        img = np.zeros((180, 320), dtype=np.uint8)
        img[:, i * 10 : i * 10 + 20] = 255
        frames.append(img)

    flow_seq = compute_sequence_optical_flow(frames, target_frames=10)
    assert flow_seq.shape == (10, OPTICAL_FLOW_DIMS)
    assert flow_seq[0].sum() == 0.0  # Frame 0 is padded with zero
    assert not np.isnan(flow_seq).any()


def test_kinematic_pipeline_extraction() -> None:
    frames: list[np.ndarray] = [np.zeros((180, 320), dtype=np.uint8) for _ in range(VIDEO_FRAMES)]
    tracker = PoseTracker()

    x_kinematic, metrics = extract_kinematic_latent(frames, tracker=tracker)
    assert x_kinematic.shape == (VIDEO_FRAMES, KINEMATIC_LATENT_D)
    assert "motion_rhythm_hz" in metrics
    assert "acute_guarding_detected" in metrics
    assert not np.isnan(x_kinematic).any()
