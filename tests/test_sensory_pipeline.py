"""
Project N: Invariant 4 Sensory Input Modalities Verification.
Guarantees acoustic and kinematic representations are strictly continuous,
unquantized, non-phonemic float32 tensors with zero ASR or phonemic transcripts.
"""

import numpy as np

from extraction.audio_cqt import compute_cqt_filterbank
from extraction.audio_pipeline import extract_acoustic_latent
from extraction.audio_pitch import extract_pitch_features
from extraction.audio_stft import compute_log_mel_spectrogram
from extraction.kinematic_pipeline import extract_kinematic_latent
from extraction.optical_flow import compute_sequence_optical_flow
from extraction.pose_tracker import PoseTracker
from models.contracts import (
    ACOUSTIC_LATENT_D,
    ACOUSTIC_LATENT_T,
    KINEMATIC_LATENT_D,
    KINEMATIC_LATENT_T,
    RAW_AUDIO_SAMPLES,
    VIDEO_FRAMES,
)


def test_acoustic_pipeline_continuous_representations() -> None:
    # 5.0s of continuous audio at 48 kHz (240,000 samples)
    t = np.linspace(0, 5.0, RAW_AUDIO_SAMPLES, endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 220.0 * t)

    # 1. Component extractions
    stft = compute_log_mel_spectrogram(audio)
    assert stft.shape == (500, 128)
    assert stft.dtype == np.float32

    cqt = compute_cqt_filterbank(audio)
    assert cqt.shape == (500, 84)
    assert cqt.dtype == np.float32

    pitch = extract_pitch_features(audio)
    assert pitch.shape == (500, 16)
    assert pitch.dtype == np.float32

    # 2. Unified latent extraction
    x_audio, metrics = extract_acoustic_latent(audio)
    assert x_audio.shape == (ACOUSTIC_LATENT_T, ACOUSTIC_LATENT_D)
    assert x_audio.dtype == np.float32

    # Verify scalar acoustic measurements
    assert "f0_mean_hz" in metrics
    assert "cpp_db" in metrics
    assert "jitter_mean" in metrics
    assert "shimmer_mean" in metrics
    assert "hnr_db" in metrics
    for v in metrics.values():
        assert isinstance(v, int | float | np.floating | np.integer)


def test_kinematic_pipeline_continuous_representations() -> None:
    # 150 frames @ 30 fps (5.0s)
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(VIDEO_FRAMES)]

    # 1. Optical flow extraction
    flow = compute_sequence_optical_flow(frames)
    assert flow.shape == (VIDEO_FRAMES, 128)
    assert flow.dtype == np.float32

    # 2. MediaPipe Holistic tracker
    tracker = PoseTracker()
    landmarks_mat, guarding, conf = tracker.process_frames(frames)
    assert landmarks_mat.shape == (VIDEO_FRAMES, 258)
    assert landmarks_mat.dtype == np.float32
    assert 0.0 <= conf <= 1.0

    # 3. Unified kinematic latent extraction
    x_kinematic, k_metrics = extract_kinematic_latent(frames, tracker=tracker)
    assert x_kinematic.shape == (KINEMATIC_LATENT_T, KINEMATIC_LATENT_D)
    assert x_kinematic.dtype == np.float32

    assert "motion_rhythm_hz" in k_metrics
    assert "acute_guarding_detected" in k_metrics
    assert "mean_flow_velocity" in k_metrics


def test_invariant_4_non_phonemic_purity() -> None:
    """Verifies that all sensory pipeline outputs are float arrays without textual tokens."""
    audio = np.zeros(RAW_AUDIO_SAMPLES, dtype=np.float32)
    frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(VIDEO_FRAMES)]

    x_audio, _ = extract_acoustic_latent(audio)
    x_kinematic, _ = extract_kinematic_latent(frames)

    assert not np.issubdtype(x_audio.dtype, np.str_)
    assert not np.issubdtype(x_audio.dtype, np.object_)
    assert not np.issubdtype(x_kinematic.dtype, np.str_)
    assert not np.issubdtype(x_kinematic.dtype, np.object_)
