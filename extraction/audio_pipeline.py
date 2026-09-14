"""
Project N: Unified Audio Extraction Pipeline.
Combines STFT log-mel, 16 pitch descriptors, and CQT filterbank into an
aligned acoustic latent matrix X_audio in R^(500, 512) and summary metrics.
"""

from typing import Any

import numpy as np

from extraction.audio_cqt import compute_cqt_filterbank
from extraction.audio_pitch import extract_pitch_features
from extraction.audio_stft import compute_log_mel_spectrogram
from models.contracts import (
    ACOUSTIC_LATENT_D,
    AUDIO_SAMPLE_RATE_HZ,
    PITCH_FRAMES,
    RAW_AUDIO_SAMPLES,
)


def extract_acoustic_latent(
    audio: np.ndarray,
    sr: int = AUDIO_SAMPLE_RATE_HZ,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Extracts unified acoustic latent representation X_audio and summary metrics.

    Args:
        audio: 1D raw PCM audio array of 240,000 samples (5.0s @ 48 kHz).
        sr: Sampling frequency (48,000 Hz).

    Returns:
        tuple (x_audio, metrics):
            - x_audio: (500, 512) float32 numpy array.
            - metrics: dictionary of measured scalars for Layer 1 and triage screener:
                       f0_mean_hz, cpp_db, jitter_mean, shimmer_mean, hnr_db.
    """
    if len(audio) < RAW_AUDIO_SAMPLES:
        padded = np.zeros(RAW_AUDIO_SAMPLES, dtype=audio.dtype)
        padded[: len(audio)] = audio
        audio = padded

    # 1. Extract component features (all aligned to 500 frames)
    pitch_mat = extract_pitch_features(audio, sr=sr, target_frames=PITCH_FRAMES)  # (500, 16)
    cqt_mat = compute_cqt_filterbank(audio, sr=sr, target_frames=PITCH_FRAMES)  # (500, 84)
    mel_mat = compute_log_mel_spectrogram(audio, sr=sr, target_frames=PITCH_FRAMES)  # (500, 128)

    # 2. Concatenate along feature axis: 16 + 84 + 128 = 228
    concat_raw = np.concatenate([pitch_mat, cqt_mat, mel_mat], axis=-1)  # (500, 228)

    # 3. Project / pad to canonical ACOUSTIC_LATENT_D = 512
    x_audio = np.zeros((PITCH_FRAMES, ACOUSTIC_LATENT_D), dtype=np.float32)
    x_audio[:, :228] = concat_raw

    # 4. Compute measured bioacoustic scalar summaries
    voiced_f0 = pitch_mat[:, 0][pitch_mat[:, 0] > 0]
    f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    cpp_mean = float(np.mean(pitch_mat[:, 8]))
    jitter_mean = float(np.mean(pitch_mat[:, 3]))
    shimmer_mean = float(np.mean(pitch_mat[:, 5]))
    hnr_mean = float(np.mean(pitch_mat[:, 7]))

    metrics = {
        "f0_mean_hz": round(f0_mean, 2),
        "cpp_db": round(cpp_mean, 2),
        "jitter_mean": round(jitter_mean, 4),
        "shimmer_mean": round(shimmer_mean, 4),
        "hnr_db": round(hnr_mean, 2),
    }

    return x_audio, metrics
