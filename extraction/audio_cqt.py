"""
Project N: Constant-Q Transform (CQT) 84-Bin Harmonic Transform.
Computes 7-octave geometrically spaced harmonic filterbank (32.7 Hz - 4186.0 Hz)
aligned temporally to 500 frames.
"""

import librosa
import numpy as np

from models.contracts import (
    AUDIO_SAMPLE_RATE_HZ,
    CQT_BINS_PER_OCTAVE,
    CQT_F_MIN_HZ,
    CQT_TOTAL_BINS,
    PITCH_FRAMES,
    RAW_AUDIO_SAMPLES,
)


def compute_cqt_filterbank(
    audio: np.ndarray,
    sr: int = AUDIO_SAMPLE_RATE_HZ,
    n_bins: int = CQT_TOTAL_BINS,
    bins_per_octave: int = CQT_BINS_PER_OCTAVE,
    fmin: float = CQT_F_MIN_HZ,
    target_frames: int = PITCH_FRAMES,
) -> np.ndarray:
    """
    Computes 84-bin Constant-Q Transform magnitude spectrum across 500 frames.

    Args:
        audio: 1D numpy array of 240,000 samples.
        sr: Sample rate in Hz (default 48,000).
        n_bins: Total frequency bins (default 84 = 7 octaves * 12 bins).
        bins_per_octave: Bins per octave (default 12).
        fmin: Minimum center frequency in Hz (default ~32.70 Hz, C1).
        target_frames: Number of output temporal frames (default 500).

    Returns:
        Magnitude CQT matrix of shape (500, 84).
    """
    if len(audio) < RAW_AUDIO_SAMPLES:
        padded = np.zeros(RAW_AUDIO_SAMPLES, dtype=audio.dtype)
        padded[: len(audio)] = audio
        audio = padded

    # Compute CQT using librosa with hop size ~ 480 (10 ms)
    hop_length = int(sr * 0.010)  # 480 samples
    cqt_complex = librosa.cqt(
        y=audio,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=0.0,
    )
    cqt_mag = np.abs(cqt_complex).T  # Shape: (T_raw, 84)

    # Normalize/interpolate to target_frames (500)
    current_frames = cqt_mag.shape[0]
    if current_frames == target_frames:
        return np.asarray(cqt_mag, dtype=np.float32)

    time_indices = np.linspace(0, current_frames - 1, target_frames)
    resampled = np.zeros((target_frames, n_bins), dtype=np.float32)
    for b in range(n_bins):
        resampled[:, b] = np.interp(time_indices, np.arange(current_frames), cqt_mag[:, b])

    return resampled
