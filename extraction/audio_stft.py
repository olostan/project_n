"""
Project N: 48 kHz STFT and Broadband Log-Mel Spectrogram Extraction.
Computes high-fidelity spectral representations matching contracts.py dimensions.
"""

import numpy as np
import scipy.signal
from librosa.filters import mel

from models.contracts import (
    AUDIO_SAMPLE_RATE_HZ,
    PITCH_FRAMES,
    RAW_AUDIO_SAMPLES,
    STFT_HOP_H,
    STFT_UNCENTERED_FRAMES,
    STFT_WINDOW_N,
)


def compute_stft_uncentered(
    audio: np.ndarray,
    n_fft: int = STFT_WINDOW_N,
    hop_length: int = STFT_HOP_H,
) -> np.ndarray:
    """
    Computes uncentered STFT magnitude spectrum over 48 kHz audio.

    Args:
        audio: 1D numpy array of shape (RAW_AUDIO_SAMPLES,) = (240,000,).
        n_fft: FFT window size (default 2048).
        hop_length: Hop length (default 160).

    Returns:
        Magnitude spectrum of shape (1488, 1025).
    """
    if len(audio) < RAW_AUDIO_SAMPLES:
        padded = np.zeros(RAW_AUDIO_SAMPLES, dtype=audio.dtype)
        padded[: len(audio)] = audio
        audio = padded

    window = scipy.signal.windows.hann(n_fft, sym=False)
    # Perform framing manually for strictly uncentered STFT
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(
        audio[: (num_frames - 1) * hop_length + n_fft], n_fft
    )[::hop_length]

    windowed_frames = frames * window
    stft_complex = np.fft.rfft(windowed_frames, n=n_fft, axis=-1)
    magnitude = np.abs(stft_complex)
    return magnitude[:STFT_UNCENTERED_FRAMES]  # (1488, 1025)


def compute_log_mel_spectrogram(
    audio: np.ndarray,
    sr: int = AUDIO_SAMPLE_RATE_HZ,
    n_mels: int = 128,
    target_frames: int = PITCH_FRAMES,
) -> np.ndarray:
    """
    Computes 128-band log-mel spectrogram, aligned temporally to target_frames (500).

    Args:
        audio: 1D audio array of 240,000 samples.
        sr: Sample rate (default 48,000 Hz).
        n_mels: Number of mel filter bands (default 128).
        target_frames: Number of output temporal frames (default 500).

    Returns:
        Log-mel spectrogram of shape (target_frames, n_mels) = (500, 128).
    """
    mag = compute_stft_uncentered(audio)  # (1488, 1025)
    mel_basis = mel(sr=sr, n_fft=STFT_WINDOW_N, n_mels=n_mels, fmin=20.0, fmax=sr / 2.0)
    mel_power = np.dot(mag**2, mel_basis.T)  # (1488, 128)
    log_mel = np.log(np.maximum(mel_power, 1e-8))

    # Temporally downsample / interpolate from 1488 frames to 500 frames
    time_indices = np.linspace(0, log_mel.shape[0] - 1, target_frames)
    resampled = np.zeros((target_frames, n_mels), dtype=np.float32)
    for b in range(n_mels):
        resampled[:, b] = np.interp(time_indices, np.arange(log_mel.shape[0]), log_mel[:, b])

    return resampled
