"""
Project N: Realistic Sensory Audio Test Fixtures.
Provides non-stationary acoustic signals with calibrated background noise floors
and harmonic vocalization bursts for reliable SNR and feature extraction testing.
"""

import numpy as np


def generate_realistic_audio_clip(
    duration_sec: float = 5.0,
    sample_rate: int = 48000,
    f0_hz: float = 260.0,
    noise_rms: float = 0.002,
    burst_start_sec: float = 1.0,
    burst_end_sec: float = 3.5,
) -> np.ndarray:
    """
    Generates realistic non-stationary audio with calibrated room background noise
    and intermittent harmonic vocalization burst.
    Guarantees acoustic SNR (P90 vs P10) >= 15 dB (typically ~40 dB).
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Calibrated room background noise floor
    noise = np.random.normal(0, noise_rms, n_samples).astype(np.float32)

    # 2. Vocalization burst
    sig = np.zeros(n_samples, dtype=np.float32)
    burst_mask = (t >= burst_start_sec) & (t <= burst_end_sec)
    t_burst = t[burst_mask]

    if len(t_burst) > 0:
        burst_dur = burst_end_sec - burst_start_sec
        # Harmonic spectrum with fundamental and upper partials
        tone = (
            0.40 * np.sin(2 * np.pi * f0_hz * t_burst)
            + 0.15 * np.sin(2 * np.pi * 2.0 * f0_hz * t_burst)
            + 0.08 * np.sin(2 * np.pi * 3.0 * f0_hz * t_burst)
        )
        # Smooth Hann-like envelope for attack and release
        envelope = np.sin(np.pi * (t_burst - burst_start_sec) / burst_dur) ** 2
        sig[burst_mask] = (tone * envelope).astype(np.float32)

    audio = sig + noise
    return audio.astype(np.float32)


def generate_stationary_audio_clip(
    duration_sec: float = 5.0,
    sample_rate: int = 48000,
    f0_hz: float = 220.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generates stationary sine wave with no bursts (P90 ~ P10, low SNR)."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * f0_hz * t)).astype(np.float32)


def generate_clipped_audio_clip(
    duration_sec: float = 5.0,
    sample_rate: int = 48000,
    clip_ratio: float = 0.10,
) -> np.ndarray:
    """Generates audio where a specified fraction of samples exceed 0.99."""
    audio = generate_realistic_audio_clip(duration_sec, sample_rate)
    n_clip = int(len(audio) * clip_ratio)
    audio[:n_clip] = 1.0
    return audio
