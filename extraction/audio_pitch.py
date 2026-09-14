"""
Project N: 16-Dimensional Pitch and Periodicity Extraction.
Tracks fundamental frequency (F0), voice perturbation (jitter, shimmer), HNR,
and Cepstral Peak Prominence (CPP) over a 10 ms hop (500 frames per 5.0s).
"""

import numpy as np
import scipy.signal

from models.contracts import (
    AUDIO_SAMPLE_RATE_HZ,
    PITCH_DIMENSIONS,
    PITCH_FRAMES,
    PITCH_HOP_MS,
    RAW_AUDIO_SAMPLES,
)


def extract_pitch_features(
    audio: np.ndarray,
    sr: int = AUDIO_SAMPLE_RATE_HZ,
    target_frames: int = PITCH_FRAMES,
) -> np.ndarray:
    """
    Extracts the 16 canonical pitch and voice quality descriptors across 500 frames.

    Args:
        audio: 1D numpy array of 240,000 samples.
        sr: Sample rate in Hz (default 48,000).
        target_frames: Desired temporal frames (default 500 = 10 ms hop).

    Returns:
        Feature matrix of shape (500, 16).
    """
    if len(audio) < RAW_AUDIO_SAMPLES:
        padded = np.zeros(RAW_AUDIO_SAMPLES, dtype=audio.dtype)
        padded[: len(audio)] = audio
        audio = padded

    hop_samples = int(sr * (PITCH_HOP_MS / 1000.0))  # 480 samples = 10 ms
    frame_len = 2048  # Analysis window (~42 ms @ 48 kHz for reliable 50 Hz tracking)

    f0_arr = np.zeros(target_frames, dtype=np.float32)
    voicing_arr = np.zeros(target_frames, dtype=np.float32)
    rms_arr = np.zeros(target_frames, dtype=np.float32)
    cpp_arr = np.zeros(target_frames, dtype=np.float32)
    hnr_arr = np.zeros(target_frames, dtype=np.float32)
    centroid_arr = np.zeros(target_frames, dtype=np.float32)
    spread_arr = np.zeros(target_frames, dtype=np.float32)
    entropy_arr = np.zeros(target_frames, dtype=np.float32)
    tilt_arr = np.zeros(target_frames, dtype=np.float32)

    min_lag = int(sr / 600.0)  # 600 Hz upper pitch limit (80 samples)
    max_lag = int(sr / 50.0)  # 50 Hz lower pitch limit (960 samples)

    for i in range(target_frames):
        start = i * hop_samples
        end = min(start + frame_len, len(audio))
        frame = audio[start:end]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(frame**2) + 1e-12))
        rms_db = float(20.0 * np.log10(max(rms, 1e-6)))
        rms_arr[i] = rms_db

        # 2. Autocorrelation for F0 and Voicing
        norm_factor = np.sum(frame**2) + 1e-12
        autocorr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        autocorr = autocorr / norm_factor

        if len(autocorr) >= max_lag:
            search_region = autocorr[min_lag:max_lag]
            peak_idx = int(np.argmax(search_region)) + min_lag
            peak_val = float(search_region[peak_idx - min_lag])

            if peak_val > 0.35 and rms > 0.005:
                # Parabolic peak interpolation on autocorrelation lags (eliminates quantization floor)
                if 0 < peak_idx < len(autocorr) - 1:
                    alpha = float(autocorr[peak_idx - 1])
                    beta = float(autocorr[peak_idx])
                    gamma = float(autocorr[peak_idx + 1])
                    denom = alpha - 2.0 * beta + gamma
                    delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-8 else 0.0
                    refined_lag = max(float(min_lag), min(float(max_lag), float(peak_idx) + delta))
                else:
                    refined_lag = float(peak_idx)
                f0_arr[i] = float(sr / refined_lag)
                voicing_arr[i] = min(peak_val, 1.0)
                hnr_arr[i] = float(10.0 * np.log10(max(peak_val / (1.0 - peak_val + 1e-6), 0.1)))
            else:
                f0_arr[i] = 0.0
                voicing_arr[i] = 0.0
                hnr_arr[i] = 0.0

        # 3. Spectral Features (Centroid, Spread, Entropy, Tilt)
        spec = np.abs(np.fft.rfft(frame * np.hanning(len(frame)))) + 1e-12
        freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
        spec_norm = spec / np.sum(spec)

        cent = float(np.sum(freqs * spec_norm))
        centroid_arr[i] = cent
        spread_arr[i] = float(np.sqrt(np.sum(((freqs - cent) ** 2) * spec_norm)))

        # Spectral Entropy
        entropy_arr[i] = float(
            -np.sum(spec_norm * np.log(spec_norm + 1e-12)) / np.log(len(spec_norm))
        )

        # Spectral Tilt (Slope over 0-4000 Hz)
        mask_4k = freqs <= 4000.0
        if np.sum(mask_4k) > 5:
            poly = np.polyfit(freqs[mask_4k], 20.0 * np.log10(spec[mask_4k]), deg=1)
            tilt_arr[i] = float(poly[0])

        # 4. Cepstral Peak Prominence (CPP) via true linear regression baseline
        log_spec = np.log(spec**2 + 1e-12)
        cepstrum = np.real(np.fft.irfft(log_spec))
        q_min, q_max = min_lag, min(max_lag, len(cepstrum) // 2)
        if q_max > q_min + 5:
            q_region = cepstrum[q_min:q_max]
            quefrencies = np.arange(q_min, q_max, dtype=np.float64)
            poly_c = np.polyfit(quefrencies, q_region, deg=1)
            baseline = np.polyval(poly_c, quefrencies)
            prominence = q_region - baseline
            cpp_arr[i] = max(0.0, float(20.0 * np.max(prominence)))

    # Derivatives
    delta_f0 = np.gradient(f0_arr)
    delta2_f0 = np.gradient(delta_f0)
    delta_rms = np.gradient(rms_arr)

    # Voice perturbation metrics strictly over voiced segments (>3 consecutive cycles)
    jitter_local = np.zeros(target_frames, dtype=np.float32)
    shimmer_local = np.zeros(target_frames, dtype=np.float32)

    voiced_indices = np.where(voicing_arr > 0.35)[0]
    if len(voiced_indices) > 3:
        segments = np.split(voiced_indices, np.where(np.diff(voiced_indices) > 1)[0] + 1)
        for seg in segments:
            if len(seg) >= 3:
                periods = sr / np.maximum(f0_arr[seg], 50.0)
                mean_p = float(np.mean(periods))
                if mean_p > 0:
                    diff_p = np.abs(np.diff(periods))
                    j_val = float(np.mean(diff_p) / mean_p)
                    jitter_local[seg] = j_val

                amps = 10.0 ** (rms_arr[seg] / 20.0)
                mean_a = float(np.mean(amps))
                if mean_a > 1e-6:
                    diff_a = np.abs(np.diff(amps))
                    s_val = float(np.mean(diff_a) / mean_a)
                    shimmer_local[seg] = s_val

    jitter_rap = scipy.signal.medfilt(jitter_local, kernel_size=3)
    shimmer_apq3 = scipy.signal.medfilt(shimmer_local, kernel_size=3)

    # Pack into (500, 16) matrix matching SPECS.md §2.1
    features = np.zeros((target_frames, PITCH_DIMENSIONS), dtype=np.float32)
    features[:, 0] = f0_arr
    features[:, 1] = delta_f0
    features[:, 2] = delta2_f0
    features[:, 3] = jitter_local
    features[:, 4] = jitter_rap
    features[:, 5] = shimmer_local
    features[:, 6] = shimmer_apq3
    features[:, 7] = hnr_arr
    features[:, 8] = cpp_arr
    features[:, 9] = tilt_arr
    features[:, 10] = entropy_arr
    features[:, 11] = centroid_arr
    features[:, 12] = spread_arr
    features[:, 13] = voicing_arr
    features[:, 14] = rms_arr
    features[:, 15] = delta_rms

    return features
