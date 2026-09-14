"""
Project N: Sensory Signal Quality Assessment and Fail-Closed Abstention Gates.
Implements the 4 canonical signal-quality gates from docs/SPECS.md §4.2:
1. Acoustic SNR >= 12 dB
2. Audio clipping <= 5%
3. Mean pose landmark confidence >= 0.40
4. Camera ego-motion / optical flow velocity <= 85.0 px/s
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SignalQualityReport:
    """Detailed summary of signal-quality metrics and validation flags."""

    audio_snr_db: float
    audio_clipping_ratio: float
    mean_pose_confidence: float
    mean_flow_velocity_px_s: float
    audio_valid: bool
    video_valid: bool
    all_modalities_failed: bool
    breaches: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_snr_db": round(self.audio_snr_db, 2),
            "audio_clipping_ratio": round(self.audio_clipping_ratio, 4),
            "mean_pose_confidence": round(self.mean_pose_confidence, 2),
            "mean_flow_velocity_px_s": round(self.mean_flow_velocity_px_s, 2),
            "audio_valid": self.audio_valid,
            "video_valid": self.video_valid,
            "all_modalities_failed": self.all_modalities_failed,
            "breaches": self.breaches,
        }


def compute_audio_snr_db(audio_pcm: np.ndarray) -> float:
    """
    Estimates acoustic Signal-to-Noise Ratio (SNR) in dB using spectral flatness
    and root-mean-square (RMS) energy relative to silence/noise floor.
    """
    if len(audio_pcm) == 0:
        return 0.0

    rms = float(np.sqrt(np.mean(audio_pcm**2)))
    if rms < 0.005:  # Below acoustic detection threshold / silence
        return 0.0

    # Spectral flatness estimation (Wiener entropy)
    spec = np.abs(np.fft.rfft(audio_pcm)) ** 2
    log_spec = np.log(spec + 1e-12)
    geo_mean = float(np.exp(np.mean(log_spec)))
    arith_mean = float(np.mean(spec))
    flatness = geo_mean / (arith_mean + 1e-12)

    # Pure noise: flatness ~ 1.0 (SNR < 3 dB). Speech/harmonic signal: flatness < 0.05 (SNR > 13 dB).
    snr_db = max(0.0, -10.0 * np.log10(max(flatness, 1e-6)))
    return float(snr_db)


def compute_audio_clipping_ratio(audio_pcm: np.ndarray, clip_threshold: float = 0.99) -> float:
    """Calculates fraction of audio samples exceeding digital clipping threshold."""
    if len(audio_pcm) == 0:
        return 0.0
    clipped_count = int(np.sum(np.abs(audio_pcm) >= clip_threshold))
    return float(clipped_count / len(audio_pcm))


def evaluate_signal_quality(
    audio_pcm: np.ndarray,
    mean_pose_confidence: float = 1.0,
    mean_flow_velocity: float = 0.0,
    snr_threshold_db: float = 12.0,
    clipping_limit: float = 0.05,
    min_pose_confidence: float = 0.40,
    max_flow_velocity_px_s: float = 85.0,
) -> SignalQualityReport:
    """
    Evaluates fail-closed sensory signal quality gates across audio and video streams.
    """
    snr_db = compute_audio_snr_db(audio_pcm)
    clipping_ratio = compute_audio_clipping_ratio(audio_pcm)

    breaches: list[str] = []

    # 1. Audio SNR Gate
    audio_snr_valid = snr_db >= snr_threshold_db
    if not audio_snr_valid:
        breaches.append(
            f"Acoustic SNR ({snr_db:.1f} dB) below {snr_threshold_db:.1f} dB threshold."
        )

    # 2. Audio Clipping Gate
    audio_clip_valid = clipping_ratio <= clipping_limit
    if not audio_clip_valid:
        breaches.append(
            f"Audio clipping ({clipping_ratio * 100:.1f}%) exceeds {clipping_limit * 100:.1f}% limit."
        )

    audio_valid = audio_snr_valid and audio_clip_valid

    # 3. Pose Confidence Gate
    pose_valid = mean_pose_confidence >= min_pose_confidence
    if not pose_valid:
        breaches.append(
            f"Mean pose tracking confidence ({mean_pose_confidence:.2f}) below {min_pose_confidence:.2f} threshold."
        )

    # 4. Camera Ego-Motion Gate
    flow_valid = mean_flow_velocity <= max_flow_velocity_px_s
    if not flow_valid:
        breaches.append(
            f"Camera optical flow velocity ({mean_flow_velocity:.1f} px/s) exceeds {max_flow_velocity_px_s:.1f} px/s stability limit."
        )

    video_valid = pose_valid and flow_valid
    all_failed = (not audio_valid) and (not video_valid)

    return SignalQualityReport(
        audio_snr_db=snr_db,
        audio_clipping_ratio=clipping_ratio,
        mean_pose_confidence=mean_pose_confidence,
        mean_flow_velocity_px_s=mean_flow_velocity,
        audio_valid=audio_valid,
        video_valid=video_valid,
        all_modalities_failed=all_failed,
        breaches=breaches,
    )
