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
    is_acceptable: bool
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
            "is_acceptable": self.is_acceptable,
            "breaches": self.breaches,
        }


def compute_audio_snr_db(
    audio_pcm: np.ndarray,
    sample_rate: int = 48000,
    frame_ms: float = 20.0,
) -> float:
    """
    Estimates acoustic Signal-to-Noise Ratio (SNR) in dB using 20 ms frame power:
    Noise floor: 10th percentile of frame power across clip (P10).
    Signal power: 90th percentile of frame power across clip (P90).
    SNR = P90 - P10 (in dB).
    """
    if len(audio_pcm) == 0:
        return 0.0

    frame_size = int(sample_rate * (frame_ms / 1000.0))
    if frame_size <= 0:
        return 0.0

    num_frames = len(audio_pcm) // frame_size
    if num_frames == 0:
        return 0.0

    trimmed = audio_pcm[: num_frames * frame_size].reshape(num_frames, frame_size)
    frame_power = np.mean(trimmed**2, axis=1)

    # Check for complete digital silence
    if float(np.max(frame_power)) < 1e-9:
        return 0.0

    # Frame power in dB
    power_db = 10.0 * np.log10(frame_power + 1e-12)
    p10 = float(np.percentile(power_db, 10))
    p90 = float(np.percentile(power_db, 90))

    snr_db = max(0.0, p90 - p10)
    return float(snr_db)


def compute_audio_clipping_ratio(audio_pcm: np.ndarray, clip_threshold: float = 0.99) -> float:
    """Calculates fraction of audio samples exceeding digital clipping threshold."""
    if len(audio_pcm) == 0:
        return 0.0
    clipped_count = int(np.sum(np.abs(audio_pcm) >= clip_threshold))
    return float(clipped_count / len(audio_pcm))


def evaluate_signal_quality(
    audio_pcm: np.ndarray,
    mean_pose_confidence: float = 0.0,
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
    is_acceptable = len(breaches) == 0

    return SignalQualityReport(
        audio_snr_db=snr_db,
        audio_clipping_ratio=clipping_ratio,
        mean_pose_confidence=mean_pose_confidence,
        mean_flow_velocity_px_s=mean_flow_velocity,
        audio_valid=audio_valid,
        video_valid=video_valid,
        all_modalities_failed=all_failed,
        is_acceptable=is_acceptable,
        breaches=breaches,
    )
