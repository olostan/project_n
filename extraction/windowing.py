"""
Project N: Window Slicing and Temporal Framing.
Partitions 30-120s continuous clips into contiguous 5.0s non-overlapping windows.
"""

from typing import Any

import numpy as np

from models.contracts import (
    AUDIO_SAMPLE_RATE_HZ,
    MAX_CLIP_DURATION_SEC,
    MAX_CLIP_WINDOWS,
    MIN_CLIP_DURATION_SEC,
    MIN_CLIP_WINDOWS,
    VIDEO_FPS,
    WINDOW_DURATION_SEC,
)


def slice_audio_windows(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE_HZ,
    window_sec: float = WINDOW_DURATION_SEC,
) -> list[np.ndarray]:
    """
    Slices 1D raw audio into contiguous non-overlapping windows of window_sec duration.

    Args:
        audio: 1D numpy array of audio samples.
        sample_rate: Audio sampling frequency in Hz (default 48,000).
        window_sec: Window length in seconds (default 5.0s = 240,000 samples).

    Returns:
        List of 1D numpy arrays, each of shape (RAW_AUDIO_SAMPLES,).
    """
    samples_per_window = int(sample_rate * window_sec)
    total_samples = len(audio)
    windows_count = total_samples // samples_per_window

    if windows_count == 0:
        # If audio is shorter than one window, pad with trailing zeros
        padded = np.zeros(samples_per_window, dtype=audio.dtype)
        padded[:total_samples] = audio
        return [padded]

    windows: list[np.ndarray] = []
    for w in range(windows_count):
        start = w * samples_per_window
        end = start + samples_per_window
        windows.append(audio[start:end])

    return windows


def slice_video_windows(
    frames: list[Any],
    fps: int = VIDEO_FPS,
    window_sec: float = WINDOW_DURATION_SEC,
) -> list[list[Any]]:
    """
    Slices video frames into contiguous non-overlapping windows.

    Args:
        frames: Sequence of video frames (images or feature vectors).
        fps: Video frame rate (default 30 fps).
        window_sec: Window length in seconds (default 5.0s = 150 frames).

    Returns:
        List of frame windows, each containing int(fps * window_sec) frames.
    """
    frames_per_window = int(fps * window_sec)
    total_frames = len(frames)
    windows_count = total_frames // frames_per_window

    if windows_count == 0:
        if not frames:
            return []
        # Pad with the last frame if shorter than one window
        padded = list(frames)
        while len(padded) < frames_per_window:
            padded.append(frames[-1])
        return [padded]

    windows: list[list[Any]] = []
    for w in range(windows_count):
        start = w * frames_per_window
        end = start + frames_per_window
        windows.append(frames[start:end])

    return windows


def validate_clip_duration(duration_sec: float) -> int:
    """
    Validates clip duration conforms to [30.0, 120.0] seconds.

    Returns:
        Number of 5.0-second windows (W in [6, 24]).
    """
    if duration_sec < MIN_CLIP_DURATION_SEC or duration_sec > MAX_CLIP_DURATION_SEC:
        raise ValueError(
            f"Clip duration {duration_sec:.1f}s outside bounds "
            f"[{MIN_CLIP_DURATION_SEC}, {MAX_CLIP_DURATION_SEC}] seconds."
        )
    w_count = int(duration_sec // WINDOW_DURATION_SEC)
    return min(max(w_count, MIN_CLIP_WINDOWS), MAX_CLIP_WINDOWS)
