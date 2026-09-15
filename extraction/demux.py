"""
Project N: Media Demuxing & Decoding Substrate.
Decodes raw encrypted video containers (MP4) to 48 kHz float32 PCM audio
and 30 fps RGB video frames without synthetic fallbacks.
"""

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from models.contracts import AUDIO_SAMPLE_RATE_HZ, VIDEO_FPS


class MediaDecodeError(Exception):
    """Raised when media container cannot be demuxed or decoded."""


def demux_clip_bytes(
    clip_bytes: bytes,
    target_audio_sr: int = AUDIO_SAMPLE_RATE_HZ,
    target_video_fps: int = VIDEO_FPS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Demuxes binary container bytes (e.g. MP4) to raw audio PCM and RGB frames.

    Args:
        clip_bytes: Raw decrypted binary bytes of the media container.
        target_audio_sr: Target audio sampling rate in Hz (48,000).
        target_video_fps: Target video frame rate in frames per second (30).

    Returns:
        tuple (audio_pcm, video_frames):
            - audio_pcm: 1D float32 numpy array of audio samples at 48 kHz.
            - video_frames: list of uint8 RGB numpy arrays (H, W, 3).

    Raises:
        MediaDecodeError: If media format is invalid or cannot be decoded.
    """
    if not clip_bytes:
        raise MediaDecodeError("Empty media container payload.")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input_clip.mp4"
        audio_wav_path = Path(tmpdir) / "audio.wav"

        input_path.write_bytes(clip_bytes)

        # 1. Demux & resample audio track to 48 kHz mono 16-bit WAV via ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            str(target_audio_sr),
            str(audio_wav_path),
        ]

        try:
            res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=15)
            if res.returncode != 0 or not audio_wav_path.exists():
                raise MediaDecodeError(f"Audio extraction failed: {res.stderr}")

            audio_data, sr = sf.read(str(audio_wav_path), dtype="float32")
            if sr != target_audio_sr:
                raise MediaDecodeError(f"Unexpected audio sampling rate: {sr} != {target_audio_sr}")
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0]
        except (subprocess.SubprocessError, sf.SoundFileError, OSError) as e:
            raise MediaDecodeError(f"Failed to decode audio track: {e}") from e

        # 2. Demux video frames at target_video_fps via OpenCV VideoCapture
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise MediaDecodeError("Could not open video stream in container.")

        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, round(cap_fps / target_video_fps)) if cap_fps and cap_fps > 0 else 1

        frames: list[np.ndarray] = []
        frame_idx = 0
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                frame_idx += 1
        finally:
            cap.release()

        if not frames:
            raise MediaDecodeError("Video stream contained zero decodable frames.")

        return audio_data, frames


def create_synthetic_mp4(duration_sec: float = 5.0, f0_hz: float = 220.0) -> bytes:
    """
    Generates a valid MP4 test container with video and non-stationary audio streams.
    Used exclusively by test suites to verify real demuxing without mocking.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "test_out.mp4"
        audio_file = Path(tmpdir) / "synth_audio.wav"

        # Generate realistic non-stationary burst audio with room noise
        sr = AUDIO_SAMPLE_RATE_HZ
        n_samples = int(duration_sec * sr)
        t = np.linspace(0, duration_sec, n_samples, endpoint=False)
        noise = np.random.normal(0, 0.002, n_samples).astype(np.float32)
        sig = np.zeros(n_samples, dtype=np.float32)

        b_start = 0.2 * duration_sec
        b_end = 0.8 * duration_sec
        mask = (t >= b_start) & (t <= b_end)
        t_burst = t[mask]
        if len(t_burst) > 0:
            burst_dur = b_end - b_start
            tone = 0.40 * np.sin(2 * np.pi * f0_hz * t_burst) + 0.15 * np.sin(
                2 * np.pi * 2.0 * f0_hz * t_burst
            )
            env = np.sin(np.pi * (t_burst - b_start) / burst_dur) ** 2
            sig[mask] = (tone * env).astype(np.float32)

        audio_pcm = (sig + noise).astype(np.float32)
        sf.write(str(audio_file), audio_pcm, sr)

        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_sec}:size=320x180:rate=30",
            "-i",
            str(audio_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(out_file),
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if res.returncode != 0 or not out_file.exists():
                raise MediaDecodeError(f"Failed to create test MP4: {res.stderr}")
            return out_file.read_bytes()
        except OSError as err:
            raise MediaDecodeError(
                f"ffmpeg executable not found or execution failed: {err}"
            ) from err
