"""
Project N: Farnebäck Optical Flow with 4x Pre-Downscaling.
Calculates 8x8 grid pooled dense optical flow (128 dims) with 320x180 pre-downscaling
to eliminate CPU bottlenecks and deliver deterministic bounds.
"""

import cv2
import numpy as np

from models.contracts import (
    FLOW_PRE_DOWNSCALE_HEIGHT,
    FLOW_PRE_DOWNSCALE_WIDTH,
    OPTICAL_FLOW_DIMS,
    OPTICAL_FLOW_GRID_X,
    OPTICAL_FLOW_GRID_Y,
    VIDEO_FRAMES,
)


def compute_frame_pair_flow(
    prev_frame_gray: np.ndarray,
    curr_frame_gray: np.ndarray,
    grid_x: int = OPTICAL_FLOW_GRID_X,
    grid_y: int = OPTICAL_FLOW_GRID_Y,
) -> np.ndarray:
    """
    Computes 8x8 grid pooled Farnebäck optical flow from a pre-downscaled frame pair.

    Args:
        prev_frame_gray: (180, 320) uint8 grayscale image.
        curr_frame_gray: (180, 320) uint8 grayscale image.
        grid_x: Horizontal grid bins (default 8).
        grid_y: Vertical grid bins (default 8).

    Returns:
        1D numpy array of length 128 (8 * 8 * 2).
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_frame_gray,
        curr_frame_gray,
        flow=None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )  # Shape: (180, 320, 2)

    h, w, _ = flow.shape
    cell_h = h // grid_y
    cell_w = w // grid_x

    pooled = np.zeros((grid_y, grid_x, 2), dtype=np.float32)
    for gy in range(grid_y):
        for gx in range(grid_x):
            cell = flow[gy * cell_h : (gy + 1) * cell_h, gx * cell_w : (gx + 1) * cell_w]
            pooled[gy, gx, 0] = np.mean(cell[:, :, 0])  # u
            pooled[gy, gx, 1] = np.mean(cell[:, :, 1])  # v

    return pooled.reshape(-1)  # (128,)


def compute_sequence_optical_flow(
    frames: list[np.ndarray],
    target_frames: int = VIDEO_FRAMES,
) -> np.ndarray:
    """
    Computes optical flow across a video sequence of 150 frames.
    Prepends zero-displacement initial frame to maintain temporal length of 150.

    Args:
        frames: List of 150 BGR or Grayscale images.
        target_frames: Number of output frames (default 150).

    Returns:
        Optical flow matrix of shape (150, 128).
    """
    flow_seq = np.zeros((target_frames, OPTICAL_FLOW_DIMS), dtype=np.float32)
    if len(frames) < 2:
        return flow_seq

    # Pre-downscale all frames to 320x180 grayscale
    downscaled_gray: list[np.ndarray] = []
    for f in frames[:target_frames]:
        if f.ndim == 3 and f.shape[2] == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        elif f.ndim == 2:
            gray = f
        else:
            gray = np.zeros((FLOW_PRE_DOWNSCALE_HEIGHT, FLOW_PRE_DOWNSCALE_WIDTH), dtype=np.uint8)

        if gray.shape != (FLOW_PRE_DOWNSCALE_HEIGHT, FLOW_PRE_DOWNSCALE_WIDTH):
            gray = cv2.resize(
                gray,
                (FLOW_PRE_DOWNSCALE_WIDTH, FLOW_PRE_DOWNSCALE_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
        downscaled_gray.append(gray)

    # Frame 0 has zero displacement
    for t in range(1, len(downscaled_gray)):
        flow_seq[t] = compute_frame_pair_flow(downscaled_gray[t - 1], downscaled_gray[t])

    return flow_seq
