"""
utils/preprocess.py
Video preprocessing utilities for DeepGuard deepfake detection.
"""

import cv2
import numpy as np


def extract_frames(video_path: str, num_frames: int = 16, size: tuple = (128, 128)):
    """
    Extract evenly-spaced frames from a video file.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract.
        size: Target (width, height) for each frame.

    Returns:
        frames: List of normalized numpy arrays, shape (H, W, 3).
        metadata: Dict with total_frames, fps, duration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)

    cap.release()

    metadata = {
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'duration': round(duration, 2),
        'frames_extracted': len(frames),
    }
    return frames, metadata


def compute_temporal_diff(frames: list) -> float:
    """Compute mean absolute difference between consecutive frames."""
    if len(frames) < 2:
        return 0.0
    diffs = [np.mean(np.abs(frames[i] - frames[i - 1])) for i in range(1, len(frames))]
    return float(np.mean(diffs))


def prepare_for_model(frames: list) -> np.ndarray:
    """
    Stack frames into a model-ready tensor.

    Returns:
        np.ndarray of shape (1, num_frames, H, W, 3)
    """
    stacked = np.array(frames)           # (T, H, W, 3)
    return np.expand_dims(stacked, 0)    # (1, T, H, W, 3)
