"""
Utilities for constructing time-domain masks used by perturbations.
"""

from __future__ import annotations

import numpy as np


def hann_window(length: int) -> np.ndarray:
    """
    Return a Hann (raised cosine) window of the requested length.
    """

    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.hanning(length).astype(np.float32)


def build_time_mask(
    num_samples: int,
    fs: float,
    center_time: float,
    window_seconds: float | None,
    *,
    use_hann: bool = True,
) -> np.ndarray:
    """
    Build a 1D mask array of length `num_samples` that emphasizes a window centered
    at `center_time` seconds. When `window_seconds` is None, the mask is all ones.
    """

    if window_seconds is None:
        return np.ones(num_samples, dtype=np.float32)

    center_idx = int(round(center_time * fs))
    half_window = int(round(window_seconds * fs / 2))
    start = max(0, center_idx - half_window)
    end = min(num_samples, center_idx + half_window)
    length = max(0, end - start)

    mask = np.zeros(num_samples, dtype=np.float32)
    if length == 0:
        return mask

    if use_hann and length > 1:
        window = hann_window(length)
    else:
        window = np.ones(length, dtype=np.float32)

    mask[start:end] = window
    return mask


def broadcast_mask(mask: np.ndarray, num_leads: int) -> np.ndarray:
    """
    Broadcast a 1D mask of length num_samples across the lead axis.
    """

    if mask.ndim != 1:
        raise ValueError("Mask must be 1D.")
    return np.repeat(mask[:, None], repeats=num_leads, axis=1)

