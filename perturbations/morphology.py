"""
Morphology-aware perturbations that operate around detected beats.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import ensure_lead_axis

try:
    from wfdb import processing as wfdb_processing
except ImportError:  # pragma: no cover - optional dependency already in requirements
    wfdb_processing = None


def detect_r_peaks(x: np.ndarray, fs: float, lead_idx: int = 1) -> np.ndarray:
    """
    Detect R-peaks using WFDB's xqrs detector on a single lead.
    """

    if wfdb_processing is None:
        raise ImportError("wfdb is required for R-peak detection.")
    signal = ensure_lead_axis(np.asarray(x, dtype=np.float32))
    lead_idx = int(np.clip(lead_idx, 0, signal.shape[1] - 1))
    sig = signal[:, lead_idx]
    sig = sig - np.mean(sig)
    peaks = wfdb_processing.xqrs_detect(sig=sig, fs=fs)
    return np.asarray(peaks, dtype=int)


def _select_beats(
    r_peaks: Sequence[int],
    fs: float,
    center_time: float,
    window_seconds: Optional[float],
) -> np.ndarray:
    if r_peaks is None:
        raise ValueError("Morphology perturbations require r_peaks.")
    r_peaks = np.asarray(r_peaks, dtype=int)
    if window_seconds is None:
        return r_peaks
    center_idx = int(round(center_time * fs))
    half = int(round(window_seconds * fs / 2))
    start = max(0, center_idx - half)
    end = center_idx + half
    mask = (r_peaks >= start) & (r_peaks <= end)
    return r_peaks[mask]


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def local_amplitude_scaling(
    x: np.ndarray,
    *,
    fs: float,
    config,
    r_peaks: Sequence[int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Slightly scale amplitudes around beats within the selected window.
    """

    signal = ensure_lead_axis(np.asarray(x, dtype=np.float32)).copy()
    extra = config.extra or {}
    beats = _select_beats(r_peaks, fs, config.center_time, config.window_seconds)
    if beats.size == 0:
        return signal

    rng = _get_rng(rng)
    gamma_max = extra.get("gamma_max", 0.15)
    sigma_ms = extra.get("sigma_ms", 80.0)
    sigma_samples = max(1, int(round(sigma_ms / 1000 * fs)))
    num_samples = signal.shape[0]
    indices = np.arange(num_samples)[:, None]

    for r_idx in beats:
        gamma = (rng.uniform(-1.0, 1.0) * gamma_max * config.strength)
        window = np.exp(-0.5 * ((indices - r_idx) / sigma_samples) ** 2)
        signal += gamma * window * signal

    return signal


def local_time_warp(
    x: np.ndarray,
    *,
    fs: float,
    config,
    r_peaks: Sequence[int],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Stretch or compress local windows around beats.
    """

    signal = ensure_lead_axis(np.asarray(x, dtype=np.float32)).copy()
    extra = config.extra or {}
    beats = _select_beats(r_peaks, fs, config.center_time, config.window_seconds)
    if beats.size == 0:
        return signal

    rng = _get_rng(rng)
    epsilon_max = extra.get("epsilon_max", 0.1) * config.strength
    pre_ms = extra.get("pre_ms", 60.0)
    post_ms = extra.get("post_ms", 80.0)
    pre = max(1, int(round(pre_ms / 1000 * fs)))
    post = max(1, int(round(post_ms / 1000 * fs)))

    num_samples, num_leads = signal.shape
    time_axis = np.arange(num_samples)

    for r_idx in beats:
        start = max(0, r_idx - pre)
        end = min(num_samples, r_idx + post)
        if end - start < 3:
            continue

        segment = signal[start:end, :]
        tau = np.linspace(0.0, 1.0, end - start)
        epsilon = rng.uniform(-epsilon_max, epsilon_max)
        tau_warped = np.clip(tau * (1.0 + epsilon), 0.0, 1.0)
        target_positions = tau_warped * (segment.shape[0] - 1)
        new_segment = np.empty_like(segment)
        base_positions = np.linspace(0, segment.shape[0] - 1, segment.shape[0])
        for lead in range(num_leads):
            new_segment[:, lead] = np.interp(
                base_positions, target_positions, segment[:, lead]
            )

        blend = np.linspace(0.0, 1.0, segment.shape[0])[:, None]
        signal[start:end, :] = (
            signal[start:end, :] * (1.0 - blend) + new_segment * blend
        )

    return signal
