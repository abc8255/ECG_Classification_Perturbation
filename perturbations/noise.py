"""
Noise perturbations that mimic acquisition artefacts.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import NOISE_DEFAULTS, ensure_lead_axis
from .masks import broadcast_mask, build_time_mask


def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _per_lead_std(x: np.ndarray) -> np.ndarray:
    """
    Compute per-lead standard deviations with safe lower bounds for scaling.
    """

    std = np.std(x, axis=0, keepdims=True)
    return np.maximum(std, 1e-6)


def baseline_wander(
    x: np.ndarray,
    *,
    fs: float,
    strength: float,
    center_time: float,
    window_seconds: float | None,
    alpha: float | None = None,
    freq_range: tuple[float, float] | None = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add low-frequency sinusoidal drift to mimic respiration or electrode motion.
    """

    x_in = ensure_lead_axis(np.asarray(x, dtype=np.float32))
    if strength <= 0:
        return x_in.copy()

    num_samples, num_leads = x_in.shape
    alpha = alpha if alpha is not None else NOISE_DEFAULTS.baseline_alpha
    freq_range = freq_range if freq_range is not None else NOISE_DEFAULTS.baseline_freq_range
    rng = _get_rng(rng)

    mask = broadcast_mask(
        build_time_mask(num_samples, fs, center_time, window_seconds, use_hann=NOISE_DEFAULTS.hann_fade),
        num_leads,
    )

    t = np.arange(num_samples, dtype=np.float32)[:, None] / float(fs)
    std = _per_lead_std(x_in)
    amplitudes = strength * alpha * std  # shape (1, leads)
    freqs = rng.uniform(freq_range[0], freq_range[1], size=(1, num_leads))
    phases = rng.uniform(0, 2 * np.pi, size=(1, num_leads))

    baseline = amplitudes * np.sin(2 * np.pi * freqs * t + phases)
    perturbed = x_in + mask * baseline
    return perturbed


def _bandlimit_noise(
    noise: np.ndarray,
    fs: float,
    band: tuple[float, float],
) -> np.ndarray:
    """
    Filter white noise in the frequency domain to keep components within `band`.
    """

    num_samples = noise.shape[0]
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / fs)
    mask = (freqs >= band[0]) & (freqs <= band[1])

    spectrum = np.fft.rfft(noise, axis=0)
    spectrum[~mask, :] = 0.0
    filtered = np.fft.irfft(spectrum, n=num_samples, axis=0)
    return filtered.astype(np.float32, copy=False)


def band_limited_noise(
    x: np.ndarray,
    *,
    fs: float,
    strength: float,
    center_time: float,
    window_seconds: float | None,
    beta: float | None = None,
    band: tuple[float, float] | None = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add band-limited Gaussian noise to mimic instrumentation or muscle noise.
    """

    x_in = ensure_lead_axis(np.asarray(x, dtype=np.float32))
    if strength <= 0:
        return x_in.copy()

    num_samples, num_leads = x_in.shape
    beta = beta if beta is not None else NOISE_DEFAULTS.band_beta
    band = band if band is not None else NOISE_DEFAULTS.band_freq_range
    rng = _get_rng(rng)

    white = rng.normal(size=(num_samples, num_leads)).astype(np.float32)
    filtered = _bandlimit_noise(white, fs, band)

    # Normalize variance to 1 per lead to make `beta` comparable across leads.
    filtered_std = np.std(filtered, axis=0, keepdims=True)
    filtered_std = np.maximum(filtered_std, 1e-6)
    filtered /= filtered_std

    std = _per_lead_std(x_in)
    scale = strength * beta * std
    mask = broadcast_mask(
        build_time_mask(num_samples, fs, center_time, window_seconds, use_hann=NOISE_DEFAULTS.hann_fade),
        num_leads,
    )

    perturbation = mask * scale * filtered
    return x_in + perturbation
