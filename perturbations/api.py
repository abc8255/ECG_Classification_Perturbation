"""
User-facing API surface for ECG perturbations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .adv_smooth import smooth_adversarial_perturbation
from .morphology import local_amplitude_scaling, local_time_warp
from .noise import band_limited_noise, baseline_wander


@dataclass
class PerturbationConfig:
    ptype: str
    strength: float
    center_time: float
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)


def apply_perturbation(
    x: np.ndarray,
    *,
    fs: float,
    config: PerturbationConfig,
    model: Optional[Any] = None,
    y_true: Optional[np.ndarray] = None,
    r_peaks: Optional[Sequence[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply the configured perturbation to `x`.
    """

    ptype = config.ptype.lower()
    strength = float(np.clip(config.strength, 0.0, 1.0))
    center_time = float(config.center_time)
    window_seconds = config.window_seconds

    if ptype == "baseline_wander":
        return baseline_wander(
            x,
            fs=fs,
            strength=strength,
            center_time=center_time,
            window_seconds=window_seconds,
            alpha=config.get_extra("alpha"),
            freq_range=config.get_extra("f_range"),
            rng=rng,
        )
    if ptype == "band_noise":
        return band_limited_noise(
            x,
            fs=fs,
            strength=strength,
            center_time=center_time,
            window_seconds=window_seconds,
            beta=config.get_extra("beta"),
            band=config.get_extra("band"),
            rng=rng,
        )
    if ptype == "morph_amp":
        if r_peaks is None:
            raise ValueError("morph_amp requires r_peaks for the sample.")
        return local_amplitude_scaling(
            x,
            fs=fs,
            config=config,
            r_peaks=r_peaks,
            rng=rng,
        )
    if ptype == "morph_time":
        if r_peaks is None:
            raise ValueError("morph_time requires r_peaks for the sample.")
        return local_time_warp(
            x,
            fs=fs,
            config=config,
            r_peaks=r_peaks,
            rng=rng,
        )
    if ptype == "smooth_adv":
        if model is None:
            raise ValueError("smooth_adv perturbations require a model instance.")
        if y_true is None:
            raise ValueError("smooth_adv perturbations require the true label vector.")
        x_adv, _ = smooth_adversarial_perturbation(
            x,
            fs=fs,
            config=config,
            model=model,
            y_true=y_true,
        )
        return x_adv

    raise ValueError(f"Unsupported perturbation type '{config.ptype}' for current implementation.")
