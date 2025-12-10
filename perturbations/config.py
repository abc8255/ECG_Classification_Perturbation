"""
Configuration constants and defaults for the ECG perturbation package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


# Order follows the MultiLabelBinarizer fit in `StartingFile.ipynb`
CLASS_NAMES: Tuple[str, ...] = ("CD", "HYP", "MI", "NORM", "STTC")
CLASS_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class NoiseDefaults:
    """
    Container for default parameters used by the noise perturbations.
    """

    baseline_alpha: float = 0.1
    baseline_freq_range: Tuple[float, float] = (0.05, 0.5)
    band_beta: float = 0.1
    band_freq_range: Tuple[float, float] = (5.0, 40.0)
    hann_fade: bool = True


NOISE_DEFAULTS = NoiseDefaults()


def ensure_lead_axis(x: np.ndarray) -> np.ndarray:
    """
    Ensure ECG signals follow the (samples, leads) layout expected by the package.

    Notebook tensors are stored as (1000, 12); some callers may pass (12, 1000).
    """

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, received shape {x.shape}")
    samples, leads = x.shape
    if leads == 12:
        return x
    if samples == 12:
        return np.swapaxes(x, 0, 1)
    raise ValueError(
        "Unable to infer lead axis; expected one dimension to equal 12 leads."
    )

