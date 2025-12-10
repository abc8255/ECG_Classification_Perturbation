"""
Shared helpers for vulnerability experiment drivers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from perturbations.config import CLASS_NAMES
from visualization import (
    plot_asr_time_class_heatmap,
    plot_asr_vs_time,
    plot_robust_fraction_by_class,
    plot_strength_boxplot_by_class,
    plot_strength_histogram,
)


def generate_figures(
    df_windows,
    df_samples,
    output_dir: Path,
) -> None:
    """
    Create the standard vulnerability figures and save them under output_dir.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_specs: Tuple[Tuple[Any, Dict[str, Any], Path], ...] = (
        (
            plot_asr_vs_time,
            {"df_windows": df_windows, "bin_width": 0.5},
            output_dir / "asr_vs_time.png",
        ),
        (
            plot_asr_time_class_heatmap,
            {"df_windows": df_windows, "class_names": CLASS_NAMES, "bin_width": 0.5},
            output_dir / "asr_time_class_heatmap.png",
        ),
        (
            plot_strength_histogram,
            {"df_samples": df_samples},
            output_dir / "strength_histogram.png",
        ),
        (
            plot_strength_boxplot_by_class,
            {"df_samples": df_samples, "class_names": CLASS_NAMES},
            output_dir / "strength_boxplot_by_class.png",
        ),
        (
            plot_robust_fraction_by_class,
            {"df_samples": df_samples, "class_names": CLASS_NAMES},
            output_dir / "robust_fraction_by_class.png",
        ),
    )

    for fn, kwargs, path in figure_specs:
        try:
            fn(show=False, save_path=str(path), **kwargs)
            logging.info("Saved %s", path)
        except ValueError as exc:
            logging.warning("Skipping %s: %s", fn.__name__, exc)


def parse_strength_schedule(text: str) -> Sequence[float]:
    """
    Parse a comma-separated string of floats into a strength schedule.
    """

    values = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise ValueError("Strength schedule must contain at least one value.")
    return values


def load_eval_arrays(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NPZ evaluation arrays ensuring required fields exist.
    """

    data = np.load(path)
    if "X_test" not in data or "y_test_enc" not in data:
        raise ValueError("NPZ file must contain 'X_test' and 'y_test_enc'.")
    return data["X_test"], data["y_test_enc"]
