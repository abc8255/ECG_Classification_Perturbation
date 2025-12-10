"""
Quick sanity check for the noise perturbations.

Run with:
    python examples/noise_demo.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
VENV_SITE = ROOT_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

sys.path.append(str(ROOT_DIR))

import numpy as np

from perturbations import PerturbationConfig, apply_perturbation

BASE_SEED = 1337

FS = 100.0
SECONDS = 10.0
SAMPLES = int(FS * SECONDS)
LEADS = 12


def synthetic_ecg() -> np.ndarray:
    """Create a crude synthetic ECG-like signal for demo purposes."""

    t = np.linspace(0, SECONDS, SAMPLES, endpoint=False)
    base = 0.8 * np.sin(2 * np.pi * 1.2 * t)  # heart rate-ish
    signals = np.stack([base + 0.05 * np.sin(2 * np.pi * (i + 1) * t) for i in range(LEADS)], axis=1)
    return signals.astype(np.float32)


def main() -> None:
    x = synthetic_ecg()

    configs = [
        PerturbationConfig(ptype="baseline_wander", strength=0.5, center_time=5.0, window_seconds=4.0),
        PerturbationConfig(ptype="band_noise", strength=0.3, center_time=2.0, window_seconds=None),
    ]

    for idx, cfg in enumerate(configs):
        rng = np.random.default_rng(BASE_SEED + idx)
        x_adv = apply_perturbation(x, fs=FS, config=cfg, rng=rng)
        delta = x_adv - x
        print(f"{cfg.ptype} -> RMS delta: {np.sqrt(np.mean(delta**2)):.4f}, "
              f"L_inf delta: {np.max(np.abs(delta)):.4f}")


if __name__ == "__main__":
    main()
