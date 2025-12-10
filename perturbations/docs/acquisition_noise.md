# Acquisition-Style Noise Perturbations

This document explains the design, implementation, and practical usage of the *baseline wander* and *band-limited noise* perturbations contained in `perturbations/noise.py`. These perturbations mimic common ECG acquisition artefacts in order to evaluate model robustness and simulate plausible clinical noise scenarios.

## 1. Motivation & Applicability

Electrocardiograms captured via electrodes routinely exhibit low-frequency drift (baseline wander) and band-limited instrumentation/muscle noise. These artefacts are:

- **Physiologically plausible**: clinicians expect minor baseline oscillations or jittery traces due to patient movement or lead contact issues.
- **Model-relevant**: classifiers not exposed to such noise distributions during training may misclassify when tested on real-world acquisition environments.

Adding controlled amounts of these noise processes allows us to:

1. Benchmark model sensitivity to realistic artefacts.
2. Combine them with adversarial perturbations for composite threat models.
3. Stress-test detection thresholds and signal processing pipelines.

## 2. Interface & Configuration

Both noise families adopt the shared `PerturbationConfig` interface:

```python
from perturbations import PerturbationConfig, apply_perturbation

config = PerturbationConfig(
    ptype='band_noise',     # or 'baseline_wander'
    strength=0.3,
    center_time=5.0,
    window_seconds=None,   # None -> entire 10 s segment
    extra={'beta': 0.1, 'band': (5.0, 40.0)}
)
x_adv = apply_perturbation(x_clean, fs=100, config=config)
```

Common parameters:

- `strength ∈ [0,1]`: normalized knob controlling amplitude relative to per-lead standard deviation.
- `center_time` + `window_seconds`: define a Hann-windowed temporal mask. If `window_seconds` is `None`, noise spans the entire segment; otherwise it localizes around `center_time`.
- `extra`: optional type-specific overrides (frequency ranges, amplitude coefficients).

## 3. Baseline Wander (`ptype='baseline_wander'`)

### 3.1 Theory

Baseline wander arises from respiratory motion, electrode impedance changes, or patient movement. It manifests as low-frequency oscillations (<0.5 Hz) that shift the entire waveform up or down. Clinicians can still interpret ECG morphology, but classifiers may misinterpret drifting baselines as ST elevation/depression or misalign R-peak detectors.

### 3.2 Implementation (`baseline_wander` function)

Location: `perturbations/noise.py:28-70`.

Steps:

1. **Input preparation**: ensure signal shape is `(samples, leads)` and compute per-lead standard deviations (`_per_lead_std`).
2. **Parameter sampling**:
   - Frequency `f_bw` sampled uniformly from `extra['f_range']` (default (0.05, 0.5) Hz).
   - Phase `φ_c` random per lead.
   - Amplitude `A_c = strength * α * std_c` with `α` defaulting to 0.1.
3. **Signal generation**: `baseline = A_c * sin(2π f t + φ_c)` computed per lead and per sample (vectorized).
4. **Time mask**: multiply by the Hann mask built from `center_time`/`window_seconds` so the baseline drift fades in/out smoothly.
5. **Addition**: `x_adv = x + baseline`.

### 3.3 Practical Usage

- Use `strength` values in [0.1, 0.5] to simulate mild-to-moderate baseline drift. Higher strengths may yield unrealistic or clinically suspicious traces.
- Combine with the evaluation helpers:

```python
noise_results = []
for idx in sample_indices:
    cfg = PerturbationConfig('baseline_wander', 0.4, center_time=4.0, window_seconds=3.0)
    res, x_adv = run_attack_on_sample(model, X_test[idx], y_test_enc[idx], cfg, fs=100)
    noise_results.append(res)
```

- Visualize with `plot_triptych` to verify that the waveform remains interpretable.

## 4. Band-Limited Noise (`ptype='band_noise'`)

### 4.1 Theory

Band-limited noise captures higher-frequency jitter caused by muscle activity, powerline interference, or device circuitry. Typical frequency ranges fall between 5–40 Hz. Such noise adds localized fuzziness without dramatically altering the baseline, yet can smear QRS complexes or distort ST segments.

### 4.2 Implementation (`band_limited_noise` function)

Location: `perturbations/noise.py:87-137`.

Steps:

1. **Generate white noise** per lead (`np.random.normal`).
2. **Frequency-domain filtering**:
   - Compute real FFT.
   - Zero out frequencies outside `extra['band']` (default 5–40 Hz).
   - Inverse FFT to obtain band-limited time-domain noise.
3. **Normalize** to unit variance per lead to ensure consistent scaling.
4. **Scale** by `strength * β * std_c`, with `β` defaulting to 0.1.
5. **Mask** with the same Hann window approach to localize noise in time.
6. **Add** to the original signal.

### 4.3 Practical Usage

- Suitable `strength` values typically lie between 0.1 and 0.4.
- Adjust `extra['band']` to simulate powerline hum (e.g., (58, 62) Hz) or muscle noise (20–50 Hz).
- Evaluate metrics the same way as baseline wander.
- Example snippet:

```python
config = PerturbationConfig(
    ptype='band_noise',
    strength=0.25,
    center_time=5.0,
    window_seconds=4.0,
    extra={'band': (10.0, 35.0), 'beta': 0.15}
)
x_noisy = apply_perturbation(x_clean, fs=100, config=config)
```

## 5. Integration with Evaluation Pipeline

- `run_attack_on_sample` automatically logs noise perturbations with norms, predictions, and smoothness metrics.
- `compute_untargeted_asr` quantifies label flips induced by noise.
- `summarize_norms` and `plot_norm_distributions` contextualize perturbation budgets relative to clean data.
- Notebook cells in `StartingFile.ipynb` demonstrate clean vs noisy evaluation and triptych visualizations.

## 6. Summary

Acquisition noise perturbations provide a lightweight but powerful way to emulate real-world ECG artefacts. By parameterizing time windows, strengths, and frequency bands, they enable systematic robustness testing while keeping perturbations firmly grounded in physiological plausibility. Use them as standalone stressors, baselines for comparison against adversarial attacks, or building blocks in composite perturbation pipelines.

