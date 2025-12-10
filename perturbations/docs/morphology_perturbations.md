# Morphology-Aware Perturbations

This document presents a detailed overview of the morphology-aware perturbations implemented in `perturbations/morphology.py`. These perturbations manipulate ECG morphology (amplitudes and local timing) around detected beats while preserving overall plausibility.

## 1. Rationale

Certain clinical conditions manifest as subtle changes to QRS or T-wave shapes (e.g., hypertrophy, ischemia). By explicitly modifying beat morphology, we can:

- Probe model sensitivity to interpretable waveform changes.
- Simulate targeted physiological variations rather than generic noise.
- Build intuitive adversarial scenarios, such as slightly widening QRS complexes or scaling T-wave amplitudes.

This complements the smooth gradient attacks by providing user-controlled, interpretable perturbations rooted in ECG physiology.

## 2. Dependencies & Beat Detection

- **R-peak detection** uses WFDB’s `xqrs_detect` (`detect_r_peaks` helper). It operates on a single lead (default lead II) and returns sample indices of detected beats.
- The repository already lists `wfdb` in `requirements.txt`, so the detector is available in the existing virtual environment.
- For batch experiments, precompute `r_peaks` for all test samples and cache them to avoid repeated detection overhead.

Example:

```python
from perturbations import detect_r_peaks

r_peaks_cache = [detect_r_peaks(x, fs=100, lead_idx=1) for x in X_test]
```

## 3. Local Amplitude Scaling (`ptype='morph_amp'`)

### 3.1 Theory

Scaling amplitudes around beats can mimic physiological variations such as:

- Increased/decreased R-wave amplitude (e.g., left/right ventricular hypertrophy indications).
- T-wave height changes associated with ischemia or electrolyte imbalances.

By applying a Gaussian weighting centered on R-peaks, we modulate morphology smoothly without introducing discontinuities.

### 3.2 Implementation (`local_amplitude_scaling`)

Location: `perturbations/morphology.py:57-88`.

Steps:

1. **Select beats**: `_select_beats` filters R-peaks based on `center_time` and `window_seconds`. If `window_seconds` is `None`, all beats are eligible.
2. **Gaussian window**: for each targeted beat, construct a Gaussian weighting `exp(-0.5 ((t - r_idx)/σ)^2)` where `σ` is derived from `extra['sigma_ms']` (default 80 ms).
3. **Amplitude gain**: draw `γ ∈ [-γ_max, γ_max]` scaled by `strength`. Default `γ_max = 0.15`.
4. **Apply scaling**: `signal += γ * window * signal`, effectively scaling samples near the R-peak.

### 3.3 Configuration & Usage

```python
config = PerturbationConfig(
    ptype='morph_amp',
    strength=0.4,
    center_time=4.5,
    window_seconds=2.0,
    extra={'gamma_max': 0.12, 'sigma_ms': 60.0},
)
x_morph = apply_perturbation(x_clean, fs=100, config=config, r_peaks=r_peaks)
```

`strength` linearly scales the maximum amplitude gain. Lower strengths yield subtle changes (<5%), while higher strengths approach the `gamma_max` bound (≤15% recommended for plausibility).

### 3.4 Theory-to-ECG Interpretation

- Positive `γ` increases amplitude around the beat (mimicking hypertrophy).
- Negative `γ` decreases amplitude (mimicking low-voltage situations).
- Gaussian weighting ensures surrounding segments taper smoothly, emulating gradual changes rather than sudden jumps.

## 4. Local Time Warping (`ptype='morph_time'`)

### 4.1 Theory

Stretching or compressing time around a beat can simulate variations in QRS duration or ST/T segment length, relevant for conduction delays, bundle branch blocks, or repolarization anomalies.

### 4.2 Implementation (`local_time_warp`)

Location: `perturbations/morphology.py:90-133`.

Steps:

1. **Window selection**: for each beat, choose `[start, end)` around the R-peak, defined by `extra['pre_ms']` and `extra['post_ms']` (defaults 60 & 80 ms, adjustable).
2. **Warp factor**: draw `ε ∈ [-ε_max, ε_max]`, where `ε_max = extra['epsilon_max'] * strength` (default `epsilon_max = 0.1`).
3. **Interpolation**:
   - Normalize time to `[0,1]` within the window.
   - Compute warped positions `τ' = τ * (1 + ε)` and map back to sample indices.
   - Use `np.interp` to resample each lead accordingly.
4. **Blending**: interpolate between original and warped segments with a linear blend to avoid discontinuities at window boundaries.

### 4.3 Configuration & Usage

```python
config = PerturbationConfig(
    ptype='morph_time',
    strength=0.3,
    center_time=4.5,
    window_seconds=2.0,
    extra={'epsilon_max': 0.1, 'pre_ms': 80.0, 'post_ms': 120.0},
)
x_warped = apply_perturbation(x_clean, fs=100, config=config, r_peaks=r_peaks)
```

### 4.4 Interpretation

- Positive `ε` stretches the local window, widening QRS/ST segments.
- Negative `ε` compresses, making waveforms narrower.
- Because warping is local and blended, the overall rhythm and baseline remain intact.

## 5. Integration with Evaluation Pipeline

- `apply_perturbation` now accepts `r_peaks` and dispatches to `local_amplitude_scaling` or `local_time_warp`.
- `run_attack_on_sample` includes an optional `r_peaks` parameter; pass precomputed peaks to ensure consistent perturbations.
- Notebook demo (“Morphology-Aware Perturbations”) detects R-peaks for a sample, applies both morph_amp and morph_time, and visualizes them with `plot_triptych`.

## 6. ECG-Theory Considerations

- **Lead selection**: `detect_r_peaks` defaults to lead II; for certain morphologies (e.g., V1-V2), detect on a more informative lead.
- **Strength ranges**: keep `strength` ≤ 0.5 to avoid unrealistic morphology distortions.
- **Window tuning**: align `center_time`/`window_seconds` with physiological events (e.g., set `center_time` near the targeted beat).
- **Clinical plausibility**: inspect perturbations with cardiologist feedback or overlay plots to ensure they remain within plausible bounds.

## 7. Example Workflow

```python
# Precompute R-peaks
r_peaks_sample = detect_r_peaks(X_test[idx], fs=sampling_rate)

# Amplitude scaling
morph_amp_cfg = PerturbationConfig(
    ptype='morph_amp',
    strength=0.35,
    center_time=3.0,
    window_seconds=1.5,
)
x_amp = apply_perturbation(X_test[idx], fs=100, config=morph_amp_cfg, r_peaks=r_peaks_sample)

# Time warp
morph_time_cfg = PerturbationConfig(ptype='morph_time', strength=0.25, center_time=3.0, window_seconds=1.5)
x_time = apply_perturbation(X_test[idx], fs=100, config=morph_time_cfg, r_peaks=r_peaks_sample)

# Evaluate predictions
res_amp, _ = run_attack_on_sample(model, X_test[idx], y_test_enc[idx], morph_amp_cfg, fs=100, r_peaks=r_peaks_sample)
res_time, _ = run_attack_on_sample(model, X_test[idx], y_test_enc[idx], morph_time_cfg, fs=100, r_peaks=r_peaks_sample)
```

## 8. Combining with Other Perturbations

- Morphology edits can follow smooth adversarial attacks to add interpretability.
- Alternatively, apply acquisition noise first, then morphology adjustments, to simulate noisy patient settings with subtle physiological changes.
- Always monitor `delta_norm_l2`/`delta_smoothness` to ensure budgets remain within acceptable bounds.

## 9. Summary

Morphology-aware perturbations provide interpretable, physiology-inspired ways to stress-test ECG classifiers. By scaling or warping localized segments around R-peaks, they emulate real clinical variations while adhering to the shared `PerturbationConfig` interface, enabling seamless integration with the broader perturbation and evaluation pipeline.

