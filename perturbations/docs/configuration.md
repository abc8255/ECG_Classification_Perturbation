# Perturbation Configuration Guide

This document explains the configuration interface used throughout the perturbation system. It covers `PerturbationConfig` fields, shared parameters (strength, temporal window), and type-specific `extra` options for each perturbation family.

## 1. PerturbationConfig Recap

```python
@dataclass
class PerturbationConfig:
    ptype: str
    strength: float
    center_time: float
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
```

- `ptype`: selects the perturbation family.
- `strength`: normalized magnitude knob (0 = no perturbation, 1 = maximum allowed for that family).
- `center_time` (seconds) and `window_seconds` define the temporal region of interest.
- `target_class`: used only by perturbations that interact with label semantics (`smooth_adv`).
- `extra`: dictionary for family-specific parameters.

## 2. Shared Parameters

### 2.1 `strength`

- Interpreted differently per family:
  - **Noise**: scales amplitude relative to per-lead standard deviation.
  - **Smooth Adv**: scales the L2 budget `eps_max`.
  - **Morphology**: scales amplitude/time warp limits (e.g., up to ±15% by default).
- Recommended range: 0.1–0.5 for physiologically plausible perturbations.

### 2.2 `center_time` & `window_seconds`

- `center_time`: seconds from the start of the 10 s segment (0 ≤ `center_time` ≤ 10).
- `window_seconds`:
  - If `None`: perturbation applies to the full segment.
  - Else: Hann window of length `window_seconds` is centered at `center_time` to fade perturbations in/out smoothly.
- Applicable to all families; morphological operations use it to select subsets of beats.

### 2.3 `target_class`

- Only relevant for `smooth_adv`.
- Example values: `'CD'`, `'HYP'`, `'MI'`, `'NORM'`, `'STTC'`.
- Works in tandem with `extra['target_mode']` and `extra['target_value']`.

## 3. Type-Specific `extra` Options

### 3.1 Baseline Wander (`ptype='baseline_wander'`)

| Key         | Default        | Description                                      |
|-------------|----------------|--------------------------------------------------|
| `alpha`     | 0.1            | Base amplitude coefficient (scaled by strength). |
| `f_range`   | (0.05, 0.5) Hz | Frequency range for sinusoidal drift.            |

### 3.2 Band-Limited Noise (`ptype='band_noise'`)

| Key       | Default        | Description                                                |
|-----------|----------------|------------------------------------------------------------|
| `beta`    | 0.1            | Amplitude coefficient (scaled by strength).                |
| `band`    | (5.0, 40.0) Hz | Passband for FFT-based filtering (can be narrower).       |

### 3.3 Smooth Adversary (`ptype='smooth_adv'`)

| Key              | Default          | Description                                                  |
|------------------|------------------|--------------------------------------------------------------|
| `eps_global_max` | 0.5              | Global L2 norm cap (before multiplying by `strength`).       |
| `lambda_smooth`  | 10.0             | Smoothness penalty weight.                                   |
| `lambda_energy`  | 0.5              | Energy penalty weight.                                       |
| `steps`          | 200              | Gradient descent iterations.                                 |
| `lr`             | 0.01             | Adam optimizer learning rate.                                |
| `target_mode`    | `'force'`        | `'force'` to drive bit to 1; `'suppress'` to drive to 0.     |
| `target_value`   | 1.0              | Desired probability when forcing a class.                    |
| `record_id`      | *user-defined*   | Optional metadata stored in `AttackResult`.                  |

### 3.4 Morphology Amplitude (`ptype='morph_amp'`)

| Key        | Default | Description                                               |
|------------|---------|-----------------------------------------------------------|
| `gamma_max`| 0.15    | Maximum amplitude change (±15%) before scaling by strength.|
| `sigma_ms` | 80.0 ms | Standard deviation of Gaussian window around R-peak.      |

### 3.5 Morphology Time Warp (`ptype='morph_time'`)

| Key          | Default | Description                                                         |
|--------------|---------|---------------------------------------------------------------------|
| `epsilon_max`| 0.1     | Max time stretch/compress percentage (scaled by strength).          |
| `pre_ms`     | 60.0 ms | Samples before R-peak to include in the warp window.                |
| `post_ms`    | 80.0 ms | Samples after R-peak to include in the warp window.                 |

## 4. Examples

### 4.1 Noise Perturbation

```python
cfg = PerturbationConfig(
    ptype='band_noise',
    strength=0.25,
    center_time=5.0,
    window_seconds=4.0,
    extra={'band': (10.0, 35.0), 'beta': 0.15},
)
x_noisy = apply_perturbation(x_clean, fs=100, config=cfg)
```

### 4.2 Smooth Adversarial Attack

```python
cfg = PerturbationConfig(
    ptype='smooth_adv',
    strength=0.3,
    center_time=4.5,
    window_seconds=2.0,
    target_class='MI',
    extra={'eps_global_max': 0.4, 'steps': 150, 'target_mode': 'force'},
)
x_adv = apply_perturbation(
    x_clean,
    fs=100,
    config=cfg,
    model=model,
    y_true=y_true_vec,
)
```

### 4.3 Morphology Time Warp

```python
r_peaks = detect_r_peaks(x_clean, fs=100)
cfg = PerturbationConfig(
    ptype='morph_time',
    strength=0.35,
    center_time=3.0,
    window_seconds=1.0,
    extra={'epsilon_max': 0.12, 'pre_ms': 80.0, 'post_ms': 120.0},
)
x_warped = apply_perturbation(x_clean, fs=100, config=cfg, r_peaks=r_peaks)
```

## 5. Best Practices

- Keep `strength` moderate (<0.5) for realistic perturbations unless intentionally stress-testing extreme conditions.
- Always visualize perturbations (e.g., `plot_triptych`) before running large-scale evaluations to ensure plausibility.
- Use deterministic RNG seeds (`np.random.default_rng(seed)`) for reproducible experiments.
- Record `record_id` or experiment identifiers in `extra` to trace `AttackResult`s back to specific configurations.

With this guide, you can confidently configure each perturbation family, understand what each parameter controls, and customize `extra` settings to match your experimental goals.

