# Perturbation API Overview

This document describes the public API layer exposed via `perturbations/__init__.py` and implemented in `perturbations/api.py`. It explains how the dispatcher works, how each perturbation family is routed, and how to integrate the API into notebooks or scripts.

## 1. Key Components

- **`PerturbationConfig`** (`perturbations/api.py:12-23`): dataclass capturing perturbation type, strength, temporal window, targeting info, and a flexible `extra` dict for type-specific parameters.
- **`apply_perturbation`** (`api.py:30-101`): central dispatcher that accepts an ECG segment and applies the configured perturbation, delegating to the appropriate implementation (noise, smooth adversary, morphology).
- **Helper exports**: `perturbations/__init__.py` also re-exports evaluation and visualization helpers for convenience (see `docs/evaluation_utilities.md` and `visualization/visualization_utilities.md`).

## 2. PerturbationConfig Fields

```python
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class PerturbationConfig:
    ptype: str                 # perturbation family identifier
    strength: float            # normalized [0, 1] magnitude
    center_time: float         # seconds from start of 10 s segment
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
```

- `ptype`: one of `baseline_wander`, `band_noise`, `smooth_adv`, `morph_amp`, `morph_time`. Additional families can be added later.
- `strength`: high-level knob interpreted per family (see `docs/configuration.md` for details).
- `center_time`/`window_seconds`: define where in time the perturbation is focused. If `window_seconds` is `None`, the perturbation spans the full segment.
- `target_class`: used by `smooth_adv` for targeted attacks; irrelevant for noise/morphology.# Design Document: Plausible ECG Perturbations for Targeted Misclassification

## 1. Context & Goals
We train and evaluate a multi-label 1D CNN in `StartingFile.ipynb` that ingests PTBâ€‘XL ECG segments stored under `ptb-xl/`. Each record is a 10â€¯s, 12â€‘lead trace sampled at 100â€¯Hz (1â€¯000 samples Ã— 12 leads). Labels are aggregated with `MultiLabelBinarizer` into the five diagnostic superclasses `['CD', 'HYP', 'MI', 'NORM', 'STTC']`, and the model is optimized with `binary_crossentropy` plus accuracy and multi-label AUC. The objective of this work is to overlay small, plausible perturbations on top of these preprocessed tensors so that the trained TensorFlow model misclassifies either untargetedly (any change in the predicted label set) or in a targeted fashion (force or suppress a specific superclass), while keeping perturbations visually consistent with realistic physiology or acquisition artefacts.

## 2. Reference Training Pipeline Snapshot

### 2.1 Dataset & Labels
- CSV metadata (`ptbxl_database.csv`) is read into `Y`, and `scp_codes` are expanded via `ast.literal_eval`.
- Raw signals are loaded with `wfdb.rdsamp` using `filename_lr` (100â€¯Hz) and stacked into `X` with shape `(num_records, 1000, 12)`.
- Diagnostic subclasses are aggregated through `scp_statements.csv`, filtered to the `diagnostic` rows, and mapped onto the five superclasses above.
- A stratified split uses `strat_fold != 10` for training and `== 10` for testing, resulting in `(X_train, y_train)` and `(X_test, y_test)`. Encoding with `MultiLabelBinarizer` produces `y_*_enc` arrays of shape `(num_records, 5)`.

### 2.2 Model
```
Sequential(
  Input(shape=(1000, 12)),
  Conv1D(32, kernel_size=5, activation='relu'),
  MaxPooling1D(pool_size=2),
  Conv1D(64, kernel_size=5, activation='relu'),
  MaxPooling1D(pool_size=2),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(5, activation='sigmoid')
)
```
This architecture expects `float32` tensors shaped `(batch, 1000, 12)` and outputs independent probabilities per superclass.

### 2.3 Loss & Metrics
- Optimization: `adam`.
- Loss: `binary_crossentropy` between the sigmoid outputs and the five-dimensional label vector.
- Metrics: accuracy and `tf.keras.metrics.AUC(multi_label=True, num_labels=5)`.

The perturbation pipeline must preserve these conventions: keep tensors in the `(1000, 12)` layout, respect the five-label scheme, and call the already-compiled model for logits/probabilities.

## 3. Data & Preprocessing Assumptions
- Signals remain in the raw microvolt scale provided by WFDB; no additional normalization is currently applied. Any perturbation must therefore be generated in the same scale so gradients remain meaningful.
- Optionally, we can compute per-record means/standard deviations to report perturbation magnitudes; however, those statistics may not be used for normalization unless the training notebook is updated accordingly.
- When batching perturbations, reshape single samples to `(1, 1000, 12)` before feeding the TensorFlow model, mirroring the `model.fit` call.

## 4. Threat Model & Plausibility Constraints

### 4.1 Threat Model
- **Type I (digital) white-box:** we directly edit stored test signals (`X_test`) and have access to model parameters and gradients via TensorFlow.
- **Objectives:**
  - Untargeted: alter the predicted label set relative to the baseline set obtained with a 0.5 sigmoid threshold.
  - Targeted: force a specific superclass to be predicted (set probability â‰¥ threshold) or suppress an existing positive class (probability < threshold).
- **Budget:** perturbations must stay within an L2 norm scaled by the `strength` knob and remain temporally localized via the `center_time` / `window_seconds` interface.

### 4.2 Plausibility Constraints
- Preserve recognizable Pâ€“QRSâ€“T morphology and heart rhythm.
- Keep changes smooth in time and coherent across leads (no single-sample spikes).
- Limit edits to windows that can pass as baseline wander, powerline noise, or small morphology variations.
- Provide smooth onset/offset windows (Hann or cosine masks) and penalize high first-order derivatives of the perturbation.

## 5. Perturbation Engine Architecture

### 5.1 Core API
```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
import tensorflow as tf

@dataclass
class PerturbationConfig:
    ptype: str                         # 'smooth_adv', 'baseline_wander', 'band_noise', 'morph_amp', 'morph_time'
    strength: float                    # [0, 1] logical knob
    center_time: float                 # seconds from the start of the 10 s segment
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None # e.g., 'MI'; None for untargeted
    extra: Optional[Dict[str, Any]] = None
```

```python
def apply_perturbation(
    x: np.ndarray,                     # shape (1000, 12)
    fs: float,                         # 100 Hz for the current notebook
    config: PerturbationConfig,
    model: Optional[tf.keras.Model] = None,
    r_peaks: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Dispatch to the configured perturbation. Returns x_adv with the same shape as x.
    """
```

### 5.2 Integration Points
- Training remains unchanged. For evaluation, we copy a clean sample, call `apply_perturbation`, reshape to `(1, 1000, 12)`, and run `model.predict` to observe label flips.
- All perturbation functions operate on NumPy arrays for compatibility with the notebook; gradient-based ones temporarily convert to `tf.Tensor` for differentiation.

### 5.3 Perturbation Families
1. **Smooth adversarial perturbations** (`smooth_adv`) â€” gradient-driven, smooth, windowed.
2. **Acquisition artefact simulations** (`baseline_wander`, `band_noise`) â€” parametric noise injections.
3. **Morphology-aware local warps** (`morph_amp`, `morph_time`) â€” interpretable edits tied to detected beats.

## 6. Perturbation Family 1: Smooth Gradient-Based Adversary (`smooth_adv`)

### 6.1 Objective
We optimize a mask-localized perturbation `Î´` that maximizes misclassification while staying smooth and energy-bounded:

\[
L(\delta) = w_{\text{cls}} L_{\text{cls}}(x + \delta) + \lambda_{\text{smooth}} L_{\text{smooth}}(\delta) + \lambda_{\text{energy}} L_{\text{energy}}(\delta)
\]

- `L_cls`: attack loss. Untargeted attacks maximize the baseline loss by setting `L_cls = -BCE(y_true, y_pred)`; targeted attacks minimize `BCE(y_target, y_pred)`, where `y_target` copies `y_true` but enforces the requested superclass bit (1 to force, 0 to suppress) in accordance with multi-label semantics.
- `L_smooth`: mean variance of the first-order differences across time and leads.
- `L_energy`: average squared magnitude.

### 6.2 Time Window Mask
Given `center_time` and `window_seconds` (default 2â€¯s), we compute sample indices at 100â€¯Hz, derive `[start, end)` bounds, and create a Hann window of length `end - start` that is embedded into a zero mask for the entire 10â€¯s record. The mask broadcasts over 12 leads to localize the perturbation smoothly.

### 6.3 Strength Mapping
- `strength` âˆˆ [0, 1].
- `eps_global_max` (default 0.5) is expressed in the normalized input space (raw WFDB units for the current notebook). `eps_max = strength * eps_global_max`.
- After each optimizer step we rescale `Î´` to satisfy `||Î´||â‚‚ â‰¤ eps_max`.
- `strength` can additionally scale `Î»_smooth` and `Î»_energy` (larger strengths relax the penalties).

### 6.4 TensorFlow Implementation Sketch
```python
def build_time_mask(L, fs, center_time, window_seconds):
    mask = np.zeros(L, dtype=np.float32)
    if window_seconds is None:
        return mask + 1.0  # full segment
    center_idx = int(center_time * fs)
    half = int(window_seconds * fs / 2)
    start = max(0, center_idx - half)
    end = min(L, center_idx + half)
    window = tf.signal.hann_window(end - start)
    mask[start:end] = window.numpy()
    return tf.convert_to_tensor(mask)

def smooth_adversarial_perturbation(x_np, fs, config, model, y_true_vec):
    x = tf.convert_to_tensor(x_np[None, ...], dtype=tf.float32)    # (1, 1000, 12)
    delta_param = tf.Variable(tf.zeros_like(x), trainable=True)
    mask = build_time_mask(x.shape[1], fs, config.center_time, config.window_seconds)
    mask = tf.reshape(mask, (1, -1, 1))                            # broadcast over leads
    opt = tf.keras.optimizers.Adam(config.extra.get("lr", 0.01))
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    eps_max = config.extra.get("eps_global_max", 0.5) * config.strength
    lambda_smooth = config.extra.get("lambda_smooth", 10.0)
    lambda_energy = config.extra.get("lambda_energy", 0.5)
    steps = config.extra.get("steps", 200)

    target_vec = y_true_vec.copy()
    if config.target_class:
        idx = CLASS_TO_INDEX[config.target_class]
        desired = config.extra.get("target_value", 1.0)
        target_vec[idx] = desired

    for _ in range(steps):
        with tf.GradientTape() as tape:
            delta = mask * delta_param
            norm = tf.norm(delta)
            delta = tf.cond(
                norm > eps_max,
                lambda: delta * (eps_max / (norm + 1e-8)),
                lambda: delta,
            )
            x_adv = x + delta
            y_pred = model(x_adv, training=False)
            cls_loss = -bce(y_true_vec, y_pred) if config.target_class is None else bce(target_vec, y_pred)
            smooth_loss = tf.reduce_mean(tf.math.squared_difference(delta[:, 1:, :], delta[:, :-1, :]))
            energy_loss = tf.reduce_mean(tf.square(delta))
            loss = cls_loss + lambda_smooth * smooth_loss + lambda_energy * energy_loss
        grads = tape.gradient(loss, [delta_param])
        opt.apply_gradients(zip(grads, [delta_param]))

    delta = mask * delta_param
    norm = tf.norm(delta)
    if eps_max > 0 and norm > eps_max:
        delta = delta * (eps_max / (norm + 1e-8))
    return (x + delta)[0].numpy(), delta[0].numpy()
```

### 6.5 Plausibility Checks
- Log per-sample L2/Lâˆž norms in both normalized and ÂµV units.
- Track `L_smooth` and energy penalties; abort or downscale when limits are exceeded.
- Overlay `x`/`x_adv` for random records and ensure thresholded predictions changed as intended.

## 7. Perturbation Family 2: Acquisition-Style Noise

### 7.1 Baseline Wander (`baseline_wander`)
- Generate per-lead low-frequency sinusoids `b_c(t) = A_c sin(2Ï€ f t + Ï†_c)` with `f âˆˆ [0.05, 0.5]` Hz and random phases.
- Amplitude `A_c = strength * Î± * std_c`, where `std_c` is the per-lead standard deviation computed after whatever preprocessing is applied to the model input (currently raw values).
- Apply the same Hann-mask window before adding to `x`.

### 7.2 Band-Limited Noise (`band_noise`)
- Create white noise per lead, filter with a Butterworth band-pass (default 5â€“40â€¯Hz), rescale to unit variance, then multiply by `strength * Î² * std_c`.
- Window with the same mask and add to `x`.

### 7.3 Combining with Adversarial Î´
- Generate `x_adv` via `smooth_adv`.
- Draw a noise config (`baseline_wander` or `band_noise`) with its own `strength`, apply it to `x_adv`, and feed the result to the model.

## 8. Perturbation Family 3: Morphology-Aware Warps

### 8.1 Beat Detection
- Use `wfdb.processing.xqrs_detect` (or a similar algorithm already bundled with WFDB, a dependency in `requirements.txt`) to precompute R-peak indices per record at 100â€¯Hz.
- Cache results next to `X_test` to avoid recomputation inside the notebook.

### 8.2 Local Amplitude Scaling (`morph_amp`)
- For beats within the selected time window, multiply samples in a Gaussian neighborhood (Ïƒ â‰ˆ 80â€¯ms by default) by a gain `1 + Î³`, where `Î³ âˆˆ [-Î³_max, Î³_max]` and `Î³_max` scales with `strength` (e.g., 0.03 at strength 0.2, up to 0.15 at strength 1.0).
- Gains can be independent per lead or tied across limb/precordial groups to maintain clinical plausibility.

### 8.3 Local Time Warping (`morph_time`)
- Select `[a, b]` windows surrounding each targeted beat (e.g., `r_k - 40` to `r_k + 60` samples).
- Define a warp factor `Îµ âˆˆ [-Îµ_max, Îµ_max]`, scale `Ï„` âˆˆ [0, 1] to `(1 + Îµ)`, clamp to [0,1], and resample the window via linear interpolation to stretch or compress the local waveform slightly.
- Enforce continuity at window edges by blending with the original signal using the Hann mask.

### 8.4 Strength Interpretation
- `strength` controls Î³ and Îµ maxima; keep them â‰¤ 15% so morphology changes remain subtle.
- Enforce per-beat caps (e.g., only edit one or two beats per 10â€¯s window for strength â‰¤ 0.3).

## 9. Interface: Strength & Time Knobs
- `strength` is a normalized knob interpreted per perturbation family:
  - `smooth_adv`: L2 budget.
  - `baseline_wander` / `band_noise`: amplitude relative to per-lead std.
  - `morph_amp` / `morph_time`: bounding box for amplitude/time scaling.
- `center_time` specifies the focal point in seconds (0â€¯â‰¤â€¯center_timeâ€¯â‰¤â€¯10). `window_seconds` defaults to 2â€¯s but can be overridden. Both are applied uniformly regardless of family to keep the user API consistent.

Example defaults:
```python
DEFAULTS = {
    "smooth_adv": {"eps_global_max": 0.5, "lambda_smooth": 10.0, "lambda_energy": 0.5},
    "baseline_wander": {"alpha": 0.1, "f_range": (0.05, 0.5)},
    "band_noise": {"beta": 0.1, "band": (5.0, 40.0)},
    "morph_amp": {"gamma_max": 0.15, "sigma_ms": 80.0},
    "morph_time": {"epsilon_max": 0.1},
}
```

## 10. Module Layout & Notebook Integration
```
perturbations/
  __init__.py
  config.py            # defaults, class-to-index map
  masks.py             # time-window utilities for NumPy / TensorFlow
  metrics.py           # smoothness, L2/Lâˆž helpers
  adv_smooth.py        # TensorFlow implementation of smooth_adv
  noise.py             # baseline_wander and band_noise
  morphology.py        # morph_amp / morph_time utilities
  api.py               # PerturbationConfig + apply_perturbation dispatcher
```
Usage inside `StartingFile.ipynb`:
```python
from perturbations.api import PerturbationConfig, apply_perturbation

config = PerturbationConfig(ptype='smooth_adv', strength=0.3, center_time=4.5,
                            window_seconds=2.0, target_class='MI')
x_clean = X_test[idx]
x_adv = apply_perturbation(x_clean, fs=100, config=config, model=model, 
                           r_peaks=r_peaks_cache[idx])
pred_clean = (model.predict(x_clean[None, ...]) >= 0.5)
pred_adv = (model.predict(x_adv[None, ...]) >= 0.5)
```

## 11. Evaluation Plan
1. **Attack Success (ASR):**
   - Compare predicted label vectors (threshold 0.5) before and after perturbation.
   - Report untargeted ASR and per-class targeted success rates across `X_test`.
2. **Perturbation Budgets:**
   - L2/Lâˆž norms per record alongside strength values.
   - Smoothness metrics vs. baseline variability between beats.
3. **Plausibility:**
   - Visual overlays of clean vs. perturbed signals for random samples (per lead).
   - Optional domain-expert review of a curated set.
4. **Quantitative Metrics:**
   - ROC/AUC degradation after applying stochastic noise families.
   - Compare `smooth_adv` vs. `smooth_adv + noise` cascades.
5. **Logging:**
   - Store metadata per sample: target class, achieved class probability, norm values, and mask window.

## 12. Suggested Implementation Order
1. **Noise families:** easiest to validate; ensure the time-window mask and `PerturbationConfig` plumbing work with raw NumPy arrays.
2. **Gradient-based attack:** start without a window, validate `tf.GradientTape` loop against a single sample, then add windowing, clipping, and targeted modes.
3. **Morphology-aware warps:** build the R-peak cache, implement amplitude scaling, then the time-warp interpolation.
4. **Evaluation utilities:** thresholded label comparison, plotting helpers, and summary tables/notebook cells that call the new API against `X_test`.

With this alignment, the perturbation engine can be dropped into the existing TensorFlow notebook without altering the training code, and every section above mirrors assumptions already present in the repository.
# Design Document: Plausible ECG Perturbations for Targeted Misclassification

## 1. Context & Goals
We train and evaluate a multi-label 1D CNN in `StartingFile.ipynb` that ingests PTBâ€‘XL ECG segments stored under `ptb-xl/`. Each record is a 10â€¯s, 12â€‘lead trace sampled at 100â€¯Hz (1â€¯000 samples Ã— 12 leads). Labels are aggregated with `MultiLabelBinarizer` into the five diagnostic superclasses `['CD', 'HYP', 'MI', 'NORM', 'STTC']`, and the model is optimized with `binary_crossentropy` plus accuracy and multi-label AUC. The objective of this work is to overlay small, plausible perturbations on top of these preprocessed tensors so that the trained TensorFlow model misclassifies either untargetedly (any change in the predicted label set) or in a targeted fashion (force or suppress a specific superclass), while keeping perturbations visually consistent with realistic physiology or acquisition artefacts.

## 2. Reference Training Pipeline Snapshot

### 2.1 Dataset & Labels
- CSV metadata (`ptbxl_database.csv`) is read into `Y`, and `scp_codes` are expanded via `ast.literal_eval`.
- Raw signals are loaded with `wfdb.rdsamp` using `filename_lr` (100â€¯Hz) and stacked into `X` with shape `(num_records, 1000, 12)`.
- Diagnostic subclasses are aggregated through `scp_statements.csv`, filtered to the `diagnostic` rows, and mapped onto the five superclasses above.
- A stratified split uses `strat_fold != 10` for training and `== 10` for testing, resulting in `(X_train, y_train)` and `(X_test, y_test)`. Encoding with `MultiLabelBinarizer` produces `y_*_enc` arrays of shape `(num_records, 5)`.

### 2.2 Model
```
Sequential(
  Input(shape=(1000, 12)),
  Conv1D(32, kernel_size=5, activation='relu'),
  MaxPooling1D(pool_size=2),
  Conv1D(64, kernel_size=5, activation='relu'),
  MaxPooling1D(pool_size=2),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(5, activation='sigmoid')
)
```
This architecture expects `float32` tensors shaped `(batch, 1000, 12)` and outputs independent probabilities per superclass.

### 2.3 Loss & Metrics
- Optimization: `adam`.
- Loss: `binary_crossentropy` between the sigmoid outputs and the five-dimensional label vector.
- Metrics: accuracy and `tf.keras.metrics.AUC(multi_label=True, num_labels=5)`.

The perturbation pipeline must preserve these conventions: keep tensors in the `(1000, 12)` layout, respect the five-label scheme, and call the already-compiled model for logits/probabilities.

## 3. Data & Preprocessing Assumptions
- Signals remain in the raw microvolt scale provided by WFDB; no additional normalization is currently applied. Any perturbation must therefore be generated in the same scale so gradients remain meaningful.
- Optionally, we can compute per-record means/standard deviations to report perturbation magnitudes; however, those statistics may not be used for normalization unless the training notebook is updated accordingly.
- When batching perturbations, reshape single samples to `(1, 1000, 12)` before feeding the TensorFlow model, mirroring the `model.fit` call.

## 4. Threat Model & Plausibility Constraints

### 4.1 Threat Model
- **Type I (digital) white-box:** we directly edit stored test signals (`X_test`) and have access to model parameters and gradients via TensorFlow.
- **Objectives:**
  - Untargeted: alter the predicted label set relative to the baseline set obtained with a 0.5 sigmoid threshold.
  - Targeted: force a specific superclass to be predicted (set probability â‰¥ threshold) or suppress an existing positive class (probability < threshold).
- **Budget:** perturbations must stay within an L2 norm scaled by the `strength` knob and remain temporally localized via the `center_time` / `window_seconds` interface.

### 4.2 Plausibility Constraints
- Preserve recognizable Pâ€“QRSâ€“T morphology and heart rhythm.
- Keep changes smooth in time and coherent across leads (no single-sample spikes).
- Limit edits to windows that can pass as baseline wander, powerline noise, or small morphology variations.
- Provide smooth onset/offset windows (Hann or cosine masks) and penalize high first-order derivatives of the perturbation.

## 5. Perturbation Engine Architecture

### 5.1 Core API
```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
import tensorflow as tf

@dataclass
class PerturbationConfig:
    ptype: str                         # 'smooth_adv', 'baseline_wander', 'band_noise', 'morph_amp', 'morph_time'
    strength: float                    # [0, 1] logical knob
    center_time: float                 # seconds from the start of the 10 s segment
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None # e.g., 'MI'; None for untargeted
    extra: Optional[Dict[str, Any]] = None
```

```python
def apply_perturbation(
    x: np.ndarray,                     # shape (1000, 12)
    fs: float,                         # 100 Hz for the current notebook
    config: PerturbationConfig,
    model: Optional[tf.keras.Model] = None,
    r_peaks: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Dispatch to the configured perturbation. Returns x_adv with the same shape as x.
    """
```

### 5.2 Integration Points
- Training remains unchanged. For evaluation, we copy a clean sample, call `apply_perturbation`, reshape to `(1, 1000, 12)`, and run `model.predict` to observe label flips.
- All perturbation functions operate on NumPy arrays for compatibility with the notebook; gradient-based ones temporarily convert to `tf.Tensor` for differentiation.

### 5.3 Perturbation Families
1. **Smooth adversarial perturbations** (`smooth_adv`) â€” gradient-driven, smooth, windowed.
2. **Acquisition artefact simulations** (`baseline_wander`, `band_noise`) â€” parametric noise injections.
3. **Morphology-aware local warps** (`morph_amp`, `morph_time`) â€” interpretable edits tied to detected beats.

## 6. Perturbation Family 1: Smooth Gradient-Based Adversary (`smooth_adv`)

### 6.1 Objective
We optimize a mask-localized perturbation `Î´` that maximizes misclassification while staying smooth and energy-bounded:

\[
L(\delta) = w_{\text{cls}} L_{\text{cls}}(x + \delta) + \lambda_{\text{smooth}} L_{\text{smooth}}(\delta) + \lambda_{\text{energy}} L_{\text{energy}}(\delta)
\]

- `L_cls`: attack loss. Untargeted attacks maximize the baseline loss by setting `L_cls = -BCE(y_true, y_pred)`; targeted attacks minimize `BCE(y_target, y_pred)`, where `y_target` copies `y_true` but enforces the requested superclass bit (1 to force, 0 to suppress) in accordance with multi-label semantics.
- `L_smooth`: mean variance of the first-order differences across time and leads.
- `L_energy`: average squared magnitude.

### 6.2 Time Window Mask
Given `center_time` and `window_seconds` (default 2â€¯s), we compute sample indices at 100â€¯Hz, derive `[start, end)` bounds, and create a Hann window of length `end - start` that is embedded into a zero mask for the entire 10â€¯s record. The mask broadcasts over 12 leads to localize the perturbation smoothly.

### 6.3 Strength Mapping
- `strength` âˆˆ [0, 1].
- `eps_global_max` (default 0.5) is expressed in the normalized input space (raw WFDB units for the current notebook). `eps_max = strength * eps_global_max`.
- After each optimizer step we rescale `Î´` to satisfy `||Î´||â‚‚ â‰¤ eps_max`.
- `strength` can additionally scale `Î»_smooth` and `Î»_energy` (larger strengths relax the penalties).

### 6.4 TensorFlow Implementation Sketch
```python
def build_time_mask(L, fs, center_time, window_seconds):
    mask = np.zeros(L, dtype=np.float32)
    if window_seconds is None:
        return mask + 1.0  # full segment
    center_idx = int(center_time * fs)
    half = int(window_seconds * fs / 2)
    start = max(0, center_idx - half)
    end = min(L, center_idx + half)
    window = tf.signal.hann_window(end - start)
    mask[start:end] = window.numpy()
    return tf.convert_to_tensor(mask)

def smooth_adversarial_perturbation(x_np, fs, config, model, y_true_vec):
    x = tf.convert_to_tensor(x_np[None, ...], dtype=tf.float32)    # (1, 1000, 12)
    delta_param = tf.Variable(tf.zeros_like(x), trainable=True)
    mask = build_time_mask(x.shape[1], fs, config.center_time, config.window_seconds)
    mask = tf.reshape(mask, (1, -1, 1))                            # broadcast over leads
    opt = tf.keras.optimizers.Adam(config.extra.get("lr", 0.01))
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    eps_max = config.extra.get("eps_global_max", 0.5) * config.strength
    lambda_smooth = config.extra.get("lambda_smooth", 10.0)
    lambda_energy = config.extra.get("lambda_energy", 0.5)
    steps = config.extra.get("steps", 200)

    target_vec = y_true_vec.copy()
    if config.target_class:
        idx = CLASS_TO_INDEX[config.target_class]
        desired = config.extra.get("target_value", 1.0)
        target_vec[idx] = desired

    for _ in range(steps):
        with tf.GradientTape() as tape:
            delta = mask * delta_param
            norm = tf.norm(delta)
            delta = tf.cond(
                norm > eps_max,
                lambda: delta * (eps_max / (norm + 1e-8)),
                lambda: delta,
            )
            x_adv = x + delta
            y_pred = model(x_adv, training=False)
            cls_loss = -bce(y_true_vec, y_pred) if config.target_class is None else bce(target_vec, y_pred)
            smooth_loss = tf.reduce_mean(tf.math.squared_difference(delta[:, 1:, :], delta[:, :-1, :]))
            energy_loss = tf.reduce_mean(tf.square(delta))
            loss = cls_loss + lambda_smooth * smooth_loss + lambda_energy * energy_loss
        grads = tape.gradient(loss, [delta_param])
        opt.apply_gradients(zip(grads, [delta_param]))

    delta = mask * delta_param
    norm = tf.norm(delta)
    if eps_max > 0 and norm > eps_max:
        delta = delta * (eps_max / (norm + 1e-8))
    return (x + delta)[0].numpy(), delta[0].numpy()
```

### 6.5 Plausibility Checks
- Log per-sample L2/Lâˆž norms in both normalized and ÂµV units.
- Track `L_smooth` and energy penalties; abort or downscale when limits are exceeded.
- Overlay `x`/`x_adv` for random records and ensure thresholded predictions changed as intended.

## 7. Perturbation Family 2: Acquisition-Style Noise

### 7.1 Baseline Wander (`baseline_wander`)
- Generate per-lead low-frequency sinusoids `b_c(t) = A_c sin(2Ï€ f t + Ï†_c)` with `f âˆˆ [0.05, 0.5]` Hz and random phases.
- Amplitude `A_c = strength * Î± * std_c`, where `std_c` is the per-lead standard deviation computed after whatever preprocessing is applied to the model input (currently raw values).
- Apply the same Hann-mask window before adding to `x`.

### 7.2 Band-Limited Noise (`band_noise`)
- Create white noise per lead, filter with a Butterworth band-pass (default 5â€“40â€¯Hz), rescale to unit variance, then multiply by `strength * Î² * std_c`.
- Window with the same mask and add to `x`.

### 7.3 Combining with Adversarial Î´
- Generate `x_adv` via `smooth_adv`.
- Draw a noise config (`baseline_wander` or `band_noise`) with its own `strength`, apply it to `x_adv`, and feed the result to the model.

## 8. Perturbation Family 3: Morphology-Aware Warps

### 8.1 Beat Detection
- Use `wfdb.processing.xqrs_detect` (or a similar algorithm already bundled with WFDB, a dependency in `requirements.txt`) to precompute R-peak indices per record at 100â€¯Hz.
- Cache results next to `X_test` to avoid recomputation inside the notebook.

### 8.2 Local Amplitude Scaling (`morph_amp`)
- For beats within the selected time window, multiply samples in a Gaussian neighborhood (Ïƒ â‰ˆ 80â€¯ms by default) by a gain `1 + Î³`, where `Î³ âˆˆ [-Î³_max, Î³_max]` and `Î³_max` scales with `strength` (e.g., 0.03 at strength 0.2, up to 0.15 at strength 1.0).
- Gains can be independent per lead or tied across limb/precordial groups to maintain clinical plausibility.

### 8.3 Local Time Warping (`morph_time`)
- Select `[a, b]` windows surrounding each targeted beat (e.g., `r_k - 40` to `r_k + 60` samples).
- Define a warp factor `Îµ âˆˆ [-Îµ_max, Îµ_max]`, scale `Ï„` âˆˆ [0, 1] to `(1 + Îµ)`, clamp to [0,1], and resample the window via linear interpolation to stretch or compress the local waveform slightly.
- Enforce continuity at window edges by blending with the original signal using the Hann mask.

### 8.4 Strength Interpretation
- `strength` controls Î³ and Îµ maxima; keep them â‰¤ 15% so morphology changes remain subtle.
- Enforce per-beat caps (e.g., only edit one or two beats per 10â€¯s window for strength â‰¤ 0.3).

## 9. Interface: Strength & Time Knobs
- `strength` is a normalized knob interpreted per perturbation family:
  - `smooth_adv`: L2 budget.
  - `baseline_wander` / `band_noise`: amplitude relative to per-lead std.
  - `morph_amp` / `morph_time`: bounding box for amplitude/time scaling.
- `center_time` specifies the focal point in seconds (0â€¯â‰¤â€¯center_timeâ€¯â‰¤â€¯10). `window_seconds` defaults to 2â€¯s but can be overridden. Both are applied uniformly regardless of family to keep the user API consistent.

Example defaults:
```python
DEFAULTS = {
    "smooth_adv": {"eps_global_max": 0.5, "lambda_smooth": 10.0, "lambda_energy": 0.5},
    "baseline_wander": {"alpha": 0.1, "f_range": (0.05, 0.5)},
    "band_noise": {"beta": 0.1, "band": (5.0, 40.0)},
    "morph_amp": {"gamma_max": 0.15, "sigma_ms": 80.0},
    "morph_time": {"epsilon_max": 0.1},
}
```

## 10. Module Layout & Notebook Integration
```
perturbations/
  __init__.py
  config.py            # defaults, class-to-index map
  masks.py             # time-window utilities for NumPy / TensorFlow
  metrics.py           # smoothness, L2/Lâˆž helpers
  adv_smooth.py        # TensorFlow implementation of smooth_adv
  noise.py             # baseline_wander and band_noise
  morphology.py        # morph_amp / morph_time utilities
  api.py               # PerturbationConfig + apply_perturbation dispatcher
```
Usage inside `StartingFile.ipynb`:
```python
from perturbations.api import PerturbationConfig, apply_perturbation

config = PerturbationConfig(ptype='smooth_adv', strength=0.3, center_time=4.5,
                            window_seconds=2.0, target_class='MI')
x_clean = X_test[idx]
x_adv = apply_perturbation(x_clean, fs=100, config=config, model=model, 
                           r_peaks=r_peaks_cache[idx])
pred_clean = (model.predict(x_clean[None, ...]) >= 0.5)
pred_adv = (model.predict(x_adv[None, ...]) >= 0.5)
```

## 11. Evaluation Plan
1. **Attack Success (ASR):**
   - Compare predicted label vectors (threshold 0.5) before and after perturbation.
   - Report untargeted ASR and per-class targeted success rates across `X_test`.
2. **Perturbation Budgets:**
   - L2/Lâˆž norms per record alongside strength values.
   - Smoothness metrics vs. baseline variability between beats.
3. **Plausibility:**
   - Visual overlays of clean vs. perturbed signals for random samples (per lead).
   - Optional domain-expert review of a curated set.
4. **Quantitative Metrics:**
   - ROC/AUC degradation after applying stochastic noise families.
   - Compare `smooth_adv` vs. `smooth_adv + noise` cascades.
5. **Logging:**
   - Store metadata per sample: target class, achieved class probability, norm values, and mask window.

## 12. Suggested Implementation Order
1. **Noise families:** easiest to validate; ensure the time-window mask and `PerturbationConfig` plumbing work with raw NumPy arrays.
2. **Gradient-based attack:** start without a window, validate `tf.GradientTape` loop against a single sample, then add windowing, clipping, and targeted modes.
3. **Morphology-aware warps:** build the R-peak cache, implement amplitude scaling, then the time-warp interpolation.
4. **Evaluation utilities:** thresholded label comparison, plotting helpers, and summary tables/notebook cells that call the new API against `X_test`.

With this alignment, the perturbation engine can be dropped into the existing TensorFlow notebook without altering the training code, and every section above mirrors assumptions already present in the repository.
# Design Document: Plausible ECG Perturbations for Targeted Misclassification

## 1. Context & Goals
We train and evaluate a multi-label 1D CNN in `StartingFile.ipynb` that ingests PTBâ€‘XL ECG segments stored under `ptb-xl/`. Each record is a 10â€¯s, 12â€‘lead trace sampled at 100â€¯Hz (1â€¯000 samples Ã— 12 leads). Labels are aggregated with `MultiLabelBinarizer` into the five diagnostic superclasses `['CD', 'HYP', 'MI', 'NORM', 'STTC']`, and the model is optimized with `binary_crossentropy` plus accuracy and multi-label AUC. The objective of this work is to overlay small, plausible perturbations on top of these preprocessed tensors so that the trained TensorFlow model misclassifies either untargetedly (any change in the predicted label set) or in a targeted fashion (force or suppress a specific superclass), while keeping perturbations visually consistent with realistic physiology or acquisition artefacts.

## 2. Reference Training Pipeline Snapshot

### 2.1 Dataset & Labels
- CSV metadata (`ptbxl_database.csv`) is read into `Y`, and `scp_codes` are expanded via `ast.literal_eval`.
- Raw signals are loaded with `wfdb.rdsamp` using `filename_lr` (100â€¯Hz) and stacked into `X` with shape `(num_records, 1000, 12)`.
- Diagnostic subclasses are aggregated through `scp_statements.csv`, filtered to the `diagnostic` rows, and mapped onto the five superclasses above.
- A stratified split uses `strat_fold != 10` for training and `== 10` for testing, resulting in `(X_train, y_train)` and `(X_test, y_test)`. Encoding with `MultiLabelBinarizer` produces `y_*_enc` arrays of shape `(num_records, 5)`.

### 2.2 Model
```
Sequential(
  Input(shape=(1000, 12)),
  Conv1D(32, kernel_size=5, activation='relu'),
  MaxPooling1D(pool_size=2),
  Conv1D(64, kernel_size=5, activation='relu'),
  MaxPooling1D(pool_size=2),
  Flatten(),
  Dense(64, activation='relu'),
  Dense(5, activation='sigmoid')
)
```
This architecture expects `float32` tensors shaped `(batch, 1000, 12)` and outputs independent probabilities per superclass.

### 2.3 Loss & Metrics
- Optimization: `adam`.
- Loss: `binary_crossentropy` between the sigmoid outputs and the five-dimensional label vector.
- Metrics: accuracy and `tf.keras.metrics.AUC(multi_label=True, num_labels=5)`.

The perturbation pipeline must preserve these conventions: keep tensors in the `(1000, 12)` layout, respect the five-label scheme, and call the already-compiled model for logits/probabilities.

## 3. Data & Preprocessing Assumptions
- Signals remain in the raw microvolt scale provided by WFDB; no additional normalization is currently applied. Any perturbation must therefore be generated in the same scale so gradients remain meaningful.
- Optionally, we can compute per-record means/standard deviations to report perturbation magnitudes; however, those statistics may not be used for normalization unless the training notebook is updated accordingly.
- When batching perturbations, reshape single samples to `(1, 1000, 12)` before feeding the TensorFlow model, mirroring the `model.fit` call.

## 4. Threat Model & Plausibility Constraints

### 4.1 Threat Model
- **Type I (digital) white-box:** we directly edit stored test signals (`X_test`) and have access to model parameters and gradients via TensorFlow.
- **Objectives:**
  - Untargeted: alter the predicted label set relative to the baseline set obtained with a 0.5 sigmoid threshold.
  - Targeted: force a specific superclass to be predicted (set probability â‰¥ threshold) or suppress an existing positive class (probability < threshold).
- **Budget:** perturbations must stay within an L2 norm scaled by the `strength` knob and remain temporally localized via the `center_time` / `window_seconds` interface.

### 4.2 Plausibility Constraints
- Preserve recognizable Pâ€“QRSâ€“T morphology and heart rhythm.
- Keep changes smooth in time and coherent across leads (no single-sample spikes).
- Limit edits to windows that can pass as baseline wander, powerline noise, or small morphology variations.
- Provide smooth onset/offset windows (Hann or cosine masks) and penalize high first-order derivatives of the perturbation.

## 5. Perturbation Engine Architecture

### 5.1 Core API
```python
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence
import numpy as np
import tensorflow as tf

@dataclass
class PerturbationConfig:
    ptype: str                         # 'smooth_adv', 'baseline_wander', 'band_noise', 'morph_amp', 'morph_time'
    strength: float                    # [0, 1] logical knob
    center_time: float                 # seconds from the start of the 10 s segment
    window_seconds: Optional[float] = None
    target_class: Optional[str] = None # e.g., 'MI'; None for untargeted
    extra: Optional[Dict[str, Any]] = None
```

```python
def apply_perturbation(
    x: np.ndarray,                     # shape (1000, 12)
    fs: float,                         # 100 Hz for the current notebook
    config: PerturbationConfig,
    model: Optional[tf.keras.Model] = None,
    r_peaks: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Dispatch to the configured perturbation. Returns x_adv with the same shape as x.
    """
```

### 5.2 Integration Points
- Training remains unchanged. For evaluation, we copy a clean sample, call `apply_perturbation`, reshape to `(1, 1000, 12)`, and run `model.predict` to observe label flips.
- All perturbation functions operate on NumPy arrays for compatibility with the notebook; gradient-based ones temporarily convert to `tf.Tensor` for differentiation.

### 5.3 Perturbation Families
1. **Smooth adversarial perturbations** (`smooth_adv`) â€” gradient-driven, smooth, windowed.
2. **Acquisition artefact simulations** (`baseline_wander`, `band_noise`) â€” parametric noise injections.
3. **Morphology-aware local warps** (`morph_amp`, `morph_time`) â€” interpretable edits tied to detected beats.

## 6. Perturbation Family 1: Smooth Gradient-Based Adversary (`smooth_adv`)

### 6.1 Objective
We optimize a mask-localized perturbation `Î´` that maximizes misclassification while staying smooth and energy-bounded:

\[
L(\delta) = w_{\text{cls}} L_{\text{cls}}(x + \delta) + \lambda_{\text{smooth}} L_{\text{smooth}}(\delta) + \lambda_{\text{energy}} L_{\text{energy}}(\delta)
\]

- `L_cls`: attack loss. Untargeted attacks maximize the baseline loss by setting `L_cls = -BCE(y_true, y_pred)`; targeted attacks minimize `BCE(y_target, y_pred)`, where `y_target` copies `y_true` but enforces the requested superclass bit (1 to force, 0 to suppress) in accordance with multi-label semantics.
- `L_smooth`: mean variance of the first-order differences across time and leads.
- `L_energy`: average squared magnitude.

### 6.2 Time Window Mask
Given `center_time` and `window_seconds` (default 2â€¯s), we compute sample indices at 100â€¯Hz, derive `[start, end)` bounds, and create a Hann window of length `end - start` that is embedded into a zero mask for the entire 10â€¯s record. The mask broadcasts over 12 leads to localize the perturbation smoothly.

### 6.3 Strength Mapping
- `strength` âˆˆ [0, 1].
- `eps_global_max` (default 0.5) is expressed in the normalized input space (raw WFDB units for the current notebook). `eps_max = strength * eps_global_max`.
- After each optimizer step we rescale `Î´` to satisfy `||Î´||â‚‚ â‰¤ eps_max`.
- `strength` can additionally scale `Î»_smooth` and `Î»_energy` (larger strengths relax the penalties).

### 6.4 TensorFlow Implementation Sketch
```python
def build_time_mask(L, fs, center_time, window_seconds):
    mask = np.zeros(L, dtype=np.float32)
    if window_seconds is None:
        return mask + 1.0  # full segment
    center_idx = int(center_time * fs)
    half = int(window_seconds * fs / 2)
    start = max(0, center_idx - half)
    end = min(L, center_idx + half)
    window = tf.signal.hann_window(end - start)
    mask[start:end] = window.numpy()
    return tf.convert_to_tensor(mask)

def smooth_adversarial_perturbation(x_np, fs, config, model, y_true_vec):
    x = tf.convert_to_tensor(x_np[None, ...], dtype=tf.float32)    # (1, 1000, 12)
    delta_param = tf.Variable(tf.zeros_like(x), trainable=True)
    mask = build_time_mask(x.shape[1], fs, config.center_time, config.window_seconds)
    mask = tf.reshape(mask, (1, -1, 1))                            # broadcast over leads
    opt = tf.keras.optimizers.Adam(config.extra.get("lr", 0.01))
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    eps_max = config.extra.get("eps_global_max", 0.5) * config.strength
    lambda_smooth = config.extra.get("lambda_smooth", 10.0)
    lambda_energy = config.extra.get("lambda_energy", 0.5)
    steps = config.extra.get("steps", 200)

    target_vec = y_true_vec.copy()
    if config.target_class:
        idx = CLASS_TO_INDEX[config.target_class]
        desired = config.extra.get("target_value", 1.0)
        target_vec[idx] = desired

    for _ in range(steps):
        with tf.GradientTape() as tape:
            delta = mask * delta_param
            norm = tf.norm(delta)
            delta = tf.cond(
                norm > eps_max,
                lambda: delta * (eps_max / (norm + 1e-8)),
                lambda: delta,
            )
            x_adv = x + delta
            y_pred = model(x_adv, training=False)
            cls_loss = -bce(y_true_vec, y_pred) if config.target_class is None else bce(target_vec, y_pred)
            smooth_loss = tf.reduce_mean(tf.math.squared_difference(delta[:, 1:, :], delta[:, :-1, :]))
            energy_loss = tf.reduce_mean(tf.square(delta))
            loss = cls_loss + lambda_smooth * smooth_loss + lambda_energy * energy_loss
        grads = tape.gradient(loss, [delta_param])
        opt.apply_gradients(zip(grads, [delta_param]))

    delta = mask * delta_param
    norm = tf.norm(delta)
    if eps_max > 0 and norm > eps_max:
        delta = delta * (eps_max / (norm + 1e-8))
    return (x + delta)[0].numpy(), delta[0].numpy()
```

### 6.5 Plausibility Checks
- Log per-sample L2/Lâˆž norms in both normalized and ÂµV units.
- Track `L_smooth` and energy penalties; abort or downscale when limits are exceeded.
- Overlay `x`/`x_adv` for random records and ensure thresholded predictions changed as intended.

## 7. Perturbation Family 2: Acquisition-Style Noise

### 7.1 Baseline Wander (`baseline_wander`)
- Generate per-lead low-frequency sinusoids `b_c(t) = A_c sin(2Ï€ f t + Ï†_c)` with `f âˆˆ [0.05, 0.5]` Hz and random phases.
- Amplitude `A_c = strength * Î± * std_c`, where `std_c` is the per-lead standard deviation computed after whatever preprocessing is applied to the model input (currently raw values).
- Apply the same Hann-mask window before adding to `x`.

### 7.2 Band-Limited Noise (`band_noise`)
- Create white noise per lead, filter with a Butterworth band-pass (default 5â€“40â€¯Hz), rescale to unit variance, then multiply by `strength * Î² * std_c`.
- Window with the same mask and add to `x`.

### 7.3 Combining with Adversarial Î´
- Generate `x_adv` via `smooth_adv`.
- Draw a noise config (`baseline_wander` or `band_noise`) with its own `strength`, apply it to `x_adv`, and feed the result to the model.

## 8. Perturbation Family 3: Morphology-Aware Warps

### 8.1 Beat Detection
- Use `wfdb.processing.xqrs_detect` (or a similar algorithm already bundled with WFDB, a dependency in `requirements.txt`) to precompute R-peak indices per record at 100â€¯Hz.
- Cache results next to `X_test` to avoid recomputation inside the notebook.

### 8.2 Local Amplitude Scaling (`morph_amp`)
- For beats within the selected time window, multiply samples in a Gaussian neighborhood (Ïƒ â‰ˆ 80â€¯ms by default) by a gain `1 + Î³`, where `Î³ âˆˆ [-Î³_max, Î³_max]` and `Î³_max` scales with `strength` (e.g., 0.03 at strength 0.2, up to 0.15 at strength 1.0).
- Gains can be independent per lead or tied across limb/precordial groups to maintain clinical plausibility.

### 8.3 Local Time Warping (`morph_time`)
- Select `[a, b]` windows surrounding each targeted beat (e.g., `r_k - 40` to `r_k + 60` samples).
- Define a warp factor `Îµ âˆˆ [-Îµ_max, Îµ_max]`, scale `Ï„` âˆˆ [0, 1] to `(1 + Îµ)`, clamp to [0,1], and resample the window via linear interpolation to stretch or compress the local waveform slightly.
- Enforce continuity at window edges by blending with the original signal using the Hann mask.

### 8.4 Strength Interpretation
- `strength` controls Î³ and Îµ maxima; keep them â‰¤ 15% so morphology changes remain subtle.
- Enforce per-beat caps (e.g., only edit one or two beats per 10â€¯s window for strength â‰¤ 0.3).

## 9. Interface: Strength & Time Knobs
- `strength` is a normalized knob interpreted per perturbation family:
  - `smooth_adv`: L2 budget.
  - `baseline_wander` / `band_noise`: amplitude relative to per-lead std.
  - `morph_amp` / `morph_time`: bounding box for amplitude/time scaling.
- `center_time` specifies the focal point in seconds (0â€¯â‰¤â€¯center_timeâ€¯â‰¤â€¯10). `window_seconds` defaults to 2â€¯s but can be overridden. Both are applied uniformly regardless of family to keep the user API consistent.

Example defaults:
```python
DEFAULTS = {
    "smooth_adv": {"eps_global_max": 0.5, "lambda_smooth": 10.0, "lambda_energy": 0.5},
    "baseline_wander": {"alpha": 0.1, "f_range": (0.05, 0.5)},
    "band_noise": {"beta": 0.1, "band": (5.0, 40.0)},
    "morph_amp": {"gamma_max": 0.15, "sigma_ms": 80.0},
    "morph_time": {"epsilon_max": 0.1},
}
```

## 10. Module Layout & Notebook Integration
```
perturbations/
  __init__.py
  config.py            # defaults, class-to-index map
  masks.py             # time-window utilities for NumPy / TensorFlow
  metrics.py           # smoothness, L2/Lâˆž helpers
  adv_smooth.py        # TensorFlow implementation of smooth_adv
  noise.py             # baseline_wander and band_noise
  morphology.py        # morph_amp / morph_time utilities
  api.py               # PerturbationConfig + apply_perturbation dispatcher
```
Usage inside `StartingFile.ipynb`:
```python
from perturbations.api import PerturbationConfig, apply_perturbation

config = PerturbationConfig(ptype='smooth_adv', strength=0.3, center_time=4.5,
                            window_seconds=2.0, target_class='MI')
x_clean = X_test[idx]
x_adv = apply_perturbation(x_clean, fs=100, config=config, model=model, 
                           r_peaks=r_peaks_cache[idx])
pred_clean = (model.predict(x_clean[None, ...]) >= 0.5)
pred_adv = (model.predict(x_adv[None, ...]) >= 0.5)
```

## 11. Evaluation Plan
1. **Attack Success (ASR):**
   - Compare predicted label vectors (threshold 0.5) before and after perturbation.
   - Report untargeted ASR and per-class targeted success rates across `X_test`.
2. **Perturbation Budgets:**
   - L2/Lâˆž norms per record alongside strength values.
   - Smoothness metrics vs. baseline variability between beats.
3. **Plausibility:**
   - Visual overlays of clean vs. perturbed signals for random samples (per lead).
   - Optional domain-expert review of a curated set.
4. **Quantitative Metrics:**
   - ROC/AUC degradation after applying stochastic noise families.
   - Compare `smooth_adv` vs. `smooth_adv + noise` cascades.
5. **Logging:**
   - Store metadata per sample: target class, achieved class probability, norm values, and mask window.

## 12. Suggested Implementation Order
1. **Noise families:** easiest to validate; ensure the time-window mask and `PerturbationConfig` plumbing work with raw NumPy arrays.
2. **Gradient-based attack:** start without a window, validate `tf.GradientTape` loop against a single sample, then add windowing, clipping, and targeted modes.
3. **Morphology-aware warps:** build the R-peak cache, implement amplitude scaling, then the time-warp interpolation.
4. **Evaluation utilities:** thresholded label comparison, plotting helpers, and summary tables/notebook cells that call the new API against `X_test`.

With this alignment, the perturbation engine can be dropped into the existing TensorFlow notebook without altering the training code, and every section above mirrors assumptions already present in the repository.

- `extra`: dictionary of family-specific overrides (e.g., frequency ranges, epsilon budgets, RNG seeds).

## 3. Dispatcher Behavior (`apply_perturbation`)

1. **Normalize inputs**:
   - Clamp `strength` to `[0,1]`.
   - Coerce `center_time` and `window_seconds` to floats.
2. **Route by `ptype`**:
   - `baseline_wander` â†’ `noise.baseline_wander`.
   - `band_noise` â†’ `noise.band_limited_noise`.
   - `smooth_adv` â†’ `adv_smooth.smooth_adversarial_perturbation`.
   - `morph_amp` â†’ `morphology.local_amplitude_scaling`.
   - `morph_time` â†’ `morphology.local_time_warp`.
3. **Validate requirements**:
   - `smooth_adv` requires both `model` and `y_true`.
   - `morph_*` requires `r_peaks`.
4. **Forward optional kwargs**:
   - `rng`: optional `np.random.Generator` allows deterministic behavior across runs.
   - `model`, `y_true`, `r_peaks` are forwarded only when relevant to avoid unnecessary dependencies.
5. **Return**: perturbed signal with the same shape as the input `(samples, leads)` (default `(1000, 12)`).

## 4. Usage Patterns

### 4.1 Direct Calls

```python
config = PerturbationConfig(ptype='band_noise', strength=0.2, center_time=5.0)
x_noisy = apply_perturbation(x_clean, fs=100, config=config)
```

### 4.2 Within Evaluation Utilities

`run_attack_on_sample` (see `docs/evaluation_utilities.md`) calls `apply_perturbation` internally, passing along `model`, `y_true`, and optionally `r_peaks`. This standardizes logging, norm calculations, and prediction comparisons.

### 4.3 Notebook Demos

`StartingFile.ipynb` demonstrates:

- **Noise evaluation**: building `noise_config` and calling `apply_perturbation` inside a batch to evaluate model accuracy on noisy signals.
- **Smooth Adversarial Demo**: constructing a `smooth_adv` config and running `run_attack_on_sample`.
- **Morphology Perturbations**: detecting R-peaks, building `morph_amp` / `morph_time` configs, and calling `apply_perturbation` with `r_peaks`.

## 5. Error Handling

- Unknown `ptype` raises `ValueError`.
- Missing prerequisites (e.g., `model` for `smooth_adv`, `r_peaks` for `morph_*`) raise descriptive `ValueError`s to guide the user.

## 6. Extensibility

To add a new perturbation family:

1. Implement it in a dedicated module (e.g., `perturbations/new_family.py`).
2. Update `apply_perturbation` with a new `ptype` branch and required parameters.
3. Document the new options in `docs/configuration.md` and provide usage examples.

With these hooks, the API remains consistent regardless of how many perturbation families are introduced.

