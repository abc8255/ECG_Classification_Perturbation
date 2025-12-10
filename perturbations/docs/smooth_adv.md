# Smooth Gradient-Based Adversarial Perturbations

This document provides an in-depth explanation of the smooth adversarial perturbations implemented in `perturbations/adv_smooth.py`. It covers the theoretical motivation, TensorFlow-based implementation, configuration interface, and usage patterns for ECG-specific attacks.

## 1. Motivation

Classical adversarial attacks often craft L∞-bounded noise that is imperceptible but visually “spiky”. For ECG signals, clinical plausibility demands perturbations that:

- Preserve P–QRS–T morphology.
- Avoid abrupt sample-level spikes.
- Remain localized in time to mimic subtle physiological changes.

Smooth adversarial perturbations enforce these constraints while leveraging gradients from the trained CNN to maximize misclassification, either untargeted (any label change) or targeted (force/suppress specific classes).

## 2. High-Level Formulation

Given an input ECG segment `x ∈ ℝ^{L×C}` (L samples, C leads), we optimize `δ` to minimize:

\[
L(δ) = L_{\text{cls}}(x + δ) + λ_{\text{smooth}} L_{\text{smooth}}(δ) + λ_{\text{energy}} L_{\text{energy}}(δ)
\]

subject to an L2 norm constraint `‖δ‖₂ ≤ strength × eps_global_max` and a temporal mask `m` localizing the perturbation around `center_time`.

### Terms

- **Classification loss (`L_cls`)**:
  - Untargeted: `-BCE(y_true, model(x + δ))` to push probabilities away from the true labels.
  - Targeted: `BCE(y_target, model(x + δ))` where `y_target` modifies selected bits (force or suppress).
- **Smoothness loss (`L_smooth`)**: mean squared first-order difference across time/leads; penalizes rapid changes.
- **Energy loss (`L_energy`)**: mean squared magnitude (L2 energy).

## 3. Implementation Details (`perturbations/adv_smooth.py:18-149`)

### 3.1 Defaults (`SmoothAdvDefaults`)

```python
SMOOTH_ADV_DEFAULTS = SmoothAdvDefaults(
    eps_global_max=0.5,
    lambda_smooth=10.0,
    lambda_energy=0.5,
    steps=200,
    learning_rate=0.01,
)
```

These values can be overridden via `config.extra`.

### 3.2 Workflow (`smooth_adversarial_perturbation`)

1. **Input prep**: ensure `(samples, leads)` layout and convert to `tf.Tensor`.
2. **Mask**: call `_build_time_mask_tf` to obtain a `(1, L, 1)` Hann mask centered at `config.center_time` with width `config.window_seconds`. If `window_seconds` is `None`, the mask is all ones.
3. **Targets**: `_prepare_targets` builds `y_true` and `y_target` tensors (the latter only used when `target_class` is specified). Supports `"force"` or `"suppress"` modes via `config.extra['target_mode']`.
4. **Optimization loop**:
   - Variable `delta_param` stores unconstrained perturbations; actual `delta = delta_param * mask`.
   - Norm clipping ensures L2 ≤ `eps_max`.
   - Forward pass: `model(x + delta)` evaluated without training mode.
   - Loss: classification + smoothness + energy.
   - Backprop: gradients wrt `delta_param` updated via Adam optimizer.
5. **Post-processing**: final `delta` is clipped to `eps_max` again (if necessary) and added to `x`.

### 3.3 Output

The function returns `(x_adv, delta)` as NumPy arrays. It is directly used in `apply_perturbation` when `ptype='smooth_adv'`.

## 4. Configuration & Usage

### 4.1 PerturbationConfig Example

```python
config = PerturbationConfig(
    ptype='smooth_adv',
    strength=0.3,
    center_time=4.5,
    window_seconds=2.0,
    target_class='MI',            # or None for untargeted
    extra={
        'eps_global_max': 0.4,
        'steps': 150,
        'target_mode': 'force',   # or 'suppress'
        'target_value': 1.0,
    },
)
x_adv = apply_perturbation(
    x_clean,
    fs=100,
    config=config,
    model=model,
    y_true=y_true_vec,
)
```

**Important**: `model` (trained TensorFlow model) and `y_true` (ground-truth label vector) must be provided. Without them, `apply_perturbation` raises an error for `smooth_adv`.

### 4.2 Notebook Demo

`StartingFile.ipynb` includes a “Smooth Adversarial Attack Demo” section:

```python
adv_config = PerturbationConfig(
    ptype='smooth_adv',
    strength=0.25,
    center_time=4.5,
    window_seconds=2.0,
    target_class=None,
    extra={'eps_global_max': 0.3, 'steps': 100},
)
adv_result, x_adv_demo = run_attack_on_sample(
    model,
    X_test[idx],
    y_test_enc[idx],
    adv_config,
    fs=sampling_rate,
)
```

It prints untargeted success flags, predictions, and norms, and visualizes the perturbation via `plot_triptych`.

### 4.3 Targeted Attacks

Set `target_class` (e.g., `'MI'`) and `extra['target_mode']` (`"force"` or `"suppress"`). `run_attack_on_sample` logs `targeted_success`, enabling metrics such as `compute_targeted_success(results, 'MI', mode='force')`.

### 4.4 Parameter Tuning Tips

- Reduce `steps` or `eps_global_max` for faster but weaker attacks.
- Adjust `lambda_smooth`/`lambda_energy` to trade off smoothness vs. attack success. Larger values enforce smoother perturbations.
- Use `window_seconds` to focus on segments containing relevant morphology (e.g., ST segments when targeting STTC).
- For targeted *suppression*, set `extra={'target_mode': 'suppress'}` so that the targeted bit is driven below 0.5.

## 5. ECG-Specific Considerations

- **Clinical plausibility**: keep `strength` ≤ 0.3 and window durations ≤ 2–3 s to avoid obvious artefacts.
- **Lead consistency**: the mask broadcasts across all leads, so perturbations affect every lead within the window. This approximates subtle physiological changes rather than single-lead glitches.
- **Interpretability**: overlay clean vs adversarial signals with `plot_triptych` or `plot_overlay_with_diff` to ensure morphological features remain plausible.

## 6. Evaluation & Metrics

- Use `run_attack_on_sample` to collect `AttackResult`s with norms, smoothness, and success flags.
- `compute_untargeted_asr` quantifies how often the attack alters predicted label sets.
- `compute_targeted_success` (per class/mode) measures targeted efficacy.
- `summarize_norms`/`summarize_smoothness` produce tables for reporting budgets.
- `plot_asr_vs_strength` can show how success rates scale with the `strength` knob.

## 7. Combining with Other Perturbations

Smooth adversarial perturbations can be composed with acquisition noise or morphology warps:

1. Generate `x_adv` via `smooth_adv`.
2. Pass the result through `baseline_wander`, `band_noise`, or `morph_*` with separate configs.
3. Evaluate predictions on the composite perturbation to simulate adversaries hiding within realistic artefacts.

## 8. Summary

Smooth adversarial perturbations extend classic adversarial methods to ECG signals by enforcing temporal localization, smoothness, and energy constraints. Their TensorFlow-based implementation integrates seamlessly with the shared API, enabling both untargeted and targeted attacks that remain clinically plausible while probing model robustness.

