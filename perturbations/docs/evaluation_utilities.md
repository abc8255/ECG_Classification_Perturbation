# Evaluation Utilities

This document details the evaluation helpers implemented in `perturbations/evaluation.py`. It explains the `AttackResult` schema, prediction helpers, attack runner, metrics, and DataFrame conversion utilities used to analyze perturbation experiments.

## 1. AttackResult Dataclass

### Purpose

Represents a single perturbation attempt on one ECG segment, storing metadata, predictions, norms, and success flags. Standardizing this structure enables consistent logging, aggregation, and plotting.

### Definition (`evaluation.py:17-44`)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class AttackResult:
    record_id: Optional[Any]
    ptype: str
    strength: float
    center_time: float
    window_seconds: Optional[float]
    target_class: Optional[str]
    target_mode: Optional[str]
    config_extra: Dict[str, Any]
    y_true: np.ndarray
    y_hat_clean: np.ndarray
    y_hat_adv: np.ndarray
    y_proba_clean: np.ndarray
    y_proba_adv: np.ndarray
    delta_norm_l2: float
    delta_norm_linf: float
    delta_smoothness: Optional[float]
    untargeted_success: bool
    targeted_success: Optional[bool]
```

- `config_extra` captures the original `extra` dict for reproducibility.
- `delta_smoothness` can be `None` if not computed (e.g., set `compute_delta_smoothness=False`).
- `targeted_success` only applies when `target_class` is defined.

## 2. Prediction Helpers

- `predict_proba_single(model, x)` → `(5,)`: runs `model.predict` on a single `(samples, leads)` tensor.
- `predict_proba_batch(model, X)` → `(N, 5)`: batch version returning probabilities for multiple samples.
- `binarize_predictions(proba, threshold=0.5)`: thresholds probabilities to 0/1 labels.

These functions encapsulate TensorFlow calls and thresholding logic, ensuring all evaluation code uses consistent reshaping and verbosity settings.

## 3. Smoothness Metric

`compute_smoothness(delta)` computes mean squared first-order differences across time and leads. It’s useful for verifying that perturbations remain smooth (particularly important for `smooth_adv` and noise families).

## 4. Running an Attack (`run_attack_on_sample`)

Signature (`evaluation.py:83-111`):

```python
def run_attack_on_sample(
    model,
    x_clean,
    y_true,
    config,
    *,
    fs,
    rng=None,
    threshold=0.5,
    compute_delta_smoothness=True,
    r_peaks=None,
) -> Tuple[AttackResult, np.ndarray]:
```

Steps:

1. Cast `x_clean`/`y_true` to NumPy arrays.
2. Call `apply_perturbation` with `model`, `y_true`, optional `r_peaks`, and `rng`.
3. Obtain clean and adversarial probabilities/labels via prediction helpers.
4. Compute `delta = x_adv - x_clean`, L2/L∞ norms, and optional smoothness.
5. Determine untargeted and targeted success flags (if applicable).
6. Return the `AttackResult` plus `x_adv`.

### Usage Example

```python
result, x_adv = run_attack_on_sample(
    model,
    X_test[idx],
    y_test_enc[idx],
    config,
    fs=sampling_rate,
    rng=np.random.default_rng(123),
    r_peaks=r_peaks_cache[idx],   # required for morph_* configs
)
```

## 5. Metrics

### 5.1 `compute_untargeted_asr(results)`

Returns the fraction of `AttackResult`s where `y_hat_clean != y_hat_adv`. Works for any perturbation family.

### 5.2 `compute_targeted_success(results, class_name, mode)`

- Filters results for `target_class == class_name` and `target_mode == mode`.
- `mode='force'`: success if target bit is 1 in `y_hat_adv`.
- `mode='suppress'`: success if bit transitions from 1 → 0.
- Returns `(success_rate, sample_count)`.

### 5.3 `compute_global_metrics(y_true, y_proba)`

Computes per-class ROC AUC plus precision/recall/F1 at a fixed threshold. Useful for global performance comparisons (clean vs noisy vs adversarial).

## 6. Aggregation Helpers

### 6.1 `results_to_dataframe(results, include_probabilities=False, include_vectors=False)`

Converts a list of `AttackResult`s into a `pandas.DataFrame` with columns like `ptype`, `strength`, `delta_norm_l2`, etc. Optional flags include raw probability or label vectors for downstream analysis.

### 6.2 `summarize_norms(results)`

Groups by `(ptype, strength)` and reports mean/std for L2 and L∞ norms. Example usage in the notebook:

```python
norm_summary = summarize_norms(noise_results)
display(norm_summary)
```

### 6.3 `summarize_smoothness(results)`

Similar to `summarize_norms` but for the smoothness metric. Useful when comparing smooth vs non-smooth perturbations.

## 7. Integration with Visualization

The DataFrame output from `results_to_dataframe` feeds into visualization utilities:

- `plot_asr_vs_strength` expects a list of `AttackResult`s.
- `plot_norm_distributions` and `plot_classwise_metric_bars` rely on aggregated metrics computed from results/labeled predictions.

## 8. Usage Workflow

1. **Collect results**:
   ```python
   results = []
   for idx in sample_indices:
       res, _ = run_attack_on_sample(model, X_test[idx], y_test_enc[idx], config, fs=100, r_peaks=r_peaks_cache[idx])
       results.append(res)
   ```
2. **Compute metrics**:
   ```python
   asr = compute_untargeted_asr(results)
   success_force, n_force = compute_targeted_success(results, 'MI', 'force')
   ```
3. **Summaries/plots**:
   ```python
   norm_summary = summarize_norms(results)
   fig, ax = plot_asr_vs_strength(results)
   ```

By adhering to the `AttackResult` schema and these helpers, all evaluation experiments remain reproducible and easily comparable.

