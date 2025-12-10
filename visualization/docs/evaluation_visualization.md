# Evaluation & Visualization Utilities for ECG Perturbations

## 1. Scope & Goals

This document specifies the evaluation and visualization utilities that complement the perturbation engine introduced in `perturbations/`. The immediate focus is on the **noise families implemented today** (`baseline_wander`, `band_noise`). The same scaffolding is future-proofed so that forthcoming perturbations (e.g., `smooth_adv`, `morph_amp`, `morph_time`) can plug in without rewrites.

Goals:

1. Quantitatively measure how perturbations impact the CNN defined in `StartingFile.ipynb`.
2. Record per-sample outcomes (predictions, norms, metadata) for reproducible analysis.
3. Provide plotting utilities for per-sample inspection and aggregate figures, including a triptych (clean vs perturbed vs difference) suitable for publications.

All utilities assume the conventions already in use:

- Each ECG sample is shaped `(1000, 12)` (10 s at 100 Hz, 12 leads).
- Labels are 5-way multi-label vectors for `['CD', 'HYP', 'MI', 'NORM', 'STTC']`.
- Predictions are sigmoid outputs thresholded at 0.5.

## 2. Core Evaluation Utilities

### 2.1 Attack Result Schema

A standard record per perturbation attempt ensures downstream tools remain consistent. Suggested fields:

| Field | Description |
| --- | --- |
| `record_id` | Index or unique identifier for the ECG sample. |
| `ptype`, `strength`, `center_time`, `window_seconds`, `extra` | Copied from `PerturbationConfig`. |
| `target_class`, `target_mode` | `None` for the current noise families; future targeted attacks can fill these. |
| `y_true` | Ground-truth 5-bit vector. |
| `y_hat_clean`, `y_hat_adv` | Binary predictions before/after perturbation. |
| `y_proba_clean`, `y_proba_adv` | Raw sigmoid probabilities (length 5). |
| `delta_norm_l2`, `delta_norm_linf` | Norms of `δ = x_adv - x_clean`. |
| `delta_smoothness` | Optional variance of first-order differences (even noise perturbations should remain smooth). |
| `untargeted_success`, `targeted_success` | Booleans based on prediction changes. |

Implementation options:

- `dataclasses.dataclass` (typed record).
- Pandas DataFrame rows (easy grouping/aggregation).

### 2.2 Prediction Helpers

Create thin wrappers around the TensorFlow model to avoid repetitive reshaping logic.

```python
def predict_proba_single(model, x):
    return model.predict(x[None, ...], verbose=0)[0]

def predict_proba_batch(model, X):
    return model.predict(X, verbose=0)

def binarize_predictions(proba, threshold=0.5):
    proba = np.asarray(proba)
    return (proba >= threshold).astype(int)
```

These functions should be defined once (e.g., in a notebook cell or `evaluation.py`) and reused everywhere.

### 2.3 Sample-Level Attack Runner

A helper to tie everything together:

```python
def run_attack_on_sample(model, x_clean, y_true, config, fs, rng=None):
    x_adv = apply_perturbation(x_clean, fs=fs, config=config, rng=rng)
    proba_clean = predict_proba_single(model, x_clean)
    proba_adv = predict_proba_single(model, x_adv)
    y_hat_clean = binarize_predictions(proba_clean)
    y_hat_adv = binarize_predictions(proba_adv)

    delta = x_adv - x_clean
    result = {
        "record_id": config.extra.get("record_id"),
        "ptype": config.ptype,
        "strength": config.strength,
        "center_time": config.center_time,
        "window_seconds": config.window_seconds,
        "target_class": config.target_class,
        "target_mode": config.extra.get("target_mode"),
        "y_true": y_true,
        "y_hat_clean": y_hat_clean,
        "y_hat_adv": y_hat_adv,
        "y_proba_clean": proba_clean,
        "y_proba_adv": proba_adv,
        "delta_norm_l2": np.linalg.norm(delta),
        "delta_norm_linf": np.max(np.abs(delta)),
        "untargeted_success": bool(np.any(y_hat_clean != y_hat_adv)),
    }
    return result, x_adv
```

Future perturbations can add smoothness metrics, targeted-success logic, etc., but the schema remains stable.

## 3. Quantitative Metrics

Assuming we have a list or DataFrame of attack results:

### 3.1 Untargeted Attack Success

```python
def compute_untargeted_asr(results):
    successes = [r["untargeted_success"] for r in results]
    return np.mean(successes) if successes else 0.0
```

Optional: compute per-class bit-flip rates by checking differences per label index.

### 3.2 Targeted Success

The current repo has no targeted perturbations yet, but the metric interface should be in place:

```python
def compute_targeted_success(results, class_name, mode="force"):
    filtered = [r for r in results if r["target_class"] == class_name and r.get("target_mode") == mode]
    if not filtered:
        return 0.0, 0
    idx = CLASS_TO_INDEX[class_name]
    if mode == "force":
        success = [r["y_hat_adv"][idx] == 1 for r in filtered]
    else:  # suppress
        success = [(r["y_hat_clean"][idx] == 1) and (r["y_hat_adv"][idx] == 0) for r in filtered]
    return float(np.mean(success)), len(filtered)
```

When targeted attacks are added later, this code path is already defined.

### 3.3 Norm & Smoothness Summaries

Group results by `ptype` and `strength` (or any other key) and summarize:

```python
def summarize_norms(results):
    df = pd.DataFrame(results)
    return df.groupby(["ptype", "strength"])["delta_norm_l2", "delta_norm_linf"].agg(["mean", "std"])
```

Smoothness metrics can be included once those values are stored in the result schema.

### 3.4 Global Performance Metrics

To measure macro-level degradation:

1. Generate perturbed batches (e.g., entire `X_test` with a fixed configuration).
2. Run `model.evaluate` as already illustrated in the notebook.
3. Optionally compute per-class AUC/precision/recall using `sklearn.metrics`.

These metrics can be reused for clean vs noise vs future adversarial conditions.

## 4. Logging & Storage

- Maintain a Pandas DataFrame with one row per attack attempt.
- Save to disk (`.csv` or `.parquet`) along with metadata describing the experiment (date, commit hash, model checkpoint).
- Optionally persist `x_adv` arrays (NumPy `.npy`) for a curated subset of samples to reuse in visualization without regenerating perturbations.

## 5. Visualization Utilities

### 5.1 Triptych (Clean / Perturbed / Difference)

Function signature:

```python
def plot_triptych(x_clean, x_adv, fs, lead_idx=1, title=None, zoom=None, share_y=True):
    ...
```

Implementation guidelines:

- Convert sample indices to time via `np.arange(num_samples) / fs`.
- Panel A: plot clean signal for selected lead (default: lead II, index 1).
- Panel B: plot perturbed signal, optionally sharing y-limits with Panel A.
- Panel C: plot difference (`x_adv - x_clean`), include a zero line and annotate max absolute deviation.
- `zoom=(t_start, t_end)` restricts the x-range for all panels.
- Annotate figure with `ptype`, `strength`, and clean vs perturbed predictions for context.

### 5.2 Overlay Variant

Compact layout with two panels:

1. Clean and perturbed overlaid in one axis.
2. Difference in a second axis.

Useful when differences are subtle and the reader needs direct comparison.

### 5.3 ASR vs Strength Curves

For each perturbation family, compute untargeted ASR across a range of strengths and plot:

- x-axis: strength.
- y-axis: ASR (0–1).
- Separate lines per `ptype` (currently baseline vs band noise).

As new perturbations arrive, simply add them to the same plotting function.

### 5.4 Norm & Smoothness Distributions

Use boxplots or violin plots for `delta_norm_l2`, `delta_norm_linf`, and `delta_smoothness` grouped by `ptype`. Confirms perturbations stay within the intended budgets.

### 5.5 Class-wise Metric Charts

For each diagnostic class, plot bars (or heatmaps) showing metrics such as AUC or F1 for clean vs perturbed conditions. This highlights which classes degrade most under noise.

## 6. Canonical Figure Set

For reports/theses, a typical figure lineup might include:

1. **Triptych Example:** clean vs baseline-wander perturbed sample, showing minimal visual change but altered predictions.
2. **ASR Curve:** untargeted ASR vs strength for baseline wander and band noise.
3. **Norm Distribution:** boxplot of L2 norms across perturbation types at fixed strength.
4. **Class-wise Performance:** bar chart of per-class AUC drop (clean vs noise).

These figures can be regenerated anytime once the logging and plotting utilities are in place.

## 7. Implementation Notes

- Keep plotting code modular—e.g., a `visualization.py` or dedicated notebook cells—so reproducing figures is straightforward.
- Use consistent seeds when sampling which records to visualize (e.g., `np.random.default_rng(42)`).
- Label axes clearly: time (s) on the x-axis, amplitude (µV-equivalent) on the y-axis.
- When sharing or publishing figures, record the perturbation configuration and model checkpoint in the caption or metadata for traceability.

With these utilities, we have a coherent path from perturbation generation to per-sample inspection, aggregate metrics, and publication-ready visuals—all aligned with the current TensorFlow notebook and the noise perturbations already implemented.

