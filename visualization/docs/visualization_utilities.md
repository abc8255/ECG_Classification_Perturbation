# Visualization Utilities

This document explains the plotting helpers contained in `visualization/visualization.py`. These utilities transform raw perturbation data into interpretable figures for both per-sample inspection and aggregate analysis.

## 1. Design Goals

- Provide consistent plotting primitives for ECG perturbation analysis.
- Support both qualitative visual inspection (clean vs perturbed waveforms) and quantitative summaries (ASR curves, norm distributions, class-wise metrics).
- Accept either raw signals (`x_clean`, `x_adv`) or aggregated `AttackResult`s / metric dictionaries.

## 2. Per-Sample Plots

### 2.1 `plot_triptych`

**Purpose**: Show clean, perturbed, and difference signals for a single lead.

**Signature**:
```python
def plot_triptych(x_clean, x_adv, *, fs, lead_idx=1, title=None, zoom=None, share_y=True)
```

**Behavior**:

- Computes time axis from `fs`.
- Extracts the specified `lead_idx` (0-based).
- Plots three stacked panels:
  1. Clean signal.
  2. Perturbed signal (with optional shared y-limits).
  3. Difference (`x_adv - x_clean`) with zero reference line and max-Î” annotation.
- Optional `zoom=(t_start, t_end)` restricts the x-axis to a specific time window.

**Use Cases**:

- Notebook demos to compare clean vs noisy/adversarial signals.
- Publication figures showing how attacks manipulate ECG morphology.

### 2.2 `plot_overlay_with_diff`

**Purpose**: More compact visualization with clean/perturbed overlays plus difference panel.

**Signature**:
```python
def plot_overlay_with_diff(x_clean, x_adv, *, fs, lead_idx=1, zoom=None, title=None)
```

**Behavior**:

- Top panel overlays clean and perturbed signals with a legend.
- Bottom panel shows the difference signal.
- Shares configuration options similar to `plot_triptych`.

## 3. Aggregate Plots

These functions consume `AttackResult` collections or metrics dictionaries to summarize large experiments.

### 3.1 `plot_asr_vs_strength`

**Input**: `results` (list of `AttackResult`).

**Behavior**:

- Converts results to DataFrame (`results_to_dataframe`).
- Groups by `(ptype, strength)` and computes mean `untargeted_success`.
- Plots strength (x-axis) vs ASR (y-axis) for each `ptype`.
- Returns `(fig, ax)` for further customization.

**Usage**:

```python
fig, ax = plot_asr_vs_strength(results, title='ASR vs Strength')
```

### 3.2 `plot_norm_distributions`

**Purpose**: Visualize L2/Lâˆž norm distributions across perturbation types/strengths via boxplots.

**Signature**:
```python
def plot_norm_distributions(results, value='delta_norm_l2', ax=None, title=None)
```

**Behavior**:

- Requires `value` column to exist (e.g., `'delta_norm_l2'`, `'delta_norm_linf'`, `'delta_smoothness'`).
+- Builds labels like `"smooth_adv (s=0.3)"`.
+- Generates boxplots with mean markers for each label.

### 3.3 `plot_classwise_metric_bars`

**Purpose**: Compare per-class metrics (AUC, F1, etc.) across conditions (clean vs various perturbations).

**Signature**:
```python
def plot_classwise_metric_bars(metrics, *, metric_name='AUC', ax=None, title=None)
```

**Behavior**:

- `metrics`: dict mapping condition names (e.g., `"clean"`, `"smooth_adv"`) to per-class metric dicts (usually produced by `compute_global_metrics`).
- Creates grouped bars per class with conditions as offsets.
- Y-axis limited to `[0,1]`.

### 3.4 `plot_asr_vs_time`

**Purpose**: Attack success rate vs. perturbation window center time (Section 2.1 aggregates).

**Signature**:
```python
def plot_asr_vs_time(df_windows, *, bin_width=0.5, ax=None, title=None, show=True, save_path=None)
```

**Behavior**:

- Expects a per-window summary DataFrame with `center_time` and `minimal_strength`/`strength_star_window`.
- Bins center times (default 0.5 s) and computes ASR per bin.
- Returns `(fig, ax)`; can optionally save to disk when `save_path` is provided.

### 3.5 `plot_asr_time_class_heatmap`

**Purpose**: Heatmap of ASR across time bins and diagnostic classes.

**Signature**:
```python
def plot_asr_time_class_heatmap(df_windows, *, class_names=CLASS_NAMES, bin_width=0.5, ...)
```

**Behavior**:

- Requires `df_windows` to contain `y_true_bits` (per-window class membership).
- Builds a matrix of ASR per `(class, time bin)` and renders it via `imshow`.
- Useful for highlighting when each class is most vulnerable.

### 3.6 `plot_strength_histogram`

**Purpose**: Distribution of minimal strengths across all samples (Section 2.2 aggregates).

**Signature**:
```python
def plot_strength_histogram(df_samples, *, max_strength=0.5, bin_width=0.025, ...)
```

**Behavior**:

- Takes the sample-level table produced by `summarize_minimal_strength_per_sample`.
- Drops NaNs (robust samples) and renders a histogram.

### 3.7 `plot_strength_boxplot_by_class`

**Purpose**: Compare minimal strengths across classes.

**Signature**:
```python
def plot_strength_boxplot_by_class(df_samples, *, class_names=CLASS_NAMES, ...)
```

**Behavior**:

- Uses `y_true_bits` embedded in `df_samples` to split by class membership.
- Produces boxplots (with means) for every class with at least one successful attack.

### 3.8 `plot_robust_fraction_by_class`

**Purpose**: Fraction of samples that remained robust (no success up to max strength) per class.

**Signature**:
```python
def plot_robust_fraction_by_class(df_samples, *, class_names=CLASS_NAMES, ...)
```

**Behavior**:

- Marks robustness as `strength_star_sample` being NaN.
- Builds a bar chart over the provided class list.

## 4. Internal Helpers

- `_ensure_axes`: utility to handle existing axes or create new figures with specified dimensions.
- `results_to_dataframe`: imported from `evaluation.py`, ensuring consistent data format across plots.

## 5. Notebook Usage

Examples in `StartingFile.ipynb`:

- `plot_triptych` used after noise, smooth, and morphology perturbations for visual inspection.
- Aggregate plots can be invoked once you have a list of `AttackResult`s (e.g., `plot_asr_vs_strength(noise_results)`).

## 6. Customization Tips

- All plotting functions return `(fig, ax)`; use `ax.set_title`, `ax.set_ylim`, `ax.axvline`, etc., for further adjustment.
- For publication-quality figures, adjust `figsize`, `dpi`, and fonts before saving via `fig.savefig`.
- When overlaying multiple leads, call `plot_triptych`/`plot_overlay_with_diff` in a loop over lead indices.

## 7. Extensibility

To add new visualizations:

1. Define a plotting function accepting either raw signals or aggregated data.
2. Reuse `_ensure_axes` for consistent figure creation.
3. Document the new plot in this file and, if applicable, integrate it into the notebook.

By centralizing plotting logic in `visualization/visualization.py`, we keep visuals consistent and reusable across experiments and notebooks.

