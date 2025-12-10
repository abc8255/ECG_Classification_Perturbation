# Main Experiments: Mapping the Vulnerability Landscape

Sections 2.1 and 2.2 describe the experiments that probe *where* the PTB-XL CNN is vulnerable to smooth, localized perturbations and *how much* perturbation strength is required to flip its predictions. The procedures below are anchored to the implementations already present in this repository:

- Model training / evaluation pipeline in `StartingFile.ipynb`.
- Perturbation primitives exposed via `perturbations.api.PerturbationConfig` and `apply_perturbation`.
- Smooth adversarial engine in `perturbations.adv_smooth`.
- Attack logging helpers in `perturbations.evaluation`.
- Plotting utilities in `visualization/visualization.py` and documentation in `visualization/docs/`.

We work with ECG tensors shaped `(1000, 12)` (10 s, 100 Hz) and 5-way multi-label targets (`['CD', 'HYP', 'MI', 'NORM', 'STTC']`), matching `StartingFile.ipynb`.

---

## 0. Evaluation Subset & Record IDs

Running the full pipeline on every `X_test` sample is feasible but slow. To make experiments reproducible we standardize how evaluation indices are selected and how those indices map onto `record_id` throughout the logging helpers.

### 0.1 Default subset

- `EVAL_MAX_SAMPLES = 1000`.
- Sampling is multi-label-aware: for each class index `k`, collect `idx_k = np.where(y_test_enc[:, k] == 1)[0]`, sample `n_k = min(EVAL_MAX_SAMPLES // 5, len(idx_k))` indices without replacement, and take the union of all sampled sets.
- The union is sorted to preserve ascending dataset indices. The final evaluation tensors are `X_eval = X_test[eval_indices]` and `Y_eval = y_test_enc[eval_indices]`.

### 0.2 Helper function (to add in `perturbations.evaluation`)

```python
def select_eval_subset(y_test_enc, max_samples=1000, random_state=42):
    """
    Returns sorted dataset indices for the evaluation subset.
    """
```

The helper seeds `np.random.default_rng(random_state)` and ensures every returned index corresponds to the same row in `X_test`. **`record_id` is defined to be this dataset index**; every `PerturbationConfig.extra` must store it, and every output DataFrame must preserve it so downstream joins can recover `y_true` or RNG seeds unambiguously.

> Quick start: run `python examples/train_ptb_cnn.py` to train the baseline CNN (this saves `models/ptb_cnn.keras` and exports `data/ptb_eval.npz` automatically), or run `python examples/prepare_ptb_eval_data.py` if you already have a trained model and only need the evaluation subset. The experiment driver `examples/run_vulnerability_experiment.py` defaults to these artifact paths.

---

## 2.1 Where in the ECG is the Model Most Vulnerable?

### 2.1.1 High-Level Goal

For each evaluation sample (we default to the `X_test` array from `StartingFile.ipynb`), identify the time regions where a smooth, windowed perturbation is most likely to cause an untargeted misclassification. The pipeline:

1. Compute a per-time-step saliency curve that reflects how sensitive the current BCE loss is to perturbations.
2. Extract the top `K = 3` salient windows of fixed duration (`window_seconds = 2.0`, i.e., 200 samples).
3. Run the `smooth_adv` untargeted attack on each window with a small strength schedule `[0.10, 0.20, 0.35, 0.50]`.
4. Log, for every `(sample, window)` pair, whether any strength succeeded and the minimal strength that did.
5. Aggregate results to analyze attack success vs. time and per-class vulnerability patterns.

All steps run with `fs = 100.0` Hz, which matches our preprocessing and the `build_time_mask` helper used by `smooth_adv`.

### 2.1.2 Saliency Definition

We use gradient-based saliency implemented directly in TensorFlow so it can plug into the trained CNN defined in `StartingFile.ipynb`.

1. Inputs:
   - `x_clean` → `(1000, 12)` NumPy array.
   - `y_true_vec` → `(5,)` float vector (0/1).
   - `model` → compiled TensorFlow model already loaded in the notebook.
2. Tensor conversions:
   ```python
   tensor_x = tf.convert_to_tensor(x_clean[None, ...])  # (1, 1000, 12)
   tensor_y = tf.convert_to_tensor(y_true_vec.reshape(1, -1))
   ```
3. Loss:
   ```python
   bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
   with tf.GradientTape() as tape:
       tape.watch(tensor_x)
       logits = model(tensor_x, training=False)
       loss = bce(tensor_y, logits)
   grads = tape.gradient(loss, tensor_x)[0].numpy()  # (1000, 12)
   ```
4. Reduce over leads using L2 (fixed choice to avoid ambiguity):
   ```python
   saliency = np.linalg.norm(grads, ord=2, axis=-1)  # shape (1000,)
   ```
5. Smooth with a centered moving average of width 11 samples (~110 ms) to suppress jitter (requires `scipy.ndimage.uniform_filter1d`, so SciPy must be listed in `requirements.txt`):
   ```python
   saliency_smooth = uniform_filter1d(saliency, size=11, mode="nearest")
   ```
6. Normalize per sample so the peak equals 1.0:
   ```python
   saliency_norm = saliency_smooth / (saliency_smooth.max() + 1e-6)
   ```

The resulting `saliency_norm` is cached (e.g., to disk or alongside metadata in memory) so window selection and later analysis can reuse it without recomputing gradients.

### 2.1.3 Selecting Top-K Salient Windows

Parameters:

- Sampling rate `fs = 100` Hz.
- Window duration `W_sec = 2.0`, so `W = 200` samples.
- Candidate centers evaluated every `center_stride = 5` samples (50 ms) to reduce computation while keeping good coverage.
- Overlap guard: at most 25% overlap between selected windows.

Procedure per sample:

1. Candidate centers:
   ```python
   half = W // 2
   centers_full = np.arange(half, 1000 - half + 1)              # length 801
   valid_centers = centers_full[::center_stride]                # stride-aligned
   ```
2. Window score = mean saliency inside the window:
   ```python
   kernel = np.ones(W) / W
   window_scores_full = np.convolve(saliency_norm, kernel, mode="valid")
   window_scores = window_scores_full[::center_stride]          # len == len(valid_centers)
   ```
3. Greedy selection (non-overlapping):
   - Rank candidate centers by `window_scores` descending.
   - Iteratively pick the best remaining center `c*`, append `(c* / fs, W_sec)` to the window list, and discard candidates whose windows overlap more than 25% of their length with `c*`.
   - Continue until `K = 3` windows are selected or no candidates remain.
4. Persist the chosen windows and associated metadata:
   - `center_time_sec = c / fs`
   - `window_seconds = W_sec`
   - `window_rank` (1..K)
   - `window_saliency = window_scores[idx]`
   - Optionally the raw saliency slice for diagnostics.

Each `(record_id, window_rank)` combination becomes a unique identifier (e.g., `window_id = f"{record_id}_w{window_rank}"`) that we store in `PerturbationConfig.extra["window_id"]` for traceability.

### 2.1.4 Attack Procedure per Window (Untargeted)

For each `(record_id, window)`:

1. Compute clean predictions once per sample:
   ```python
   proba_clean = predict_proba_single(model, x_clean)
   y_hat_clean = binarize_predictions(proba_clean, threshold=0.5)
   ```
2. Iterate over the strength schedule `[0.10, 0.20, 0.35, 0.50]`:
   ```python
   config = PerturbationConfig(
       ptype="smooth_adv",
       strength=strength,
       center_time=center_time_sec,
       window_seconds=2.0,
       target_class=None,
       extra={
           "window_id": window_id,
           "record_id": record_id,
           "saliency_rank": window_rank,
           # Defaults exposed in perturbations.adv_smooth can still be overridden:
           "eps_global_max": 0.5,
           "lambda_smooth": 10.0,
           "lambda_energy": 0.5,
           "steps": 200,
           "lr": 0.01,
       },
   )
   result, x_adv = run_attack_on_sample(
       model,
       x_clean,
       y_true,
       config,
       fs=100.0,
       rng=np.random.default_rng(seed_base + record_id),
   )
   ```
3. The helper returns an `AttackResult`; append it to a list and also track the minimal strength for this window:
   ```python
   if result.untargeted_success and minimal_strength is None:
       minimal_strength = strength
       best_result_for_window = result
       break
   ```
4. After the loop, record per-window summary in a dedicated table with columns:
   - `record_id`
   - `window_id`
   - `center_time`
   - `window_seconds`
   - `saliency_rank`
   - `saliency_mean`
   - `y_true_bits` (copy the 5-element vector so time/class analysis never has to refetch labels)
   - `minimal_strength` (float or `np.nan` if never succeeded)
   - `delta_norm_l2`, `delta_norm_linf`, `delta_smoothness` from `best_result_for_window` (if any)
   - `y_hat_clean`, `y_hat_adv` at that minimal strength

This summary table coexists with the raw `AttackResult` list produced by `run_attack_on_sample`, so downstream analysis can either look at every attempt or only the “best so far” view.

> **Implementation note:** extend `results_to_dataframe` so that it copies over `record_id`, `window_id`, `saliency_rank`, `saliency_mean`, and any other `config_extra` entries you need by calling `row.update(res.config_extra)`. Without this change the aggregation utilities in Section 2.1.5 cannot associate attempts with their parent windows.

### 2.1.5 Aggregation & Analysis

#### 2.1.5.1 Attack Success Rate vs. Time

1. Convert the per-window summary table into a DataFrame (via `pd.DataFrame` or by aggregating `results_to_dataframe` output grouped by `window_id`).
2. Define bins of width `0.5 s` (50 samples). Bin centers: `0.25, 0.75, …, 9.75`.
3. For each bin, compute:
   ```python
   bin_mask = (center_time >= bin_start) & (center_time < bin_end)
   N_total = bin_mask.sum()
   N_success = np.sum(bin_mask & ~df["minimal_strength"].isna())
   asr = N_success / N_total if N_total else np.nan
   ```
4. Plot ASR vs. time using a new helper `visualization.plot_asr_vs_time(df_windows, bin_width=0.5)` that follows the existing Matplotlib style in `visualization/visualization.py`. Include confidence intervals via binomial proportion CI if desired.
5. Interpretation guide: highlight peaks that line up with physiologically meaningful regions (e.g., R-peak clusters).

#### 2.1.5.2 Class-Conditioned Vulnerability

1. Ensure `df_windows` already contains `y_true_bits` (copied when each row is created; no joins are needed at analysis time).
2. For each class `c`, filter samples where `y_true[c] == 1`.
3. Recompute ASR per time bin on this subset, producing a matrix `ASR[class, bin]`.
4. Visualize via `visualization.plot_asr_time_class_heatmap(df_windows, class_names=CLASS_NAMES, bin_width=0.5)`, labeling axes with class names (`perturbations.config.CLASS_NAMES`) and bin centers.

This yields statements such as “`MI` windows between 3–6 s succeed 70% of the time, whereas `NORM` windows are uniformly <40%.”

---

## 2.2 How Much Strength is Needed? (Minimal Perturbation Analysis)

### 2.2.1 High-Level Goal

Using the same top-K windows from Section 2.1, determine the minimal smooth-adv strength that causes an untargeted success for each sample. The outcome is a single scalar `strength*_sample` per sample plus supporting per-window values.

### 2.2.2 Per-Window Minimal Strength with Binary Search

We refine the coarse schedule using a deterministic binary search. `smooth_adversarial_perturbation` initializes `delta_param` to zeros, so repeated calls with the same config and seed are deterministic—no warm-starting is required or supported at the moment.

Algorithm per window:

1. Run the coarse schedule described in 2.1.4, capturing the largest failing strength `s_fail` (default `0.0`) and the smallest success `s_succ` (if any).
2. If no success occurred up to `0.50`, mark the window as “unsuccessful (>=0.50)” and skip binary search.
3. Otherwise, run binary search with `max_binary_iters = 6` and `tolerance = 0.01`:
   ```python
   lo, hi = s_fail, s_succ
   for _ in range(max_binary_iters):
       if hi - lo < tolerance:
           break
       mid = 0.5 * (lo + hi)
       config.strength = mid
       result, _ = run_attack_on_sample(...)
       if result.untargeted_success:
           hi = mid
           best_result_for_window = result
       else:
           lo = mid
   strength_star_window = hi
   ```
4. Persist `strength_star_window`, the number of binary-search iterations, and the final `AttackResult` metrics into the per-window summary table (additional columns: `binary_iters`, `binary_converged`).

### 2.2.3 Best Window per Sample

1. For each sample `i`, gather its windows `{w_1, w_2, …}` and drop any where `strength_star_window` is missing.
2. If no window succeeded, mark the sample as **robust** up to `0.50`: `strength*_sample = np.nan`, `best_window_id = None`.
3. Otherwise:
   ```python
   best_row = rows.loc[rows["strength_star_window"].idxmin()]
   strength_star_sample = best_row["strength_star_window"]
   best_window_id = best_row["window_id"]
   ```
4. Create a sample-level table capturing:
   - `record_id`
   - `strength_star_sample`
   - `best_window_id`
   - `best_window_center_time`
   - `best_window_saliency_rank`
   - `y_true_bits` and `y_hat_clean` (copied from the clean evaluation once per sample).

### 2.2.4 Distribution Analysis

#### 2.2.4.1 All Samples

- Use the sample-level table to plot a histogram (bins from 0 to 0.5 in 0.025 increments) of `strength_star_sample` excluding NaNs via `visualization.plot_strength_histogram(df_samples)`.
- Compute summary stats: mean, median, 25th/75th percentiles.
- Report the robust fraction: `np.mean(sample_df["strength_star_sample"].isna())`.

#### 2.2.4.2 Per-Class Distributions

- For each class `c`, filter `sample_df` by `y_true[:, idx_c] == 1`.
- Plot boxplots for `strength_star_sample` grouped by class with `visualization.plot_strength_boxplot_by_class(df_samples, class_names=CLASS_NAMES)`.
- Optionally overlay per-class histograms to emphasize tails.

#### 2.2.4.3 Robust Samples

- Compute per-class robust fractions (count of NaN-strength samples / total samples for that class).
- Plot a simple bar chart (classes on x-axis, robust fraction on y-axis) to highlight which diagnoses remain resistant even at high strength using `visualization.plot_robust_fraction_by_class(...)`.

### 2.2.5 Storage Format & Paths

- Output root: `results/vulnerability/` (created if missing).
- Persist three canonical Parquet files plus optional CSV mirrors:
  1. `saliency_guided_attempts.parquet`: every `(record_id, window_id, strength)` attempt. Columns: base AttackResult metrics, `y_true_bits`, `y_hat_clean`, `y_hat_adv`, prediction probabilities, and all `config_extra` keys (window metadata).
  2. `saliency_guided_windows.parquet`: aggregated window-level table (pre- and post-binary-search) with `record_id`, `window_id`, `center_time`, saliency stats, `y_true_bits`, `minimal_strength`, `strength_star_window`, `binary_iters`, `binary_converged`, and norms from the best attempt.
  3. `minimal_strength_samples.parquet`: sample-level summary with `record_id`, `y_true_bits`, `y_hat_clean`, `strength_star_sample`, `best_window_id`, and references to the winning window stats.
- Helper functions in `perturbations.evaluation` handle serialization:
  ```python
  def save_vulnerability_results(df_attempts, df_windows, df_samples, root=\"results/vulnerability\"):
      ...

  def load_vulnerability_results(root=\"results/vulnerability\"):
      ...
  ```
- The helpers write Parquet plus companion CSVs (same base names) for quick inspection. Loading functions always return DataFrames with the standard schema so downstream notebooks remain decoupled from the storage choice.

### 2.2.6 Practical Considerations

- **Sample budget**: Start with 500 random `X_test` samples to validate the pipeline; expand to the full test fold once runtime is acceptable.
- **Window budget**: `K = 3` windows already yields `~1500` attack sequences for 500 samples; drop to `K = 2` if runtime is prohibitive.
- **Parallelization**: Attack attempts are independent per `(sample, window, strength)`; consider simple batching with multiprocessing or distributing across GPUs if available.
- **Reproducibility**: Seed `numpy.random.default_rng` with `seed_base + record_id` so each sample/window pair receives a deterministic RNG stream even when executed in parallel.
- **Logging**: Persist both the raw `AttackResult` list (via `results_to_dataframe`) and the window/sample summary tables to disk (CSV or Parquet) so downstream notebooks can reload without rerunning attacks.

### 2.2.7 Linking 2.1 and 2.2

Combining the analyses yields a two-dimensional narrative:

- **Spatial axis (Section 2.1)** pinpoints when in the ECG each class is most susceptible to localized smooth perturbations.
- **Magnitude axis (Section 2.2)** quantifies how strong the perturbation must be, enabling statements like “MI samples typically flip at 0.18 strength, whereas NORM flips at 0.11.”

These paired views should be surfaced together in the report to demonstrate both *where* and *how much* the model can be manipulated.
