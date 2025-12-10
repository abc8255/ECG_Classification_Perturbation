"""
Evaluation utilities for ECG perturbations.

These helpers standardize prediction, attack logging, metrics, and aggregation so
that different perturbation families can share a single analysis pipeline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional dependency
    tqdm = None

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import uniform_filter1d
from sklearn import metrics as sk_metrics

from .api import PerturbationConfig, apply_perturbation
from .config import CLASS_TO_INDEX, CLASS_NAMES
from .tf_utils import tf_device_scope

DEFAULT_STRENGTH_SCHEDULE: Tuple[float, ...] = (0.10, 0.20, 0.35, 0.50)
DEFAULT_NOISE_TYPES: Tuple[str, ...] = ("baseline_wander", "band_noise")
DEFAULT_NOISE_TRIALS: int = 3
SMOOTH_ATTACK_FAMILY = "smooth_adv"
NOISE_ATTACK_FAMILY = "noise"


@dataclass
class AttackResult:
    """
    Structured representation of a single perturbation attempt on one sample.
    """

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


def predict_proba_single(model, x: np.ndarray) -> np.ndarray:
    """
    Run the model on a single sample shaped (samples, leads) and return probs.
    """

    x = np.asarray(x, dtype=np.float32)
    with tf_device_scope():
        proba = model.predict(x[None, ...], verbose=0)
    return np.asarray(proba[0])


def predict_proba_batch(model, X: np.ndarray) -> np.ndarray:
    """
    Run the model on a batch shaped (N, samples, leads) and return probs.
    """

    X = np.asarray(X, dtype=np.float32)
    with tf_device_scope():
        proba = model.predict(X, verbose=0)
    return np.asarray(proba)


def binarize_predictions(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert probability vectors to binary predictions using the given threshold.
    """

    proba = np.asarray(proba)
    return (proba >= threshold).astype(int)


def compute_smoothness(delta: np.ndarray) -> float:
    """
    Smoothness metric defined as the mean squared first-order difference.
    """

    diffs = np.diff(delta, axis=0)
    return float(np.mean(np.square(diffs)))


def run_attack_on_sample(
    model,
    x_clean: np.ndarray,
    y_true: np.ndarray,
    config: PerturbationConfig,
    *,
    fs: float,
    rng: Optional[np.random.Generator] = None,
    threshold: float = 0.5,
    compute_delta_smoothness: bool = True,
    r_peaks: Optional[Sequence[int]] = None,
) -> Tuple[AttackResult, np.ndarray]:
    """
    Apply the configured perturbation, evaluate predictions, and log results.
    """

    x_clean = np.asarray(x_clean, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)

    x_adv = apply_perturbation(
        x_clean,
        fs=fs,
        config=config,
        model=model,
        y_true=y_true,
        r_peaks=r_peaks,
        rng=rng,
    )

    proba_clean = predict_proba_single(model, x_clean)
    proba_adv = predict_proba_single(model, x_adv)
    y_hat_clean = binarize_predictions(proba_clean, threshold=threshold)
    y_hat_adv = binarize_predictions(proba_adv, threshold=threshold)

    delta = x_adv - x_clean
    delta_norm_l2 = float(np.linalg.norm(delta))
    delta_norm_linf = float(np.max(np.abs(delta)))
    delta_smoothness = (
        compute_smoothness(delta) if compute_delta_smoothness else None
    )

    untargeted_success = bool(np.any(y_hat_clean != y_hat_adv))
    targeted_success: Optional[bool] = None
    if config.target_class is not None:
        idx = CLASS_TO_INDEX[config.target_class]
        mode = config.extra.get("target_mode", "force") if config.extra else "force"
        if mode == "suppress":
            targeted_success = bool(y_hat_clean[idx] == 1 and y_hat_adv[idx] == 0)
        else:
            targeted_success = bool(y_hat_adv[idx] == 1)

    result = AttackResult(
        record_id=config.extra.get("record_id") if config.extra else None,
        ptype=config.ptype,
        strength=config.strength,
        center_time=config.center_time,
        window_seconds=config.window_seconds,
        target_class=config.target_class,
        target_mode=config.extra.get("target_mode") if config.extra else None,
        config_extra=config.extra or {},
        y_true=y_true,
        y_hat_clean=y_hat_clean,
        y_hat_adv=y_hat_adv,
        y_proba_clean=proba_clean,
        y_proba_adv=proba_adv,
        delta_norm_l2=delta_norm_l2,
        delta_norm_linf=delta_norm_linf,
        delta_smoothness=delta_smoothness,
        untargeted_success=untargeted_success,
        targeted_success=targeted_success,
    )

    return result, x_adv


def compute_untargeted_asr(results: Sequence[AttackResult]) -> float:
    """
    Fraction of attacks where the binary prediction changed.
    """

    if not results:
        return 0.0
    return float(
        np.mean([1.0 if res.untargeted_success else 0.0 for res in results])
    )


def compute_targeted_success(
    results: Sequence[AttackResult],
    *,
    class_name: str,
    mode: str = "force",
) -> Tuple[float, int]:
    """
    Compute targeted success for a specific class and mode ('force' or 'suppress').
    """

    filtered = [
        res
        for res in results
        if res.target_class == class_name
        and (res.target_mode or "force") == mode
    ]
    if not filtered:
        return 0.0, 0

    idx = CLASS_TO_INDEX[class_name]
    if mode == "suppress":
        successes = [
            1.0 if (res.y_hat_clean[idx] == 1 and res.y_hat_adv[idx] == 0) else 0.0
            for res in filtered
        ]
    else:
        successes = [1.0 if res.y_hat_adv[idx] == 1 else 0.0 for res in filtered]
    return float(np.mean(successes)), len(filtered)


def _serialize_extra_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def results_to_dataframe(
    results: Sequence[AttackResult],
    *,
    include_probabilities: bool = False,
    include_vectors: bool = False,
) -> pd.DataFrame:
    """
    Convert a list of AttackResult objects into a DataFrame.
    """

    rows: List[Dict[str, Any]] = []
    for res in results:
        row = {
            "record_id": res.record_id,
            "ptype": res.ptype,
            "strength": res.strength,
            "center_time": res.center_time,
            "window_seconds": res.window_seconds,
            "target_class": res.target_class,
            "target_mode": res.target_mode,
            "delta_norm_l2": res.delta_norm_l2,
            "delta_norm_linf": res.delta_norm_linf,
            "delta_smoothness": res.delta_smoothness,
            "untargeted_success": res.untargeted_success,
            "targeted_success": res.targeted_success,
        }
        if include_probabilities:
            row["y_proba_clean"] = res.y_proba_clean
            row["y_proba_adv"] = res.y_proba_adv
        if include_vectors:
            row["y_true"] = res.y_true
            row["y_hat_clean"] = res.y_hat_clean
            row["y_hat_adv"] = res.y_hat_adv
        if res.config_extra:
            for key, value in res.config_extra.items():
                if key in row:
                    continue
                row[key] = _serialize_extra_value(value)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_norms(results: Sequence[AttackResult]) -> pd.DataFrame:
    """
    Group by (ptype, strength) and summarize L2/Linf norms.
    """

    df = results_to_dataframe(results)
    if df.empty:
        return df
    grouped = (
        df.groupby(["ptype", "strength"])[["delta_norm_l2", "delta_norm_linf"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    return grouped


def summarize_smoothness(results: Sequence[AttackResult]) -> pd.DataFrame:
    """
    Group by (ptype, strength) and summarize smoothness metrics.
    """

    df = results_to_dataframe(results)
    if df.empty or "delta_smoothness" not in df:
        return df
    grouped = (
        df.groupby(["ptype", "strength"])["delta_smoothness"]
        .agg(["mean", "std"])
        .reset_index()
    )
    return grouped


def compute_global_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute per-class ROC AUC and thresholded precision/recall/F1 metrics.
    """

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    metrics: Dict[str, Any] = {}

    try:
        auc_per_class = sk_metrics.roc_auc_score(
            y_true, y_proba, average=None, multi_class="ovr"
        )
    except ValueError:
        auc_per_class = [np.nan] * y_true.shape[1]
    metrics["auc_per_class"] = dict(zip(CLASS_NAMES, auc_per_class))
    metrics["auc_macro"] = float(
        np.nanmean(list(metrics["auc_per_class"].values()))
    )

    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics["precision_per_class"] = dict(zip(CLASS_NAMES, precision))
    metrics["recall_per_class"] = dict(zip(CLASS_NAMES, recall))
    metrics["f1_per_class"] = dict(zip(CLASS_NAMES, f1))
    metrics["f1_macro"] = float(np.mean(f1))
    return metrics


def select_eval_subset(
    y_test_enc: np.ndarray,
    max_samples: int = 1000,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """
    Multi-label-aware sampling of evaluation indices.
    """

    labels = np.asarray(y_test_enc)
    if labels.ndim == 1:
        labels = labels.reshape(-1, 1)
    num_samples = labels.shape[0]
    if max_samples >= num_samples:
        return np.arange(num_samples, dtype=int)

    rng = np.random.default_rng(random_state)
    class_count = labels.shape[1] if labels.shape[1] else 1
    per_class_target = max(1, max_samples // class_count)
    selected: List[np.ndarray] = []

    for idx in range(class_count):
        class_indices = np.flatnonzero(labels[:, idx] == 1)
        if class_indices.size == 0:
            continue
        take = min(per_class_target, class_indices.size)
        chosen = rng.choice(class_indices, size=take, replace=False)
        selected.append(chosen)

    if selected:
        indices = np.unique(np.concatenate(selected))
    else:
        indices = np.arange(min(max_samples, num_samples))

    if indices.size > max_samples:
        indices = np.sort(rng.choice(indices, size=max_samples, replace=False))
    elif indices.size < max_samples:
        remaining = np.setdiff1d(np.arange(num_samples), indices, assume_unique=True)
        if remaining.size:
            extra = rng.choice(
                remaining, size=min(max_samples - indices.size, remaining.size), replace=False
            )
            indices = np.sort(np.concatenate([indices, extra]))
    return indices.astype(int)


def compute_time_saliency(
    x_clean: np.ndarray,
    y_true_vec: np.ndarray,
    model,
    *,
    fs: float = 100.0,
    smooth_size: int = 11,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute a normalized time saliency curve for a single sample.
    """

    _ = fs  # kept for signature clarity; currently unused but reserved for future logic.
    x_np = np.asarray(x_clean, dtype=np.float32)
    y_np = np.asarray(y_true_vec, dtype=np.float32).reshape(1, -1)
    with tf_device_scope():
        tensor_x = tf.convert_to_tensor(x_np[None, ...])
        tensor_y = tf.convert_to_tensor(y_np)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        with tf.GradientTape() as tape:
            tape.watch(tensor_x)
            proba = model(tensor_x, training=False)
            loss = bce(tensor_y, proba)
        grads = tape.gradient(loss, tensor_x)
        if grads is None:
            raise RuntimeError("Unable to compute gradients for saliency.")
        grad_np = grads.numpy()[0]
    saliency = np.linalg.norm(grad_np, ord=2, axis=-1)
    if smooth_size > 1:
        saliency = uniform_filter1d(saliency, size=smooth_size, mode="nearest")
    if normalize:
        saliency = saliency / (np.max(saliency) + 1e-6)
    return saliency


def select_salient_windows(
    saliency: np.ndarray,
    *,
    fs: float = 100.0,
    window_seconds: float = 2.0,
    K: int = 3,
    center_stride: int = 5,
    max_overlap: float = 0.25,
) -> List[Dict[str, Any]]:
    """
    Select up to K high-saliency, minimally overlapping windows.
    """

    saliency = np.asarray(saliency, dtype=float)
    length = saliency.shape[0]
    window_samples = int(round(window_seconds * fs))
    if window_samples <= 0 or window_samples > length:
        raise ValueError("window_seconds leads to invalid window length for saliency array.")
    half = window_samples // 2
    centers_full = np.arange(half, length - half + 1)
    valid_centers = centers_full[::center_stride]
    kernel = np.ones(window_samples) / window_samples
    window_scores_full = np.convolve(saliency, kernel, mode="valid")
    window_scores = window_scores_full[::center_stride]

    selected: List[Dict[str, Any]] = []
    for idx in np.argsort(window_scores)[::-1]:
        if len(selected) >= K:
            break
        center = int(valid_centers[idx])
        score = float(window_scores[idx])
        too_close = False
        for existing in selected:
            overlap = 1.0 - abs(center - existing["center_index"]) / window_samples
            if overlap > max_overlap:
                too_close = True
                break
        if too_close:
            continue
        selected.append({"center_index": center, "saliency_mean": score})

    selected.sort(key=lambda item: item["saliency_mean"], reverse=True)
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        item["center_time"] = item["center_index"] / fs
        item["window_seconds"] = window_seconds
    return selected


def run_saliency_guided_attacks(
    X: np.ndarray,
    Y: np.ndarray,
    model,
    eval_indices: Sequence[int],
    *,
    fs: float = 100.0,
    strength_schedule: Optional[Sequence[float]] = None,
    window_seconds: float = 2.0,
    top_k: int = 3,
    center_stride: int = 5,
    max_overlap: float = 0.25,
    seed_base: int = 0,
    saliency_cache: Optional[Dict[int, np.ndarray]] = None,
    progress: bool = True,
) -> Tuple[List[AttackResult], pd.DataFrame]:
    """
    Compute saliency, select windows, run smooth_adv attacks, and log attempts.
    """

    schedule = tuple(strength_schedule) if strength_schedule else DEFAULT_STRENGTH_SCHEDULE
    attack_results: List[AttackResult] = []
    window_rows: List[Dict[str, Any]] = []
    cache = saliency_cache or {}
    record_ids = [int(idx) for idx in eval_indices]
    total = len(record_ids)

    for processed, record_id in enumerate(record_ids, start=1):
        x_clean = np.asarray(X[record_id], dtype=np.float32)
        y_true = np.asarray(Y[record_id], dtype=int)
        saliency = cache.get(record_id)
        if saliency is None:
            saliency = compute_time_saliency(x_clean, y_true, model, fs=fs)
            cache[record_id] = saliency
        windows = select_salient_windows(
            saliency,
            fs=fs,
            window_seconds=window_seconds,
            K=top_k,
            center_stride=center_stride,
            max_overlap=max_overlap,
        )
        y_true_bits = y_true.astype(int).tolist()

        for window in windows:
            window_id = f"{record_id}_w{window['rank']}"
            minimal_strength: Optional[float] = None
            best_result: Optional[AttackResult] = None
            first_clean: Optional[List[int]] = None
            attempt_idx = 0

            for strength in schedule:
                cfg = PerturbationConfig(
                    ptype="smooth_adv",
                    strength=float(strength),
                    center_time=float(window["center_time"]),
                    window_seconds=window_seconds,
                    target_class=None,
                    extra={
                        "attack_family": SMOOTH_ATTACK_FAMILY,
                        "record_id": record_id,
                        "window_id": window_id,
                        "saliency_rank": window["rank"],
                        "saliency_mean": window["saliency_mean"],
                    },
                )
                rng_seed = seed_base + record_id * 100 + window["rank"] * 10 + attempt_idx
                result, _ = run_attack_on_sample(
                    model,
                    x_clean,
                    y_true,
                    cfg,
                    fs=fs,
                    rng=np.random.default_rng(rng_seed),
                )
                attack_results.append(result)
                attempt_idx += 1
                if first_clean is None:
                    first_clean = result.y_hat_clean.astype(int).tolist()
                if result.untargeted_success and minimal_strength is None:
                    minimal_strength = float(strength)
                    best_result = result
                    break

            window_rows.append(
                {
                    "record_id": record_id,
                    "window_id": window_id,
                    "attack_family": SMOOTH_ATTACK_FAMILY,
                    "noise_type": None,
                    "center_time": window["center_time"],
                    "center_index": window["center_index"],
                    "window_seconds": window_seconds,
                    "saliency_rank": window["rank"],
                    "saliency_mean": window["saliency_mean"],
                    "y_true_bits": y_true_bits,
                    "minimal_strength": minimal_strength,
                    "strength_star_window": minimal_strength,
                    "binary_iters": 0,
                    "binary_converged": False,
                    "delta_norm_l2": best_result.delta_norm_l2 if best_result else np.nan,
                    "delta_norm_linf": best_result.delta_norm_linf if best_result else np.nan,
                    "delta_smoothness": best_result.delta_smoothness if best_result else np.nan,
                    "y_hat_clean": best_result.y_hat_clean.astype(int).tolist()
                    if best_result is not None
                    else first_clean,
                    "y_hat_adv": best_result.y_hat_adv.astype(int).tolist()
                    if best_result is not None
                    else None,
                }
            )

        if progress:
            logging.info(
                "[smooth] Processed %d/%d samples (record_id=%s)",
                processed,
                total,
                record_id,
            )

    df_windows = pd.DataFrame(window_rows)
    return attack_results, df_windows


def run_noise_vulnerability_experiment(
    X: np.ndarray,
    Y: np.ndarray,
    model,
    eval_indices: Sequence[int],
    *,
    fs: float = 100.0,
    noise_types: Optional[Sequence[str]] = None,
    strength_schedule: Optional[Sequence[float]] = None,
    n_trials: int = DEFAULT_NOISE_TRIALS,
    window_seconds: float = 2.0,
    top_k: int = 3,
    center_stride: int = 5,
    max_overlap: float = 0.25,
    seed_base: int = 0,
    saliency_cache: Optional[Dict[int, np.ndarray]] = None,
    progress: bool = True,
    report_every: int = 50,
) -> Tuple[List[AttackResult], pd.DataFrame]:
    """
    Run stochastic noise perturbations on saliency-selected windows.
    """

    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")
    schedule = tuple(strength_schedule) if strength_schedule else DEFAULT_STRENGTH_SCHEDULE
    families = tuple(noise_types) if noise_types else DEFAULT_NOISE_TYPES
    if not families:
        raise ValueError("noise_types must contain at least one entry.")

    attack_results: List[AttackResult] = []
    window_rows: List[Dict[str, Any]] = []
    cache = saliency_cache or {}
    record_ids = [int(idx) for idx in eval_indices]
    iterator: Iterable[int]
    if progress and tqdm is not None:
        iterator = tqdm(record_ids, desc="Noise vulnerability", unit="sample")
    else:
        iterator = record_ids

    start_time = time.time()
    total = len(record_ids)

    for processed, record_id in enumerate(iterator, start=1):
        x_clean = np.asarray(X[record_id], dtype=np.float32)
        y_true = np.asarray(Y[record_id], dtype=int)
        saliency = cache.get(record_id)
        if saliency is None:
            saliency = compute_time_saliency(x_clean, y_true, model, fs=fs)
            cache[record_id] = saliency
        windows = select_salient_windows(
            saliency,
            fs=fs,
            window_seconds=window_seconds,
            K=top_k,
            center_stride=center_stride,
            max_overlap=max_overlap,
        )
        y_true_bits = y_true.astype(int).tolist()

        for window in windows:
            window_id = f"{record_id}_w{window['rank']}"
            center_time = float(window["center_time"])
            center_index = int(window["center_index"])
            saliency_rank = int(window["rank"])
            saliency_mean = float(window["saliency_mean"])

            for noise_offset, noise_type in enumerate(families):
                minimal_strength: Optional[float] = None
                best_result: Optional[AttackResult] = None
                first_clean: Optional[List[int]] = None
                success_rates: Dict[float, float] = {}

                for strength_idx, strength in enumerate(schedule):
                    successes = 0
                    for trial_idx in range(n_trials):
                        cfg = PerturbationConfig(
                            ptype=noise_type,
                            strength=float(strength),
                            center_time=center_time,
                            window_seconds=window_seconds,
                            target_class=None,
                            extra={
                                "attack_family": NOISE_ATTACK_FAMILY,
                                "noise_type": noise_type,
                                "record_id": record_id,
                                "window_id": window_id,
                                "saliency_rank": saliency_rank,
                                "saliency_mean": saliency_mean,
                                "trial_index": trial_idx,
                                "n_noise_trials": n_trials,
                                "strength_schedule": list(schedule),
                            },
                        )
                        seed = (
                            seed_base
                            + record_id * 10_000
                            + saliency_rank * 1_000
                            + noise_offset * 100
                            + strength_idx * 10
                            + trial_idx
                        )
                        result, _ = run_attack_on_sample(
                            model,
                            x_clean,
                            y_true,
                            cfg,
                            fs=fs,
                            rng=np.random.default_rng(seed),
                        )
                        attack_results.append(result)
                        if first_clean is None:
                            first_clean = result.y_hat_clean.astype(int).tolist()
                        if result.untargeted_success:
                            successes += 1
                            if minimal_strength is None:
                                minimal_strength = float(strength)
                                best_result = result
                    success_rates[float(strength)] = successes / float(n_trials)

                success_rates_json = json.dumps(
                    [
                        {"strength": float(strength), "success_rate": float(success_rates[strength])}
                        for strength in schedule
                    ]
                )
                window_rows.append(
                    {
                        "record_id": record_id,
                        "window_id": window_id,
                        "attack_family": NOISE_ATTACK_FAMILY,
                        "noise_type": noise_type,
                        "center_time": center_time,
                        "center_index": center_index,
                        "window_seconds": window_seconds,
                        "saliency_rank": saliency_rank,
                        "saliency_mean": saliency_mean,
                        "y_true_bits": y_true_bits,
                        "minimal_strength": minimal_strength,
                        "strength_star_window": minimal_strength,
                        "success_rate_by_strength": success_rates_json,
                        "binary_iters": 0,
                        "binary_converged": False,
                        "delta_norm_l2": best_result.delta_norm_l2 if best_result else np.nan,
                        "delta_norm_linf": best_result.delta_norm_linf if best_result else np.nan,
                        "delta_smoothness": best_result.delta_smoothness if best_result else np.nan,
                        "y_hat_clean": best_result.y_hat_clean.astype(int).tolist()
                        if best_result is not None
                        else first_clean,
                        "y_hat_adv": best_result.y_hat_adv.astype(int).tolist()
                        if best_result is not None
                        else None,
                    }
                )

        if progress:
            logging.info(
                "[noise] Processed %d/%d samples (record_id=%s)",
                processed,
                total,
                record_id,
            )
        if progress and report_every > 0 and processed % report_every == 0 and total:
            elapsed = max(time.time() - start_time, 1e-9)
            rate = processed / elapsed
            remaining = total - processed
            eta = remaining / rate if rate > 0 else float("nan")
            logging.info(
                "[noise] Processed %d/%d samples (%.1f%%) - %.2f samples/s - ETA %.1f min",
                processed,
                total,
                100.0 * processed / total,
                rate,
                eta / 60.0 if np.isfinite(eta) else float("nan"),
            )

    df_windows = pd.DataFrame(window_rows)
    return attack_results, df_windows


def refine_window_strengths_with_binary_search(
    df_windows: pd.DataFrame,
    df_attempts: pd.DataFrame,
    model,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    fs: float = 100.0,
    tolerance: float = 0.01,
    max_iters: int = 6,
    seed_base: int = 0,
) -> Tuple[pd.DataFrame, List[AttackResult]]:
    """
    Refine minimal strengths via binary search between fail/success brackets.
    """

    if df_windows.empty:
        return df_windows.copy(), []

    updated = df_windows.copy()
    if "strength_star_window" not in updated.columns:
        updated["strength_star_window"] = updated["minimal_strength"]
    if "binary_iters" not in updated.columns:
        updated["binary_iters"] = 0
    if "binary_converged" not in updated.columns:
        updated["binary_converged"] = False

    new_results: List[AttackResult] = []
    metric_col = "strength_star_window"

    for idx, row in updated.iterrows():
        minimal = row["minimal_strength"]
        if pd.isna(minimal):
            continue
        window_id = row["window_id"]
        record_id = int(row["record_id"])
        if "window_id" not in df_attempts.columns:
            continue
        attempts = df_attempts[df_attempts["window_id"] == window_id]
        if attempts.empty:
            continue
        successes = attempts[attempts["untargeted_success"]].sort_values("strength")
        if successes.empty:
            continue
        s_succ = float(successes["strength"].iloc[0])
        fails = attempts[(attempts["strength"] < s_succ) & (~attempts["untargeted_success"])]
        s_fail = float(fails["strength"].max()) if not fails.empty else 0.0

        lo, hi = s_fail, s_succ
        converged = hi - lo < tolerance
        best_result: Optional[AttackResult] = None
        iters_used = 0

        if not converged:
            for iter_idx in range(1, max_iters + 1):
                mid = 0.5 * (lo + hi)
                cfg = PerturbationConfig(
                    ptype="smooth_adv",
                    strength=mid,
                    center_time=float(row["center_time"]),
                    window_seconds=float(row["window_seconds"]),
                    target_class=None,
                    extra={
                        "record_id": record_id,
                        "window_id": window_id,
                        "saliency_rank": int(row["saliency_rank"]),
                        "saliency_mean": float(row["saliency_mean"]),
                    },
                )
                rng_seed = seed_base + record_id * 100 + int(row["saliency_rank"]) * 10 + iter_idx
                result, _ = run_attack_on_sample(
                    model,
                    np.asarray(X[record_id], dtype=np.float32),
                    np.asarray(Y[record_id], dtype=int),
                    cfg,
                    fs=fs,
                    rng=np.random.default_rng(rng_seed),
                )
                new_results.append(result)
                iters_used = iter_idx
                if result.untargeted_success:
                    hi = mid
                    best_result = result
                else:
                    lo = mid
                if hi - lo < tolerance:
                    converged = True
                    break

        updated.at[idx, "binary_iters"] = iters_used
        updated.at[idx, "binary_converged"] = converged
        if best_result is not None:
            updated.at[idx, metric_col] = hi
            updated.at[idx, "delta_norm_l2"] = best_result.delta_norm_l2
            updated.at[idx, "delta_norm_linf"] = best_result.delta_norm_linf
            updated.at[idx, "delta_smoothness"] = best_result.delta_smoothness
            updated.at[idx, "y_hat_adv"] = best_result.y_hat_adv.astype(int).tolist()

    return updated, new_results


def summarize_minimal_strength_per_sample(df_windows: pd.DataFrame) -> pd.DataFrame:
    """
    Build a sample-level table from per-window summaries.
    """

    if df_windows.empty:
        return pd.DataFrame(
            columns=[
                "record_id",
                "strength_star_sample",
                "best_window_id",
                "best_window_center_time",
                "best_window_saliency_rank",
                "best_window_saliency_mean",
                "y_true_bits",
                "y_hat_clean",
            ]
        )

    metric_col = "strength_star_window" if "strength_star_window" in df_windows else "minimal_strength"
    rows: List[Dict[str, Any]] = []
    for record_id, group in df_windows.groupby("record_id"):
        valid = group[~group[metric_col].isna()]
        if valid.empty:
            best_row = None
            strength = np.nan
        else:
            idx_best = valid[metric_col].astype(float).idxmin()
            best_row = group.loc[idx_best]
            strength = float(best_row[metric_col])
        base_row = group.iloc[0]
        row = {
            "record_id": int(record_id),
            "strength_star_sample": strength,
            "best_window_id": best_row["window_id"] if best_row is not None else None,
            "best_window_center_time": float(best_row["center_time"]) if best_row is not None else np.nan,
            "best_window_saliency_rank": int(best_row["saliency_rank"]) if best_row is not None else np.nan,
            "best_window_saliency_mean": float(best_row["saliency_mean"]) if best_row is not None else np.nan,
            "y_true_bits": base_row["y_true_bits"],
            "y_hat_clean": best_row["y_hat_clean"] if best_row is not None else base_row["y_hat_clean"],
        }
        if "attack_family" in group.columns:
            row["attack_family"] = (
                best_row["attack_family"] if best_row is not None else base_row.get("attack_family")
            )
        if "noise_type" in group.columns:
            row["best_noise_type"] = best_row["noise_type"] if best_row is not None else base_row.get("noise_type")
        rows.append(row)
    return pd.DataFrame(rows)


def save_vulnerability_results(
    df_attempts: pd.DataFrame,
    df_windows: pd.DataFrame,
    df_samples: pd.DataFrame,
    *,
    root: str = "results/vulnerability",
) -> None:
    """
    Persist vulnerability experiment outputs (Parquet + CSV).
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, stem: str) -> None:
        parquet_path = root_path / f"{stem}.parquet"
        csv_path = root_path / f"{stem}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

    _write(df_attempts, "saliency_guided_attempts")
    _write(df_windows, "saliency_guided_windows")
    _write(df_samples, "minimal_strength_samples")


def load_vulnerability_results(
    root: str = "results/vulnerability",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved vulnerability experiment outputs.
    """

    root_path = Path(root)

    def _read(stem: str) -> pd.DataFrame:
        parquet_path = root_path / f"{stem}.parquet"
        csv_path = root_path / f"{stem}.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    return (
        _read("saliency_guided_attempts"),
        _read("saliency_guided_windows"),
        _read("minimal_strength_samples"),
    )


def save_noise_vulnerability_results(
    df_attempts: pd.DataFrame,
    df_windows: pd.DataFrame,
    df_samples: pd.DataFrame,
    *,
    root: str = "results/vulnerability",
) -> None:
    """
    Persist noise vulnerability experiment outputs.
    """

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    def _write(df: pd.DataFrame, stem: str) -> None:
        parquet_path = root_path / f"{stem}.parquet"
        csv_path = root_path / f"{stem}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

    _write(df_attempts, "noise_attempts")
    _write(df_windows, "noise_windows")
    _write(df_samples, "noise_samples")


def load_noise_vulnerability_results(
    root: str = "results/vulnerability",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load saved noise vulnerability experiment outputs.
    """

    root_path = Path(root)

    def _read(stem: str) -> pd.DataFrame:
        parquet_path = root_path / f"{stem}.parquet"
        csv_path = root_path / f"{stem}.csv"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    return _read("noise_attempts"), _read("noise_windows"), _read("noise_samples")


def load_all_vulnerability_results(
    root: str = "results/vulnerability",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load both smooth_adv and noise vulnerability tables and concatenate them.
    """

    smooth = load_vulnerability_results(root)
    noise = load_noise_vulnerability_results(root)

    def _concat(idx: int) -> pd.DataFrame:
        frames = [smooth[idx], noise[idx]]
        frames = [df for df in frames if not df.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    return _concat(0), _concat(1), _concat(2)
