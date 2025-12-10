"""
Visualization utilities for ECG perturbations.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from perturbations.config import CLASS_NAMES
from perturbations.evaluation import AttackResult, results_to_dataframe


def _ensure_axes(ax=None, nrows=1, ncols=1, figsize=(10, 6)):
    if ax is not None:
        return ax.figure, ax
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    return fig, axes


def _label_matrix_from_series(series: pd.Series, class_count: int) -> np.ndarray:
    rows: List[np.ndarray] = []
    for value in series:
        arr = np.asarray(value, dtype=int)
        if arr.size != class_count:
            raise ValueError(
                f"y_true_bits entries must have length {class_count}, received {arr.size}"
            )
        rows.append(arr)
    if not rows:
        return np.zeros((0, class_count), dtype=int)
    return np.vstack(rows)


def _compute_asr_bins(
    times: np.ndarray,
    success_mask: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    asr = np.full(len(bin_edges) - 1, np.nan, dtype=float)
    for idx in range(len(bin_edges) - 1):
        start, end = bin_edges[idx], bin_edges[idx + 1]
        mask = (times >= start) & (times < end)
        total = mask.sum()
        if total:
            successes = success_mask[mask].sum()
            asr[idx] = successes / total
    return asr


def plot_triptych(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    *,
    fs: float,
    lead_idx: int = 1,
    title: Optional[str] = None,
    zoom: Optional[Tuple[float, float]] = None,
    share_y: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot clean, perturbed, and difference signals for a single lead.
    """

    times = np.arange(x_clean.shape[0]) / fs
    lead_idx = int(lead_idx)
    x_clean_lead = x_clean[:, lead_idx]
    x_adv_lead = x_adv[:, lead_idx]
    delta_lead = x_adv_lead - x_clean_lead

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax_clean, ax_adv, ax_delta = axes

    ax_clean.plot(times, x_clean_lead, color="tab:blue")
    ax_clean.set_ylabel("Amplitude")
    ax_clean.set_title(f"Lead {lead_idx + 1} – Clean")

    ax_adv.plot(times, x_adv_lead, color="tab:orange")
    ax_adv.set_ylabel("Amplitude")
    ax_adv.set_title(f"Lead {lead_idx + 1} – Perturbed")

    if share_y:
        ymin = min(ax_clean.get_ylim()[0], ax_adv.get_ylim()[0])
        ymax = max(ax_clean.get_ylim()[1], ax_adv.get_ylim()[1])
        ax_clean.set_ylim(ymin, ymax)
        ax_adv.set_ylim(ymin, ymax)

    ax_delta.plot(times, delta_lead, color="tab:green")
    ax_delta.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_delta.set_ylabel("Difference")
    ax_delta.set_xlabel("Time (s)")
    ax_delta.set_title("Perturbed − Clean")
    max_delta = np.max(np.abs(delta_lead))
    ax_delta.text(
        0.01,
        0.9,
        f"max |Δ| = {max_delta:.4f}",
        transform=ax_delta.transAxes,
        fontsize=9,
    )

    if zoom:
        ax_clean.set_xlim(zoom)
        ax_adv.set_xlim(zoom)
        ax_delta.set_xlim(zoom)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_overlay_with_diff(
    x_clean: np.ndarray,
    x_adv: np.ndarray,
    *,
    fs: float,
    lead_idx: int = 1,
    zoom: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot clean vs perturbed traces overlayed plus a difference panel.
    """

    times = np.arange(x_clean.shape[0]) / fs
    lead_idx = int(lead_idx)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax_overlay, ax_diff = axes

    ax_overlay.plot(times, x_clean[:, lead_idx], label="Clean", color="tab:blue")
    ax_overlay.plot(
        times, x_adv[:, lead_idx], label="Perturbed", color="tab:orange", alpha=0.8
    )
    ax_overlay.set_ylabel("Amplitude")
    ax_overlay.legend(loc="upper right")
    ax_overlay.set_title(f"Lead {lead_idx + 1} – Overlay")

    delta = x_adv[:, lead_idx] - x_clean[:, lead_idx]
    ax_diff.plot(times, delta, color="tab:green")
    ax_diff.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_diff.set_ylabel("Difference")
    ax_diff.set_xlabel("Time (s)")
    ax_diff.set_title("Perturbed − Clean")

    if zoom:
        ax_overlay.set_xlim(zoom)
        ax_diff.set_xlim(zoom)
    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def plot_asr_vs_strength(
    results: Sequence[AttackResult],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot untargeted ASR as a function of strength for each perturbation type.
    """

    df = results_to_dataframe(results)
    if df.empty:
        raise ValueError("No results provided for ASR plotting.")
    grouped = (
        df.groupby(["ptype", "strength"])["untargeted_success"]
        .mean()
        .reset_index()
    )

    fig, ax = _ensure_axes(ax, figsize=(8, 5))
    pivot = grouped.pivot(index="strength", columns="ptype", values="untargeted_success")
    for ptype in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[ptype],
            marker="o",
            label=ptype,
        )

    ax.set_xlabel("Strength")
    ax.set_ylabel("Untargeted ASR")
    ax.set_ylim(0, 1)
    ax.legend(title="Perturbation")
    if title:
        ax.set_title(title)
    return fig, ax


def plot_norm_distributions(
    results: Sequence[AttackResult],
    *,
    value: str = "delta_norm_l2",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Boxplot of norm/smoothness metrics grouped by perturbation type and strength.
    """

    df = results_to_dataframe(results)
    if df.empty or value not in df.columns:
        raise ValueError(f"No '{value}' values available for plotting.")

    df["label"] = df.apply(
        lambda row: f"{row['ptype']} (s={row['strength']})", axis=1
    )

    labels = df["label"].unique()
    data = [df[df["label"] == label][value].dropna() for label in labels]

    fig, ax = _ensure_axes(ax, figsize=(10, 5))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(value)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_classwise_metric_bars(
    metrics: Dict[str, Dict[str, float]],
    *,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    metric_name: str = "AUC",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot per-class metrics for different evaluation conditions.
    """

    conditions = list(metrics.keys())
    classes = CLASS_NAMES
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(conditions))

    fig, ax = _ensure_axes(ax, figsize=(10, 5))
    for idx, condition in enumerate(conditions):
        scores = [metrics[condition].get(cls, np.nan) for cls in classes]
        offset = (idx - (len(conditions) - 1) / 2) * width
        ax.bar(x + offset, scores, width=width, label=condition)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylabel(metric_name)
    ax.set_ylim(0, 1)
    ax.legend(title="Condition")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_asr_vs_time(
    df_windows: pd.DataFrame,
    *,
    bin_width: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot untargeted ASR against window center time.
    """

    if df_windows.empty:
        raise ValueError("df_windows is empty; nothing to plot.")
    metric_col = "strength_star_window" if "strength_star_window" in df_windows else "minimal_strength"
    centers = df_windows["center_time"].to_numpy(dtype=float)
    duration = max(10.0, float(np.nanmax(centers)) if centers.size else 10.0)
    bin_edges = np.arange(0.0, duration + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([0.0, bin_width])
    success_mask = df_windows[metric_col].notna().to_numpy()
    asr = _compute_asr_bins(centers, success_mask.astype(float), bin_edges)
    bin_centers = bin_edges[:-1] + bin_width / 2

    fig, ax = _ensure_axes(ax, figsize=(8, 4))
    ax.plot(bin_centers, asr, marker="o")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Window Center Time (s)")
    ax.set_ylabel("Attack Success Rate")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_asr_time_class_heatmap(
    df_windows: pd.DataFrame,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    bin_width: float = 0.5,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of ASR as a function of time and class membership.
    """

    if df_windows.empty:
        raise ValueError("df_windows is empty; nothing to plot.")
    metric_col = "strength_star_window" if "strength_star_window" in df_windows else "minimal_strength"
    centers = df_windows["center_time"].to_numpy(dtype=float)
    duration = max(10.0, float(np.nanmax(centers)) if centers.size else 10.0)
    bin_edges = np.arange(0.0, duration + bin_width, bin_width)
    if len(bin_edges) < 2:
        bin_edges = np.array([0.0, bin_width])
    label_matrix = _label_matrix_from_series(df_windows["y_true_bits"], len(class_names))
    success_mask = df_windows[metric_col].notna().to_numpy()
    heatmap = np.full((len(class_names), len(bin_edges) - 1), np.nan, dtype=float)

    for idx, _ in enumerate(class_names):
        class_rows = label_matrix[:, idx] == 1
        if not np.any(class_rows):
            continue
        class_asr = _compute_asr_bins(
            centers[class_rows],
            success_mask[class_rows].astype(float),
            bin_edges,
        )
        heatmap[idx] = class_asr

    fig, ax = _ensure_axes(ax, figsize=(10, 4))
    cax = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[bin_edges[0], bin_edges[-1], -0.5, len(class_names) - 0.5],
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Window Center Time (s)")
    ax.set_ylabel("Class")
    if title:
        ax.set_title(title)
    fig.colorbar(cax, ax=ax, label="ASR")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_strength_histogram(
    df_samples: pd.DataFrame,
    *,
    max_strength: float = 0.5,
    bin_width: float = 0.025,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Histogram of minimal strengths across samples.
    """

    strengths = df_samples["strength_star_sample"].dropna().to_numpy(dtype=float)
    bins = np.arange(0.0, max_strength + bin_width, bin_width)
    if strengths.size == 0:
        raise ValueError("No successful samples to plot.")
    fig, ax = _ensure_axes(ax, figsize=(8, 4))
    ax.hist(strengths, bins=bins, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Minimal Strength*")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_strength_boxplot_by_class(
    df_samples: pd.DataFrame,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Boxplots of minimal strength grouped by ground-truth class.
    """

    label_matrix = _label_matrix_from_series(df_samples["y_true_bits"], len(class_names))
    strengths = df_samples["strength_star_sample"].to_numpy()
    data: List[np.ndarray] = []
    class_labels: List[str] = []
    for idx, name in enumerate(class_names):
        class_mask = label_matrix[:, idx] == 1
        class_strengths = strengths[class_mask]
        class_strengths = class_strengths[~np.isnan(class_strengths)]
        if class_strengths.size == 0:
            continue
        data.append(class_strengths)
        class_labels.append(name)
    if not data:
        raise ValueError("No class-specific data available for boxplot.")

    fig, ax = _ensure_axes(ax, figsize=(10, 4))
    ax.boxplot(data, labels=class_labels, showmeans=True)
    ax.set_ylabel("Minimal Strength*")
    ax.set_xlabel("Class")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


def plot_robust_fraction_by_class(
    df_samples: pd.DataFrame,
    *,
    class_names: Sequence[str] = CLASS_NAMES,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart of the fraction of robust samples per class.
    """

    label_matrix = _label_matrix_from_series(df_samples["y_true_bits"], len(class_names))
    robust_mask = df_samples["strength_star_sample"].isna().to_numpy()
    fractions: List[float] = []
    for idx in range(len(class_names)):
        class_mask = label_matrix[:, idx] == 1
        if not np.any(class_mask):
            fractions.append(np.nan)
        else:
            fractions.append(float(np.mean(robust_mask[class_mask])))

    fig, ax = _ensure_axes(ax, figsize=(8, 4))
    ax.bar(range(len(class_names)), fractions, color="tab:green")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Robust Fraction")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
