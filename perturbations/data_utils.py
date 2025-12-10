"""
Utilities for loading PTB-XL data and preparing splits for training/evaluation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import MultiLabelBinarizer

from .config import CLASS_NAMES


def load_ptbxl_metadata(ptb_root: Path) -> pd.DataFrame:
    """
    Load ptbxl_database.csv with parsed SCP code dictionaries.
    """

    df = pd.read_csv(ptb_root / "ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def load_diagnostic_map(ptb_root: Path) -> pd.Series:
    """
    Load diagnostic aggregation table filtered to diagnostic statements.
    """

    agg_df = pd.read_csv(ptb_root / "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df["diagnostic"] == 1]
    return agg_df["diagnostic_class"]


def aggregate_superclasses(
    scp_codes: Dict[str, float],
    diag_map: pd.Series,
) -> List[str]:
    """
    Map SCP codes to diagnostic superclasses and return a deduplicated list.
    """

    classes: List[str] = []
    for code in scp_codes.keys():
        if code in diag_map.index:
            classes.append(diag_map.loc[code])
    return sorted(set(classes))


def attach_superclasses(df: pd.DataFrame, diag_map: pd.Series) -> pd.DataFrame:
    """
    Return a copy of `df` with a `superclasses` column.
    """

    enriched = df.copy()
    enriched["superclasses"] = enriched["scp_codes"].apply(
        lambda codes: aggregate_superclasses(codes, diag_map)
    )
    return enriched


def encode_superclasses(superclasses: Iterable[Sequence[str]]) -> np.ndarray:
    """
    Convert iterable of superclass lists into a multi-label binary matrix.
    """

    mlb = MultiLabelBinarizer(classes=list(CLASS_NAMES))
    mlb.fit([list(CLASS_NAMES)])
    return mlb.transform(superclasses).astype(np.float32)


def load_ptbxl_signals(
    df: pd.DataFrame,
    ptb_root: Path,
    *,
    sampling_rate: int = 100,
    progress_interval: int = 200,
) -> np.ndarray:
    """
    Load ECG signals specified in the dataframe into a NumPy array.
    """

    column = "filename_lr" if sampling_rate <= 100 else "filename_hr"
    signals: List[np.ndarray] = []
    for idx, rel_path in enumerate(df[column].tolist(), start=1):
        record_path = ptb_root / rel_path
        signal, _ = wfdb.rdsamp(str(record_path))
        signals.append(signal.astype(np.float32))
        if progress_interval and idx % progress_interval == 0:
            print(f"... loaded {idx}/{len(df)} segments", flush=True)
    return np.stack(signals)


def sample_dataframe(
    df: pd.DataFrame,
    max_rows: Optional[int],
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample up to `max_rows` rows without replacement (if specified).
    """

    if max_rows is None or max_rows >= len(df):
        return df
    return df.sample(n=max_rows, random_state=random_state)


def load_ptbxl_split(
    ptb_root: Path,
    *,
    sampling_rate: int = 100,
    test_fold: int = 10,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Load PTB-XL splits for training/evaluation.

    Returns:
        ((X_train, y_train, train_ids), (X_test, y_test, test_ids))
    """

    ptb_root = ptb_root.resolve()
    metadata = load_ptbxl_metadata(ptb_root)
    diag_map = load_diagnostic_map(ptb_root)
    metadata = attach_superclasses(metadata, diag_map)

    train_df = metadata[metadata["strat_fold"] != test_fold]
    test_df = metadata[metadata["strat_fold"] == test_fold]
    train_df = sample_dataframe(train_df, max_train, random_state=random_state)
    test_df = sample_dataframe(test_df, max_test, random_state=random_state)

    print(f"Loading {len(train_df)} training segments...")
    X_train = load_ptbxl_signals(train_df, ptb_root, sampling_rate=sampling_rate)
    print(f"Loading {len(test_df)} evaluation segments...")
    X_test = load_ptbxl_signals(test_df, ptb_root, sampling_rate=sampling_rate)

    y_train = encode_superclasses(train_df["superclasses"])
    y_test = encode_superclasses(test_df["superclasses"])

    return (
        (X_train, y_train, train_df.index.to_numpy()),
        (X_test, y_test, test_df.index.to_numpy()),
    )
