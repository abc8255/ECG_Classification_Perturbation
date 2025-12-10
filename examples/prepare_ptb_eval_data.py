"""
Prepare an evaluation NPZ (X_test, y_test_enc) straight from the PTB-XL files.

Example:
    python examples/prepare_ptb_eval_data.py --ptb-root ptb-xl --output data/ptb_eval.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
VENV_SITE = ROOT_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

sys.path.append(str(ROOT_DIR))

import numpy as np

from perturbations.data_utils import (
    attach_superclasses,
    encode_superclasses,
    load_diagnostic_map,
    load_ptbxl_metadata,
    load_ptbxl_signals,
    sample_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PTB-XL evaluation subset to NPZ.")
    parser.add_argument("--ptb-root", type=pathlib.Path, default=pathlib.Path("ptb-xl"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data") / "ptb_eval.npz")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--test-fold", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ptb_root = args.ptb_root.resolve()
    if not ptb_root.exists():
        raise FileNotFoundError(f"PTB-XL root {ptb_root} does not exist.")

    print(f"Loading metadata from {ptb_root} ...")
    metadata = load_ptbxl_metadata(ptb_root)
    diag_map = load_diagnostic_map(ptb_root)
    metadata = attach_superclasses(metadata, diag_map)

    eval_df = metadata[metadata["strat_fold"] == args.test_fold]
    eval_df = sample_dataframe(eval_df, args.max_samples, random_state=args.random_state)
    print(f"Preparing {len(eval_df)} evaluation samples from fold {args.test_fold}.")
    X_test = load_ptbxl_signals(eval_df, ptb_root, sampling_rate=args.sampling_rate)
    y_test = encode_superclasses(eval_df["superclasses"])
    record_ids = eval_df.index.to_numpy()

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X_test=X_test,
        y_test_enc=y_test,
        record_ids=record_ids,
        fs=float(args.sampling_rate),
    )
    print(f"Wrote evaluation subset to {output_path}")


if __name__ == "__main__":
    main()
