"""
Train the baseline PTB-XL CNN and export the evaluation split for downstream experiments.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Tuple

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
VENV_SITE = ROOT_DIR / ".venv" / "Lib" / "site-packages"
if VENV_SITE.exists():
    sys.path.insert(0, str(VENV_SITE))

sys.path.append(str(ROOT_DIR))

import numpy as np
import tensorflow as tf

from perturbations.config import CLASS_NAMES
from perturbations.data_utils import load_ptbxl_split
from perturbations.tf_utils import configure_tensorflow_device, tf_device_scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PTB-XL baseline CNN.")
    parser.add_argument("--ptb-root", type=pathlib.Path, default=pathlib.Path("ptb-xl"))
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("models") / "ptb_cnn.keras",
        help="Path to save the trained model (.keras or .h5).",
    )
    parser.add_argument("--export-eval", type=pathlib.Path, default=pathlib.Path("data") / "ptb_eval.npz")
    parser.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    parser.add_argument("--test-fold", type=int, default=10)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="TensorFlow device to run on ('auto' prefers GPU when available).",
    )
    return parser.parse_args()


def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=5, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=5, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="sigmoid"),
        ]
    )
    return model


def main() -> None:
    args = parse_args()
    device = configure_tensorflow_device(args.device)
    ptb_root = args.ptb_root.resolve()
    if not ptb_root.exists():
        raise FileNotFoundError(f"PTB-XL root {ptb_root} does not exist.")

    (train_split, test_split) = load_ptbxl_split(
        ptb_root,
        sampling_rate=args.sampling_rate,
        test_fold=args.test_fold,
        max_train=args.max_train,
        max_test=args.max_test,
        random_state=args.random_state,
    )
    X_train, y_train, _ = train_split
    X_test, y_test, test_ids = test_split

    output_path = args.output
    with tf_device_scope(device):
        model = build_model(input_shape=X_train.shape[1:])
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        auc = tf.keras.metrics.AUC(
            multi_label=True,
            num_labels=len(CLASS_NAMES),
            name="auc",
        )
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy", auc])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True, monitor="val_auc", mode="max"
            )
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        final_val_auc = float(history.history["val_auc"][-1])

        if output_path.suffix.lower() not in {".keras", ".h5"}:
            output_path = output_path.with_suffix(".keras")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)

    print(f"Training complete. Final val metrics: {final_val_auc:.4f} (AUC)")

    print(f"Saved model to {output_path}")

    if args.export_eval:
        export_path = args.export_eval
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            export_path,
            X_test=X_test,
            y_test_enc=y_test,
            record_ids=test_ids,
            fs=float(args.sampling_rate),
        )
        print(f"Wrote evaluation split to {export_path}")


if __name__ == "__main__":
    main()
