"""
TensorFlow device helpers for optional GPU execution.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Iterator, Optional

import tensorflow as tf

_DEVICE_OVERRIDE: Optional[str] = None
_MEMORY_GROWTH_CONFIGURED = False


def _enable_memory_growth() -> None:
    """
    Configure TensorFlow to allocate GPU memory lazily when possible.
    """

    global _MEMORY_GROWTH_CONFIGURED
    if _MEMORY_GROWTH_CONFIGURED:
        return
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:  # Happens if GPUs are already initialized.
            logging.warning("Unable to enable memory growth for %s: %s", gpu, exc)
    _MEMORY_GROWTH_CONFIGURED = True


def resolve_tf_device(preference: Optional[str]) -> str:
    """
    Convert a user preference into a TensorFlow device string.
    """

    pref = (preference or "auto").lower()
    if pref == "cpu":
        return "/CPU:0"
    if pref == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("GPU requested but TensorFlow did not detect one.")
        return "/GPU:0"
    if pref in {"auto", "default"}:
        return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    if pref.startswith("/"):
        return pref
    raise ValueError(f"Unrecognized TensorFlow device preference '{preference}'.")


def configure_tensorflow_device(
    preference: Optional[str] = None,
    *,
    enable_memory_growth: bool = True,
    log: bool = True,
) -> str:
    """
    Configure TensorFlow to run on the requested device and return it.
    """

    global _DEVICE_OVERRIDE
    device = resolve_tf_device(preference)
    if enable_memory_growth and device.startswith("/GPU"):
        _enable_memory_growth()
    _DEVICE_OVERRIDE = device
    if log:
        logging.info("Using TensorFlow device %s", device)
    return device


def current_tf_device() -> str:
    """
    Return the current TensorFlow device, auto-detecting if needed.
    """

    global _DEVICE_OVERRIDE
    if _DEVICE_OVERRIDE is None:
        configure_tensorflow_device("auto", log=False)
    return _DEVICE_OVERRIDE or "/CPU:0"


@contextlib.contextmanager
def tf_device_scope(device: Optional[str] = None) -> Iterator[None]:
    """
    Context manager that pins subsequent TensorFlow ops to a device.
    """

    with tf.device(device or current_tf_device()):
        yield


__all__ = [
    "configure_tensorflow_device",
    "current_tf_device",
    "resolve_tf_device",
    "tf_device_scope",
]
