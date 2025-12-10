"""
Smooth adversarial perturbations implemented with TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .config import CLASS_TO_INDEX, ensure_lead_axis
from .masks import build_time_mask
from .tf_utils import tf_device_scope


@dataclass(frozen=True)
class SmoothAdvDefaults:
    eps_global_max: float = 0.5
    lambda_smooth: float = 10.0
    lambda_energy: float = 0.5
    steps: int = 200
    learning_rate: float = 0.01


SMOOTH_ADV_DEFAULTS = SmoothAdvDefaults()


def _build_time_mask_tf(
    num_samples: int,
    fs: float,
    center_time: float,
    window_seconds: Optional[float],
) -> tf.Tensor:
    mask_np = build_time_mask(
        num_samples,
        fs,
        center_time,
        window_seconds,
        use_hann=True,
    ).astype(np.float32)
    return tf.convert_to_tensor(mask_np.reshape(1, -1, 1))


def _prepare_targets(
    y_true: np.ndarray,
    *,
    target_class: Optional[str],
    target_mode: str,
    target_value: float,
) -> Tuple[tf.Tensor, tf.Tensor]:
    y_true_vec = np.asarray(y_true, dtype=np.float32).reshape(1, -1)
    y_true_tf = tf.convert_to_tensor(y_true_vec)
    if target_class is None:
        return y_true_tf, y_true_tf

    target = y_true_vec.copy()
    idx = CLASS_TO_INDEX[target_class]
    if target_mode == "suppress":
        target[0, idx] = 0.0
    else:
        target[0, idx] = target_value
    y_target_tf = tf.convert_to_tensor(target)
    return y_true_tf, y_target_tf


def smooth_adversarial_perturbation(
    x: np.ndarray,
    *,
    fs: float,
    config,
    model,
    y_true: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a smooth adversarial perturbation for a single sample.
    """

    if model is None:
        raise ValueError("smooth_adv requires a trained model instance.")
    if y_true is None:
        raise ValueError("smooth_adv requires the true label vector.")

    with tf_device_scope():
        x_np = ensure_lead_axis(np.asarray(x, dtype=np.float32))
        tensor_x = tf.convert_to_tensor(x_np[None, ...])  # (1, L, C)
        delta_param = tf.Variable(tf.zeros_like(tensor_x), trainable=True)

        defaults = SMOOTH_ADV_DEFAULTS
        extra = config.extra or {}
        eps_max = extra.get("eps_global_max", defaults.eps_global_max) * config.strength
        lambda_smooth = extra.get("lambda_smooth", defaults.lambda_smooth)
        lambda_energy = extra.get("lambda_energy", defaults.lambda_energy)
        steps = int(extra.get("steps", defaults.steps))
        lr = extra.get("lr", defaults.learning_rate)
        target_mode = extra.get("target_mode", "force")
        target_value = extra.get("target_value", 1.0)

        mask = _build_time_mask_tf(
            tensor_x.shape[1],
            fs,
            config.center_time,
            config.window_seconds,
        )

        y_true_tf, y_target_tf = _prepare_targets(
            y_true,
            target_class=config.target_class,
            target_mode=target_mode,
            target_value=target_value,
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        @tf.function
        def _train_step():
            with tf.GradientTape() as tape:
                delta = delta_param * mask
                norm = tf.norm(delta)
                if eps_max > 0:
                    delta = tf.cond(
                        norm > eps_max,
                        lambda: delta * (eps_max / (norm + 1e-8)),
                        lambda: delta,
                    )
                x_adv = tensor_x + delta
                logits = model(x_adv, training=False)
                if config.target_class is None:
                    cls_loss = -bce(y_true_tf, logits)
                else:
                    cls_loss = bce(y_target_tf, logits)
                smooth_loss = tf.reduce_mean(
                    tf.square(delta[:, 1:, :] - delta[:, :-1, :])
                )
                energy_loss = tf.reduce_mean(tf.square(delta))
                loss = cls_loss + lambda_smooth * smooth_loss + lambda_energy * energy_loss
            grads = tape.gradient(loss, [delta_param])
            optimizer.apply_gradients(zip(grads, [delta_param]))

        for _ in range(steps):
            _train_step()

        delta = delta_param * mask
        norm = tf.norm(delta)
        if eps_max > 0 and norm > eps_max:
            delta = delta * (eps_max / (norm + 1e-8))
        x_adv = tensor_x + delta
        return x_adv.numpy()[0], delta.numpy()[0]
