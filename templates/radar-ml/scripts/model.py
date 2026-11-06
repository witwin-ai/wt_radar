"""The range regressor: a small multilayer perceptron in numpy.

``train.py`` fits it and ``evaluate.py`` scores it. numpy keeps training on
the CPU and out of the GPU queue, and lets the tests run it. For a bigger
model use torch: ``witwin.studio.datasets.open(path).split("train").torch()``
returns the same samples as a torch ``Dataset``.

Inputs and the target are standardized with the training split's mean and
standard deviation; the saved file keeps them, so ``predict`` takes raw
dataset values and returns metres.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

FEATURE = "profile"   # dataset field the model reads: dB per range bin
TARGET = "range_m"    # dataset field it predicts, in metres


def arrays(samples: Iterable[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """``(x, y)`` of a dataset split: one row of ``FEATURE`` and one ``TARGET`` per sample."""
    rows, targets = [], []
    for item in samples:
        rows.append(np.asarray(item[FEATURE], dtype=np.float64).reshape(-1))
        targets.append(float(item[TARGET]))
    if not rows:
        return np.zeros((0, 0)), np.zeros(0)
    return np.stack(rows), np.asarray(targets)


class Model:
    """ReLU layers, then one linear output; ``layers`` is ``[(W, b), ...]``."""

    def __init__(self, layers: list[tuple[np.ndarray, np.ndarray]], x_mean: np.ndarray,
                 x_std: np.ndarray, y_mean: float, y_std: float) -> None:
        self.layers = layers
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = float(y_mean), float(y_std)
        self._adam: list[list[np.ndarray]] | None = None
        self._t = 0

    @classmethod
    def new(cls, x: np.ndarray, y: np.ndarray, hidden: list[int], rng: np.random.Generator) -> "Model":
        """A model sized for ``x`` with He-initialized weights; scales come from ``x`` and ``y``."""
        sizes = [x.shape[1], *hidden, 1]
        layers = [(rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out)), np.zeros(n_out))
                  for n_in, n_out in zip(sizes[:-1], sizes[1:])]
        # A constant feature has std 0; 1 keeps it at 0 after centring instead of dividing by 0.
        x_std = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
        return cls(layers, x.mean(axis=0), x_std, y.mean(), max(float(y.std()), 1e-12))

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicted ``TARGET`` for each row of raw features."""
        out, _ = self._forward((x - self.x_mean) / self.x_std)
        return out[:, 0] * self.y_std + self.y_mean

    def step(self, x: np.ndarray, y: np.ndarray, lr: float) -> float:
        """One Adam step on mean squared error; returns the batch MSE in ``TARGET`` units squared."""
        out, activations = self._forward((x - self.x_mean) / self.x_std)
        error = out[:, 0] - (y - self.y_mean) / self.y_std
        grad = (2.0 / len(y)) * error[:, None]
        grads = []
        for i in range(len(self.layers) - 1, -1, -1):
            w, _ = self.layers[i]
            grads.append((activations[i].T @ grad, grad.sum(axis=0)))
            if i:
                grad = (grad @ w.T) * (activations[i] > 0)
        self._adam_update(grads[::-1], lr)
        return float(np.mean(error ** 2)) * self.y_std ** 2

    def save(self, path: Path, meta: Mapping[str, Any]) -> None:
        """Write weights, scales and ``meta`` (JSON) to one ``.npz`` file."""
        arrays_out: dict[str, Any] = {"x_mean": self.x_mean, "x_std": self.x_std,
                                      "y": np.array([self.y_mean, self.y_std]),
                                      "meta": np.array(json.dumps(dict(meta)))}
        for i, (w, b) in enumerate(self.layers):
            arrays_out[f"w{i}"], arrays_out[f"b{i}"] = w, b
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:  # a file handle: np.savez would add ".npz" to a str path
            np.savez(handle, **arrays_out)

    @classmethod
    def load(cls, path: Path) -> tuple["Model", dict[str, Any]]:
        """The model in ``path`` and its ``meta``."""
        with np.load(path, allow_pickle=False) as data:
            count = sum(1 for name in data.files if name.startswith("w"))
            layers = [(data[f"w{i}"], data[f"b{i}"]) for i in range(count)]
            y_mean, y_std = data["y"]
            model = cls(layers, data["x_mean"], data["x_std"], y_mean, y_std)
            return model, json.loads(str(data["meta"]))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [x]
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
            activations.append(x)
        return x, activations[:-1]

    # Adam with the usual betas (0.9, 0.999).
    def _adam_update(self, grads: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        params = [p for layer in self.layers for p in layer]
        flat = [g for pair in grads for g in pair]
        if self._adam is None:
            self._adam = [[np.zeros_like(p) for p in params], [np.zeros_like(p) for p in params]]
        self._t += 1
        m, v = self._adam
        for i, (p, g) in enumerate(zip(params, flat)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            m_hat, v_hat = m[i] / (1 - 0.9 ** self._t), v[i] / (1 - 0.999 ** self._t)
            p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
