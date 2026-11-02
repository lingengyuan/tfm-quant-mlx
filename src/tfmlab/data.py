"""Evaluation data.

Two sources:

* ETT-small (ETTh1/ETTh2/ETTm1/ETTm2), downloaded from the ETDataset repository.
  Seven real channels per table.
* A deterministic synthetic suite with eight regimes. It runs without network
  access and reaches the 15,360-point context used for the long-context tests.

Every series is evaluated with rolling-origin windows: the last ``context``
points predict the next ``horizon`` points. Windows are shared across channels
so a batch covers the same time span.
"""

from __future__ import annotations

import dataclasses
import os
import urllib.request

import numpy as np
import pandas as pd

ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
ETT_NAMES = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")

SYNTHETIC_LENGTH = 24_000


@dataclasses.dataclass(frozen=True)
class SeriesSet:
    """A named group of one-dimensional series."""

    name: str
    series: tuple[np.ndarray, ...]
    freq: str = "unknown"

    def __len__(self) -> int:
        return len(self.series)


def load_ett(name: str, cache_dir: str = "data/raw") -> SeriesSet:
    """Load an ETT table as one series per channel."""
    if name not in ETT_NAMES:
        raise ValueError(f"unknown ETT dataset: {name}")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.csv")
    if not os.path.exists(path):
        urllib.request.urlretrieve(ETT_URL.format(name=name), path)
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame.columns if c.lower() == "date"])
    values = frame.to_numpy(dtype=np.float32)
    freq = "15min" if name.startswith("ETTm") else "1h"
    return SeriesSet(name, tuple(values[:, i] for i in range(values.shape[1])), freq)


def _t(index: np.ndarray, period: float) -> np.ndarray:
    return 2.0 * np.pi * index / period


def _synthetic_series(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    if kind == "trend_season":
        y = 0.003 * idx + 2.0 * np.sin(_t(idx, 24)) + 0.7 * np.sin(_t(idx, 168))
        y += rng.normal(0.0, 0.25, n)
    elif kind == "multi_season":
        y = np.sin(_t(idx, 12)) + 0.8 * np.sin(_t(idx, 96) + 0.7)
        y += 0.5 * np.sin(_t(idx, 672) + 1.3) + rng.normal(0.0, 0.15, n)
    elif kind == "ar1":
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = 0.92 * y[i - 1] + rng.normal(0.0, 1.0)
    elif kind == "random_walk":
        y = np.cumsum(rng.normal(0.0, 0.35, n))
    elif kind == "regime":
        y = np.empty(n)
        level, slope = 0.0, 0.0
        for i in range(n):
            if i % 4000 == 0:
                level += rng.normal(0.0, 3.0)
                slope = rng.normal(0.0, 0.002)
            y[i] = level + slope * (i % 4000) + rng.normal(0.0, 0.4)
    elif kind == "spiky":
        y = np.sin(_t(idx, 48)) + rng.normal(0.0, 0.2, n)
        spikes = rng.random(n) < 0.01
        y[spikes] += rng.normal(0.0, 6.0, int(spikes.sum()))
    elif kind == "hetero":
        noise = rng.normal(0.0, 1.0, n) * (0.2 + 1.8 * (0.5 + 0.5 * np.sin(_t(idx, 3000))))
        y = np.sin(_t(idx, 24)) + noise
    elif kind == "square":
        y = np.sign(np.sin(_t(idx, 96))) + 0.1 * rng.normal(0.0, 1.0, n)
    else:
        raise ValueError(f"unknown synthetic kind: {kind}")
    return np.asarray(y, dtype=np.float32)


def synthetic_set(name: str = "synthetic", length: int = SYNTHETIC_LENGTH) -> SeriesSet:
    rng = np.random.default_rng(20261102)
    kinds = ("trend_season", "multi_season", "ar1", "random_walk", "regime", "spiky", "hetero", "square")
    return SeriesSet(name, tuple(_synthetic_series(rng, k, length) for k in kinds), freq="step")


def build_windows(
    series_set: SeriesSet,
    context: int,
    horizon: int,
    num_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
    """Return (contexts, targets, scales, channel_ids) for one dataset.

    ``scales`` is each window's context standard deviation, floored at 1e-6, and
    is used to express forecast deviations in native amplitude units.
    """
    min_len = context + horizon
    usable = [i for i, s in enumerate(series_set.series) if len(s) >= min_len]
    if not usable:
        return None

    length = min(len(series_set.series[i]) for i in usable)
    if length < min_len:
        return None
    first_end = context
    last_end = length - horizon
    count = max(1, min(num_windows, (last_end - first_end) // max(1, horizon) + 1))
    ends = np.linspace(first_end, last_end, count).astype(int)

    contexts, targets, scales, channels = [], [], [], []
    for end in ends:
        for ch in usable:
            s = series_set.series[ch]
            ctx = s[end - context : end]
            tgt = s[end : end + horizon]
            if not np.isfinite(ctx).all() or not np.isfinite(tgt).all():
                continue
            contexts.append(ctx)
            targets.append(tgt)
            scales.append(max(float(np.std(ctx)), 1e-6))
            channels.append(ch)
    if not contexts:
        return None
    return (
        np.stack(contexts).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.asarray(scales, dtype=np.float32),
        channels,
    )


def build_multi_windows(
    series_set: SeriesSet,
    context: int,
    horizon: int,
    num_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
    """Keep variates of the same time span in one window.

    Returns ``(contexts, targets, scales, channels)`` with shapes
    ``(W, V, context)``, ``(W, V, horizon)`` and ``(W, V)``. One decoder call
    then mixes all variates through variate attention.
    """
    min_len = context + horizon
    usable = [i for i, s in enumerate(series_set.series) if len(s) >= min_len]
    if len(usable) < 2:
        return None
    length = min(len(series_set.series[i]) for i in usable)
    first_end = context
    last_end = length - horizon
    count = max(1, min(num_windows, (last_end - first_end) // max(1, horizon) + 1))
    ends = np.linspace(first_end, last_end, count).astype(int)

    contexts, targets, scales = [], [], []
    for end in ends:
        ctx = np.stack([series_set.series[ch][end - context : end] for ch in usable])
        tgt = np.stack([series_set.series[ch][end : end + horizon] for ch in usable])
        if not np.isfinite(ctx).all() or not np.isfinite(tgt).all():
            continue
        contexts.append(ctx)
        targets.append(tgt)
        scales.append(np.maximum(np.std(ctx, axis=1), 1e-6))
    if not contexts:
        return None
    return (
        np.stack(contexts).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.stack(scales).astype(np.float32),
        usable,
    )


def build_covariate_windows(
    series_set: SeriesSet,
    context: int,
    horizon: int,
    num_windows: int,
    target_channel: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Target is one channel; the others are past-only covariates.

    Two sin/cos calendar features over the window's absolute index serve as
    past-future covariates. Returns ``(contexts, targets, scales, past_only,
    past_future)`` with shapes ``(N, C)``, ``(N, H)``, ``(N,)``, ``(N, V, C)``
    and ``(N, 2, C + H)``.
    """
    min_len = context + horizon
    usable = [i for i, s in enumerate(series_set.series) if len(s) >= min_len]
    if len(usable) < 2:
        return None
    target = usable[target_channel]
    covariates = [i for i in usable if i != target]
    length = min(len(series_set.series[i]) for i in usable)
    first_end = context
    last_end = length - horizon
    count = max(1, min(num_windows, (last_end - first_end) // max(1, horizon) + 1))
    ends = np.linspace(first_end, last_end, count).astype(int)
    period = {"1h": 24.0, "15min": 96.0}.get(series_set.freq, 24.0)

    contexts, targets, scales, past_only, past_future = [], [], [], [], []
    for end in ends:
        ctx = series_set.series[target][end - context : end]
        tgt = series_set.series[target][end : end + horizon]
        if not np.isfinite(ctx).all() or not np.isfinite(tgt).all():
            continue
        contexts.append(ctx)
        targets.append(tgt)
        scales.append(max(float(np.std(ctx)), 1e-6))
        past_only.append(np.stack([series_set.series[ch][end - context : end] for ch in covariates]))
        t = np.arange(end - context, end + horizon)
        past_future.append(
            np.stack([np.sin(2.0 * np.pi * t / period), np.cos(2.0 * np.pi * t / period)])
        )
    if not contexts:
        return None
    return (
        np.stack(contexts).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.asarray(scales, dtype=np.float32),
        np.stack(past_only).astype(np.float32),
        np.stack(past_future).astype(np.float32),
    )


def eval_windows(
    data_dir: str,
    datasets: tuple[str, ...],
    synthetic: bool,
    context: int,
    horizon: int,
    windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack windows from the named ETT tables plus the synthetic suite."""
    sets = [load_ett(name, data_dir) for name in datasets]
    if synthetic:
        sets.append(synthetic_set())
    frames = []
    for series_set in sets:
        built = build_windows(series_set, context, horizon, windows)
        if built is not None:
            frames.append(built)
    if not frames:
        raise ValueError("no evaluation windows")
    return tuple(np.concatenate([f[i] for f in frames]) for i in range(3))
