"""Forecast a CSV with an optional weight and KV configuration.

Modes:

* default: every numeric column is an independent univariate series;
* ``multi``: all numeric columns are decoded together as variates;
* ``covariates``: the last numeric column is the target, the others are
  past-only covariates, and sin/cos calendar features are past-future.

Output rows carry the point forecast and nine quantiles per step.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .attention import install_kv_named
from .runtime import load_model, predict, predict_with_covariates, unload_model
from .weights import apply_weight_spec, weight_specs

TAUS = ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")


@dataclasses.dataclass
class PredictConfig:
    input: str
    out: str = "forecasts.csv"
    checkpoint: str = "models/timesfm-3.0"
    context: int = 2048
    horizon: int = 64
    weights: str = "fp32"
    kv: str = "none"
    multi: bool = False
    covariates: bool = False
    period: float = 24.0


def series_from_frame(frame: pd.DataFrame, context: int) -> dict[str, np.ndarray]:
    """Take the last ``context`` finite values of every numeric column."""
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("input has no numeric columns")
    series = {}
    for name in numeric.columns:
        values = numeric[name].to_numpy(dtype=np.float32)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        series[str(name)] = values[-context:]
    if not series:
        raise ValueError("no numeric column has at least two finite values")
    return series


def stacked_series(
    frame: pd.DataFrame, context: int
) -> tuple[list[str], np.ndarray, int]:
    """Rows with any missing value dropped, then stacked as variates.

    Dropping per column would shift different timestamps together, so the row
    mask is shared across all numeric columns.
    """
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("input has no numeric columns")
    numeric = numeric.dropna(axis=0, how="any")
    if len(numeric) < 2:
        raise ValueError("fewer than two complete rows across numeric columns")
    values = numeric.to_numpy(dtype=np.float32)[-context:]
    return [str(column) for column in numeric.columns], values.T, len(values)


def format_forecast(name: str, pred: np.ndarray) -> pd.DataFrame:
    """Long-form rows: series, step, point, q0.1 .. q0.9 (sorted per step)."""
    pred = np.sort(pred, axis=-1)
    frame = pd.DataFrame(
        {
            "series": name,
            "step": np.arange(1, pred.shape[0] + 1),
            "point": pred[:, 4],
        }
    )
    for index, tau in enumerate(TAUS):
        frame[f"q{tau}"] = pred[:, index]
    return frame


def run(cfg: PredictConfig) -> pd.DataFrame:
    if cfg.multi and cfg.covariates:
        raise ValueError("choose one of multi or covariates")
    frame = pd.read_csv(cfg.input)
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, weight_specs([cfg.weights])[cfg.weights])
    if cfg.kv not in ("none", "off", ""):
        install_kv_named(model, cfg.kv)

    if cfg.multi:
        names, contexts, _ = stacked_series(frame, cfg.context)
        pred = predict(model, contexts[None], cfg.horizon, 1)[0]
        outputs = [format_forecast(name, pred[i]) for i, name in enumerate(names)]
        series = {name: None for name in names}
    elif cfg.covariates:
        names, contexts, length = stacked_series(frame, cfg.context)
        if len(names) < 2:
            raise ValueError("covariates mode needs at least two numeric columns")
        t = np.arange(length + cfg.horizon)
        past_future = np.stack(
            [np.sin(2.0 * np.pi * t / cfg.period), np.cos(2.0 * np.pi * t / cfg.period)]
        ).astype(np.float32)
        pred = predict_with_covariates(
            model,
            contexts[-1][None],
            contexts[:-1][None],
            past_future[None],
            cfg.horizon,
            1,
        )[0]
        outputs = [format_forecast(names[-1], pred)]
        series = {names[-1]: None}
    else:
        series = series_from_frame(frame, cfg.context)
        outputs = [
            format_forecast(name, predict(model, values[None], cfg.horizon, 1)[0])
            for name, values in series.items()
        ]
    unload_model(model)

    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(cfg.out, index=False)
    print(f"wrote {cfg.out}: {len(series)} series x {cfg.horizon} steps", flush=True)
    return result
