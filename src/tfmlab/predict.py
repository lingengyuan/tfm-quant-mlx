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

from .attention import install_kv_named, set_attention_backend
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
    attention: str = "manual"
    batch: int = 8
    multi: bool = False
    covariates: bool = False
    period: float = 24.0
    target: str | None = None
    future_covariates: tuple[str, ...] = ()
    time: str | None = None


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
    """Rows with any missing or non-finite value dropped, then stacked.

    Dropping per column would shift different timestamps together, so the row
    mask is shared across all numeric columns. The mask is applied after the
    float32 cast, because large float64 values can overflow to ``inf``.
    """
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("input has no numeric columns")
    names = [str(column) for column in numeric.columns]
    values = numeric.to_numpy(dtype=np.float32)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 2:
        raise ValueError("fewer than two complete finite rows across numeric columns")
    values = values[-context:]
    return names, values.T, len(values)


def format_forecast(name: str, pred: np.ndarray) -> pd.DataFrame:
    """Long-form rows: series, step, point, q0.1 .. q0.9 (sorted per step)."""
    if not np.isfinite(pred).all():
        raise ValueError(f"non-finite forecast for {name}")
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


def covariate_arrays(
    frame: pd.DataFrame,
    context: int,
    horizon: int,
    target_name: str | None = None,
    future_names: tuple[str, ...] = (),
    period: float = 24.0,
    time_name: str | None = None,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray | None]]:
    """Target history, past-only and past-future covariates, plus masks.

    Missing values stay in place and are masked, so gaps do not compress the
    time axis. With explicit future covariates the last ``horizon`` rows hold
    their forecast values and the target history ends before them.
    """
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError("input has no numeric columns")
    # An integer timestamp column is numeric but never a series.
    if time_name is not None and time_name in numeric.columns:
        numeric = numeric.drop(columns=[time_name])
    times = None
    if time_name is not None:
        if time_name not in frame.columns:
            raise ValueError(f"time column not found: {time_name}")
        stamps = pd.to_datetime(frame[time_name])
        order = stamps.argsort()
        numeric = numeric.iloc[order].reset_index(drop=True)
        times = stamps.iloc[order].reset_index(drop=True)
    if target_name is not None:
        if target_name not in numeric.columns:
            raise ValueError(f"target column not found: {target_name}")
        target = str(target_name)
    else:
        target = str(numeric.columns[-1])
    missing_future = [name for name in future_names if name not in numeric.columns]
    if missing_future:
        raise ValueError(f"future covariates not found: {missing_future}")

    rows = len(numeric)
    hist_end = rows - horizon if future_names else rows
    hist_start = hist_end - context
    if hist_start < 0:
        raise ValueError(
            f"need {context + (horizon if future_names else 0)} rows, found {rows}"
        )

    def _values(columns: list[str], start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        block = numeric[columns].iloc[start:end].to_numpy(dtype=np.float32)
        mask = ~np.isfinite(block)
        return np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0), mask

    target_block, target_mask = _values([target], hist_start, hist_end)
    target_values = target_block[:, 0]
    target_mask = target_mask[:, 0]
    past_names = [
        str(column)
        for column in numeric.columns
        if str(column) != target and str(column) not in future_names
    ]
    if past_names:
        past_block, past_mask = _values(past_names, hist_start, hist_end)
        past_only = past_block.T
        past_only_mask = past_mask.T
    else:
        past_only = np.zeros((0, context), dtype=np.float32)
        past_only_mask = np.zeros((0, context), dtype=bool)

    length = context + horizon
    if future_names:
        future_block, future_mask = _values(list(future_names), hist_start, rows)
        past_future = future_block.T
        past_future_mask = future_mask.T
    else:
        past_future = np.zeros((0, length), dtype=np.float32)
        past_future_mask = np.zeros((0, length), dtype=bool)

    index = np.arange(length, dtype=np.float64)
    if times is not None:
        # With explicit future rows their timestamps cover the whole span.
        span_end = rows if future_names else hist_end
        stamps = times.iloc[hist_start:span_end]
        hours = (stamps - stamps.iloc[0]).dt.total_seconds().to_numpy() / 3600.0
        if future_names:
            index = hours
        else:
            step = (
                float(np.median(np.diff(np.sort(np.unique(hours)))))
                if len(hours) > 1
                else 1.0
            )
            index = np.concatenate(
                [hours, hours[-1] + step * np.arange(1, horizon + 1)]
            )
    calendar = np.stack(
        [
            np.sin(2.0 * np.pi * index / period),
            np.cos(2.0 * np.pi * index / period),
        ]
    ).astype(np.float32)
    past_future = np.concatenate([past_future, calendar], axis=0)
    past_future_mask = np.concatenate(
        [past_future_mask, np.zeros_like(calendar, dtype=bool)], axis=0
    )
    masks = {
        "target": target_mask,
        "past_only": past_only_mask,
        "past_future": past_future_mask,
    }
    return target, target_values, past_only, past_future, masks


def _batched_forecasts(
    model, series: dict[str, np.ndarray], horizon: int, batch: int
) -> list[pd.DataFrame]:
    """Group equal-length columns so one decode covers several series.

    ``predict`` needs a rectangular batch, so lengths are grouped; the output
    order still follows the input column order.
    """
    groups: dict[int, list[str]] = {}
    for name, values in series.items():
        groups.setdefault(len(values), []).append(name)
    forecasts: dict[str, np.ndarray] = {}
    for names in groups.values():
        pred = predict(model, np.stack([series[name] for name in names]), horizon, batch)
        forecasts.update({name: pred[index] for index, name in enumerate(names)})
    return [format_forecast(name, forecasts[name]) for name in series]


def _validate(cfg: PredictConfig) -> None:
    if cfg.context < 1:
        raise ValueError("context must be positive")
    if cfg.horizon < 1:
        raise ValueError("horizon must be positive")
    if cfg.batch < 1:
        raise ValueError("batch must be positive")
    if cfg.covariates and cfg.period <= 0:
        raise ValueError("period must be positive in covariates mode")
    if (cfg.target or cfg.future_covariates or cfg.time) and not cfg.covariates:
        raise ValueError("target, future-covariates, and time need --covariates")
    if cfg.attention not in ("manual", "sdpa"):
        raise ValueError(f"unknown attention backend: {cfg.attention}")


def run(cfg: PredictConfig) -> pd.DataFrame:
    if cfg.multi and cfg.covariates:
        raise ValueError("choose one of multi or covariates")
    _validate(cfg)
    frame = pd.read_csv(cfg.input)
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, weight_specs([cfg.weights])[cfg.weights])
    if cfg.kv not in ("none", "off", ""):
        install_kv_named(model, cfg.kv)
    set_attention_backend(model, cfg.attention)

    if cfg.multi:
        names, contexts, _ = stacked_series(frame, cfg.context)
        pred = predict(model, contexts[None], cfg.horizon, 1)[0]
        outputs = [format_forecast(name, pred[i]) for i, name in enumerate(names)]
        series = {name: None for name in names}
    elif cfg.covariates:
        target, values, past_only, past_future, masks = covariate_arrays(
            frame,
            cfg.context,
            cfg.horizon,
            cfg.target,
            tuple(cfg.future_covariates),
            cfg.period,
            cfg.time,
        )
        pred = predict_with_covariates(
            model,
            values[None],
            past_only[None] if past_only.size else None,
            past_future[None] if past_future.size else None,
            cfg.horizon,
            1,
            target_mask=masks["target"][None],
            past_only_mask=masks["past_only"][None] if past_only.size else None,
            past_future_mask=masks["past_future"][None] if past_future.size else None,
        )[0]
        outputs = [format_forecast(target, pred)]
        series = {target: None}
    else:
        series = series_from_frame(frame, cfg.context)
        outputs = _batched_forecasts(model, series, cfg.horizon, cfg.batch)
    unload_model(model)

    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(cfg.out, index=False)
    print(f"wrote {cfg.out}: {len(series)} series x {cfg.horizon} steps", flush=True)
    return result
