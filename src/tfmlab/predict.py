"""Forecast a CSV with an optional weight and KV configuration.

Each numeric column is decoded as an independent univariate series. The last
``context`` finite values are used; output rows carry the point forecast and
nine quantiles per step.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from .attention import install_kv_scopes
from .kv import make_scope_plans
from .runtime import load_model, predict, unload_model
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


def _install_kv(model, name: str) -> None:
    plans = make_scope_plans(name, len(model.transformer_stack.layers))
    for scope, plan in plans:
        for index, compressor in enumerate(plan):
            assignments = (
                {"seq": compressor, "var": compressor}
                if scope == "both"
                else {scope: compressor}
            )
            install_kv_scopes(model, [index], assignments)


def run(cfg: PredictConfig) -> pd.DataFrame:
    frame = pd.read_csv(cfg.input)
    series = series_from_frame(frame, cfg.context)
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, weight_specs([cfg.weights])[cfg.weights])
    if cfg.kv not in ("none", "off", ""):
        _install_kv(model, cfg.kv)

    outputs = []
    for name, values in series.items():
        pred = predict(model, values[None, :], cfg.horizon, 1)[0]
        outputs.append(format_forecast(name, pred))
    unload_model(model)

    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(cfg.out, index=False)
    print(f"wrote {cfg.out}: {len(series)} series x {cfg.horizon} steps", flush=True)
    return result
