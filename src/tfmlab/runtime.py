"""Model loading, batched decoding, timing and memory probes."""

from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
import time

import mlx.core as mx
import numpy as np

from .attention import patch_class
from .kv import KVCompressor

HORIZON = 64


def load_model(checkpoint: str):
    from timesfm3.mlx import TimesFM3Forecaster

    patch_class()
    from .convert import is_converted, load_quantized

    if is_converted(checkpoint):
        return load_quantized(checkpoint)
    forecaster = TimesFM3Forecaster.from_pretrained(checkpoint)
    return forecaster.model


def unload_model(model) -> None:
    del model
    gc.collect()
    mx.clear_cache()


def predict(model, contexts: np.ndarray, horizon: int, batch_size: int) -> np.ndarray:
    """Decode ``(N, context)`` or ``(N, variates, context)`` series in chunks.

    Returns ``(N, horizon, 9)`` or ``(N, variates, horizon, 9)`` to match the
    input shape.
    """
    outputs = []
    multi = contexts.ndim == 3
    for start in range(0, len(contexts), batch_size):
        chunk = np.asarray(contexts[start : start + batch_size], dtype=np.float32)
        x = mx.array(chunk)
        if not multi:
            x = x[:, None, :]
        out = model.decode(x, horizon)
        mx.eval(out)
        arr = np.array(out)
        outputs.append(arr if multi else arr[:, 0])
    return np.concatenate(outputs, axis=0)


def predict_with_covariates(
    model,
    contexts: np.ndarray,
    past_only: np.ndarray | None,
    past_future: np.ndarray | None,
    horizon: int,
    batch_size: int,
    target_mask: np.ndarray | None = None,
    past_only_mask: np.ndarray | None = None,
    past_future_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Decode one target variate with past-only and past-future covariates.

    Masks follow the model convention: ``True`` marks a missing value.
    Returns ``(N, horizon, 9)`` for the target only.
    """
    outputs = []
    for start in range(0, len(contexts), batch_size):
        stop = start + batch_size
        target = mx.array(np.asarray(contexts[start:stop], dtype=np.float32))[:, None, :]

        def _array(values, mask):
            if values is None:
                return None, None
            chunk = mx.array(np.asarray(values[start:stop], dtype=np.float32))
            return chunk, (mx.array(mask[start:stop]) if mask is not None else None)

        past_only_chunk, past_only_mask_chunk = _array(past_only, past_only_mask)
        past_future_chunk, past_future_mask_chunk = _array(past_future, past_future_mask)
        out = model.decode(
            target,
            horizon,
            past_only_covariates=past_only_chunk,
            past_future_covariates=past_future_chunk,
            target_mask=(
                mx.array(target_mask[start:stop])[:, None, :]
                if target_mask is not None
                else None
            ),
            past_only_mask=past_only_mask_chunk,
            past_future_mask=past_future_mask_chunk,
        )
        mx.eval(out)
        outputs.append(np.array(out)[:, 0])
    return np.concatenate(outputs, axis=0)


def _input(batch: int, context: int, variates: int = 1) -> mx.array:
    base = np.sin(np.linspace(0.0, 40.0, context, dtype=np.float32))
    ramp = np.linspace(-1.0, 1.0, context, dtype=np.float32)
    rows = np.stack([base + 0.1 * ramp * (i % 5) for i in range(batch)])
    if variates > 1:
        rows = np.stack([rows + 0.05 * j for j in range(variates)], axis=1)
        return mx.array(rows)
    return mx.array(rows)[:, None, :]


def measure_latency(
    model,
    batch: int,
    context: int,
    horizon: int = HORIZON,
    repeats: int = 3,
    warmup: int = 2,
    variates: int = 1,
    samples: bool = False,
) -> dict:
    x = _input(batch, context, variates)
    for _ in range(warmup):
        mx.eval(model.decode(x, horizon))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(model.decode(x, horizon))
        times.append(time.perf_counter() - start)
    times = np.asarray(times)
    out = {
        "batch": batch,
        "context": context,
        "variates": variates,
        "latency_ms_p50": float(np.median(times) * 1000.0),
        "latency_ms_p95": float(np.percentile(times, 95) * 1000.0),
        "series_per_s": float(batch * variates / np.median(times)),
    }
    if samples:
        out["latency_ms_samples"] = [float(t * 1000.0) for t in times]
    return out


def synthetic_input(batch: int, context: int, variates: int = 1) -> mx.array:
    """Deterministic input used by latency and memory probes."""
    return _input(batch, context, variates)


def measure_peak_memory(
    model, batch: int, context: int, horizon: int = HORIZON, variates: int = 1
) -> dict:
    x = _input(batch, context, variates)
    mx.eval(model.decode(x, horizon))
    gc.collect()
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    mx.eval(model.decode(x, horizon))
    peak = mx.get_peak_memory()
    return {
        "batch": batch,
        "context": context,
        "variates": variates,
        "peak_mb": peak / 1e6,
        "resident_mb": base / 1e6,
        "activation_mb": (peak - base) / 1e6,
    }


def env_info(checkpoint: str, model) -> dict:
    from timesfm3.mlx.configs import TimesFM3MlxConfig

    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = "unknown"
    try:
        timesfm_version = importlib.metadata.version("timesfm")
    except importlib.metadata.PackageNotFoundError:
        timesfm_version = "unknown"
    cfg = model.config if isinstance(model.config, TimesFM3MlxConfig) else None
    return {
        # Local hostnames identify people; set TFMQ_HOST to record a real one.
        "host": os.environ.get("TFMQ_HOST", "local"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "mlx": mlx_version,
        "timesfm": timesfm_version,
        "checkpoint": checkpoint,
        "architecture": {
            "layers": getattr(cfg, "num_layers", None),
            "model_dims": getattr(cfg, "model_dims", None),
            "heads": getattr(cfg, "num_heads", None),
            "head_dim": getattr(cfg, "head_dim", None),
            "input_patch_len": getattr(cfg, "input_patch_len", None),
            "quantiles": list(getattr(cfg, "quantiles", ())),
            "max_context": 15360,
        },
    }


def kv_bytes_for_shape(compressor: KVCompressor, shape: tuple[int, ...], part: str) -> int:
    import math

    return int(round(math.prod(shape) * compressor.bits_per_value(part) / 8.0))
