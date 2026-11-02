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
    forecaster = TimesFM3Forecaster.from_pretrained(checkpoint)
    return forecaster.model


def unload_model(model) -> None:
    del model
    gc.collect()
    mx.clear_cache()


def predict(model, contexts: np.ndarray, horizon: int, batch_size: int) -> np.ndarray:
    """Decode ``(N, context)`` series in chunks. Returns ``(N, horizon, 9)``."""
    outputs = []
    for start in range(0, len(contexts), batch_size):
        chunk = np.asarray(contexts[start : start + batch_size], dtype=np.float32)
        out = model.decode(mx.array(chunk)[:, None, :], horizon)
        mx.eval(out)
        outputs.append(np.array(out)[:, 0])
    return np.concatenate(outputs, axis=0)


def _input(batch: int, context: int) -> mx.array:
    base = np.sin(np.linspace(0.0, 40.0, context, dtype=np.float32))
    ramp = np.linspace(-1.0, 1.0, context, dtype=np.float32)
    rows = np.stack([base + 0.1 * ramp * (i % 5) for i in range(batch)])
    return mx.array(rows)[:, None, :]


def measure_latency(
    model,
    batch: int,
    context: int,
    horizon: int = HORIZON,
    repeats: int = 3,
    warmup: int = 2,
) -> dict:
    x = _input(batch, context)
    for _ in range(warmup):
        mx.eval(model.decode(x, horizon))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(model.decode(x, horizon))
        times.append(time.perf_counter() - start)
    times = np.asarray(times)
    return {
        "batch": batch,
        "context": context,
        "latency_ms_p50": float(np.median(times) * 1000.0),
        "latency_ms_p95": float(np.percentile(times, 95) * 1000.0),
        "series_per_s": float(batch / np.median(times)),
    }


def measure_peak_memory(model, batch: int, context: int, horizon: int = HORIZON) -> dict:
    x = _input(batch, context)
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
