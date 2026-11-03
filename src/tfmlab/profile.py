"""Where the time goes: quantized GEMM versus the full forward.

Weights already go through ``nn.QuantizedLinear`` -> ``mx.quantized_matmul``, so
the end-to-end result is not limited by a missing matmul kernel. This probe
measures the per-GEMM gain, the forward at a long context for each precision,
and MLX's fused attention kernel, so the missing speedup is explained by data.

Run through ``tfmq profile``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import mlx.core as mx
import numpy as np

from .attention import install_kv
from .kv import PassThroughKV, make_kv
from .runtime import load_model, measure_latency, measure_peak_memory, unload_model
from .weights import apply_weight_spec, weight_specs


@dataclasses.dataclass
class ProfileConfig:
    checkpoint: str = "models/timesfm-3.0"
    context: int = 15360
    batch: int = 8
    repeats: int = 3
    out: str = "results/profile.json"


def _bench(fn, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        mx.eval(fn())
        times.append(time.perf_counter() - start)
    return float(np.median(times) * 1000.0)


def gemm_rows(tokens: tuple[int, ...] = (482, 15424), dim: int = 1280) -> list[dict]:
    """fp32 versus quantized matmul at the shapes the model actually runs."""
    rng = np.random.default_rng(0)
    rows = []
    for count in tokens:
        x = mx.array(rng.standard_normal((count, dim), dtype=np.float32))
        w = mx.array(rng.standard_normal((dim, dim), dtype=np.float32))
        row = {"tokens": count, "dim": dim, "fp32_ms": _bench(lambda: x @ w.T)}
        for bits in (8, 4):
            wq, scales, biases = mx.quantize(w, 64, bits)
            row[f"int{bits}_ms"] = _bench(
                lambda: mx.quantized_matmul(
                    x, wq, scales=scales, biases=biases,
                    transpose=True, group_size=64, bits=bits,
                )
            )
        rows.append(row)
    return rows


def _e2e_runs() -> list[tuple[str, str | None, str | None]]:
    return [
        ("fp32", None, None),
        ("int8_g64", "int8_g64", None),
        ("int4_g64", "int4_g64", None),
        ("fp16", "fp16", None),
        ("sdpa", None, "sdpa"),
        ("affine4", None, "affine4"),
        ("affine4_sdpa", None, "affine4_sdpa"),
    ]


def _kv_for(name: str | None):
    if name is None:
        return None
    if name == "sdpa":
        kv = PassThroughKV()
        kv.fast = True
        return kv
    if name == "affine4_sdpa":
        kv = make_kv("affine4")
        kv.fast = True
        return kv
    return make_kv(name)


def run(cfg: ProfileConfig) -> dict:
    result: dict = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": dataclasses.asdict(cfg),
        "gemm": gemm_rows(),
        "e2e": [],
    }
    print(f"[profile] context {cfg.context}, batch {cfg.batch}", flush=True)
    for label, weight_name, kv_name in _e2e_runs():
        model = load_model(cfg.checkpoint)
        if weight_name is not None:
            apply_weight_spec(model, weight_specs([weight_name])[weight_name])
        kv = _kv_for(kv_name)
        if kv is not None:
            install_kv(model, kv)
        latency = measure_latency(
            model, cfg.batch, cfg.context, repeats=cfg.repeats
        )
        memory = measure_peak_memory(model, cfg.batch, cfg.context)
        row = {
            "name": label,
            "latency_ms_p50": latency["latency_ms_p50"],
            "series_per_s": latency["series_per_s"],
            "peak_mb": memory["peak_mb"],
        }
        result["e2e"].append(row)
        print(
            f"  {label:16s} {row['latency_ms_p50']:8.1f} ms  "
            f"{row['series_per_s']:5.1f} series/s  peak {row['peak_mb']:.0f} MB",
            flush=True,
        )
        unload_model(model)
        del model

    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {cfg.out}", flush=True)
    return result
