"""Where the time goes: quantized GEMM versus the full forward.

Weights already go through ``nn.QuantizedLinear`` -> ``mx.quantized_matmul``, so
the end-to-end result is not limited by a missing matmul kernel. This probe
measures the per-GEMM gain, the forward at a long context for each precision,
and MLX's fused attention kernel, so the missing speedup is explained by data.

Run through ``tfmq profile``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from .attention import install_kv
from .kv import PassThroughKV, make_kv
from .runtime import (
    HORIZON,
    env_info,
    load_model,
    measure_latency,
    measure_peak_memory,
    synthetic_input,
    unload_model,
)
from .weights import apply_weight_spec, weight_specs


@dataclasses.dataclass
class ProfileConfig:
    checkpoint: str = "models/timesfm-3.0"
    context: int = 15360
    horizon: int = HORIZON
    batch: int = 8
    repeats: int = 3
    out: str = "results/profile.json"


def token_count(
    context: int, horizon: int, patch_len: int = 32, output_patch_len: int = 64
) -> int:
    """Patched sequence length that reaches the transformer at this context.

    Mirrors ``TimesFM3Mlx.decode``: left-pad to whole patches, add the forecast
    patches (with the rolling output window), then add the ``rolls - 1`` extra
    rows used by the output patch. Context 15,360 and horizon 64 give 482.
    """
    context += (-context) % patch_len
    patches = context // patch_len
    extract = min(2 * patch_len, output_patch_len)
    overlap = extract - patch_len
    forecast = max(math.ceil((horizon - overlap) / patch_len), 1)
    rolls = output_patch_len // patch_len
    return patches + forecast + rolls - 1


def _bench_repeated(
    fns: dict[str, Callable[[], object]], warmup: int, repeats: int
) -> dict[str, dict]:
    """Time several shapes/backends in one interleaved loop.

    Running one backend for all its repetitions lets thermal drift decide the
    comparison; rotating the order spreads drift across every candidate.
    """
    for fn in fns.values():
        for _ in range(warmup):
            mx.eval(fn())
    samples: dict[str, list[float]] = {name: [] for name in fns}
    names = list(fns)
    for index in range(repeats):
        order = names[index % len(names):] + names[: index % len(names)]
        for name in order:
            start = time.perf_counter()
            mx.eval(fns[name]())
            samples[name].append(float((time.perf_counter() - start) * 1000.0))
    return {
        name: {"median_ms": float(np.median(values)), "samples": values}
        for name, values in samples.items()
    }


def gemm_rows(tokens: tuple[int, ...], dim: int = 1280, repeats: int = 15) -> list[dict]:
    """fp32 versus quantized matmul at the shapes the model actually runs."""
    rng = np.random.default_rng(0)
    w = mx.array(rng.standard_normal((dim, dim), dtype=np.float32))
    quantized = {}
    for bits in (8, 4):
        quantized[bits] = mx.quantize(w, 64, bits)
    rows = []
    for count in tokens:
        x = mx.array(rng.standard_normal((count, dim), dtype=np.float32))
        fns = {"fp32": lambda: x @ w.T}
        for bits, (wq, scales, biases) in quantized.items():
            fns[f"int{bits}"] = (
                lambda bits=bits, wq=wq, scales=scales, biases=biases: mx.quantized_matmul(
                    x, wq, scales=scales, biases=biases,
                    transpose=True, group_size=64, bits=bits,
                )
            )
        timed = _bench_repeated(fns, warmup=2, repeats=repeats)
        row = {"tokens": count, "dim": dim, "repeats": repeats}
        for name, values in timed.items():
            row[f"{name}_ms"] = values["median_ms"]
            row[f"{name}_ms_samples"] = values["samples"]
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hash(path: str) -> dict:
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    digest = hashlib.sha256()
    total = 0
    for file in files:
        digest.update(str(file.relative_to(root)).encode())
        digest.update(_sha256_file(file).encode())
        total += file.stat().st_size
    return {
        "path": path,
        "files": len(files),
        "bytes": total,
        "sha256": digest.hexdigest(),
    }


def _git_state() -> dict:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
    }


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
    base = token_count(cfg.context, cfg.horizon)
    actual = base * cfg.batch
    # The default batch sees ``batch`` times the per-series token count; 15,424
    # is the long-context stress shape used for the earlier comparison.
    tokens = tuple(sorted({base, actual, 15424}))
    result: dict = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": dataclasses.asdict(cfg),
        "shapes": {"tokens_per_series": base, "tokens_actual": actual},
        "provenance": {
            "git": _git_state(),
            "checkpoint": _checkpoint_hash(cfg.checkpoint),
            "input": {
                "kind": "synthetic",
                "batch": cfg.batch,
                "context": cfg.context,
                "horizon": cfg.horizon,
                "sha256": hashlib.sha256(
                    np.array(synthetic_input(cfg.batch, cfg.context)).tobytes()
                ).hexdigest(),
            },
        },
        "gemm": gemm_rows(tokens),
        "e2e": [],
    }
    print(
        f"[profile] context {cfg.context}, batch {cfg.batch}, "
        f"tokens {base} per series / {actual} per call",
        flush=True,
    )
    for label, weight_name, kv_name in _e2e_runs():
        model = load_model(cfg.checkpoint)
        if "env" not in result["provenance"]:
            result["provenance"]["env"] = env_info(cfg.checkpoint, model)
        if weight_name is not None:
            apply_weight_spec(model, weight_specs([weight_name])[weight_name])
        kv = _kv_for(kv_name)
        if kv is not None:
            install_kv(model, kv)
        latency = measure_latency(
            model, cfg.batch, cfg.context, repeats=cfg.repeats, samples=True
        )
        memory = measure_peak_memory(model, cfg.batch, cfg.context)
        row = {
            "name": label,
            "latency_ms_p50": latency["latency_ms_p50"],
            "latency_ms_samples": latency["latency_ms_samples"],
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
