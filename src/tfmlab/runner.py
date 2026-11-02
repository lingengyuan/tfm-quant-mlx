"""Benchmark orchestration for weight quantization and KV compression."""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import time
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from . import metrics
from .attention import install_kv
from .data import SeriesSet, build_windows, load_ett, synthetic_set
from .kv import KV_NAMES, make_kv_plan
from .runtime import (
    env_info,
    load_model,
    measure_latency,
    measure_peak_memory,
    predict,
    unload_model,
)
from .weights import (
    WeightSpec,
    apply_weight_spec,
    logical_quant_bits,
    weight_nbytes,
    weight_specs,
)

PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "out": "results/quick.json",
        "datasets": ["ETTh1"],
        "synthetic": True,
        "contexts": [512, 2048, 15360],
        "windows": 2,
        "weights": ["fp32", "fp16", "int8_g64", "int4_g64"],
        "kv": ["affine8", "affine4", "affine4_l19fp16", "turbo3_mse", "turbo3_l19fp16", "qjl1"],
        "kv_contexts": [2048, 15360],
        "throughput_contexts": [512, 15360],
        "throughput_batches": [1, 8],
        "repeats": 2,
        "eval_batch": 16,
    },
    "full": {
        "out": "results/full.json",
        "datasets": ["ETTh1", "ETTh2", "ETTm1", "ETTm2"],
        "synthetic": True,
        "contexts": [512, 2048, 8192, 15360],
        "windows": 6,
        "weights": ["fp32", "fp16", "bf16", "int8_g64", "int4_g64", "int4_g32", "int4_g128"],
        "kv": [
            "affine8",
            "affine4",
            "affine4_l19fp16",
            "turbo3_mse",
            "turbo3_l19fp16",
            "turbo3_prod",
            "turbo2_mse",
            "qjl1",
        ],
        "kv_contexts": [2048, 8192, 15360],
        "throughput_contexts": [512, 2048, 15360],
        "throughput_batches": [1, 8, 32],
        "repeats": 3,
        "eval_batch": 32,
    },
}


@dataclasses.dataclass
class RunConfig:
    checkpoint: str = "models/timesfm-3.0"
    data_dir: str = "data/raw"
    datasets: tuple[str, ...] = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
    synthetic: bool = True
    contexts: tuple[int, ...] = (512, 2048, 8192, 15360)
    horizon: int = 64
    windows: int = 6
    weights: tuple[str, ...] = ("fp32", "fp16", "bf16", "int8_g64", "int4_g64", "int4_g32", "int4_g128")
    kv: tuple[str, ...] = (
        "affine8",
        "affine4",
        "affine4_l19fp16",
        "turbo3_mse",
        "turbo3_l19fp16",
        "turbo3_prod",
        "turbo2_mse",
        "qjl1",
    )
    # Weight specs to combine with the KV grid. Each pair gets its own row, so
    # weight-only and KV-only errors can be separated from the combined error.
    kv_weights: tuple[str, ...] = ("fp32",)
    kv_contexts: tuple[int, ...] = (2048, 8192, 15360)
    throughput_contexts: tuple[int, ...] = (512, 2048, 15360)
    throughput_batches: tuple[int, ...] = (1, 8, 32)
    repeats: int = 3
    eval_batch: int = 32
    mem_batch: int = 8
    out: str = "results/bench.json"

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - fields
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        clean = {}
        for key, value in data.items():
            if isinstance(value, list):
                value = tuple(value)
            clean[key] = value
        for key in ("weights", "kv"):
            if key in clean and clean[key] and not isinstance(clean[key], tuple):
                clean[key] = tuple(clean[key])
        return cls(**clean)


def config_from_preset(name: str, **overrides) -> RunConfig:
    if name not in PRESETS:
        raise ValueError(f"unknown preset: {name}")
    data = dict(PRESETS[name])
    data.update(overrides)
    return RunConfig.from_dict(data)


def _save(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _aggregate(entries: list[dict]) -> dict:
    """Element-wise mean of metric dictionaries."""
    if not entries:
        return {}
    keys = entries[0].keys()
    out = {}
    for key in keys:
        values = [e[key] for e in entries if key in e]
        if not values:
            continue
        if isinstance(values[0], list):
            out[key] = [float(np.mean([v[i] for v in values])) for i in range(len(values[0]))]
        elif isinstance(values[0], (int, float)):
            out[key] = float(np.mean(values))
    return out


@dataclasses.dataclass
class EvalData:
    contexts: np.ndarray
    targets: np.ndarray
    scales: np.ndarray


def _load_data(cfg: RunConfig, contexts: tuple[int, ...]) -> tuple[dict[str, SeriesSet], dict[tuple[str, int], EvalData]]:
    sets: dict[str, SeriesSet] = {}
    for name in cfg.datasets:
        sets[name] = load_ett(name, cfg.data_dir)
    if cfg.synthetic:
        sets["synthetic"] = synthetic_set()
    eval_data: dict[tuple[str, int], EvalData] = {}
    for name, series_set in sets.items():
        for context in contexts:
            windows = build_windows(series_set, context, cfg.horizon, cfg.windows)
            if windows is None:
                continue
            x, y, s, _ = windows
            eval_data[(name, context)] = EvalData(x, y, s)
    return sets, eval_data


def _accuracy(
    model,
    cfg: RunConfig,
    eval_data: dict[tuple[str, int], EvalData],
    contexts: tuple[int, ...],
    ref: dict[tuple[str, int], np.ndarray] | None,
    batch_size: int | None = None,
) -> tuple[dict, dict, dict[tuple[str, int], np.ndarray]]:
    per_dataset: dict[str, dict[str, dict]] = {}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for (name, context), data in eval_data.items():
        if context not in contexts:
            continue
        pred = predict(model, data.contexts, cfg.horizon, batch_size or cfg.eval_batch)
        predictions[(name, context)] = pred
        if ref is None:
            m = metrics.evaluate(pred, data.targets, data.scales).to_dict()
        else:
            m = metrics.compare(pred, ref[(name, context)], data.targets, data.scales)
        per_dataset.setdefault(name, {})[str(context)] = m
    macro: dict[str, dict] = {}
    for context in contexts:
        entries = [
            per_dataset[name][str(context)]
            for name in per_dataset
            if str(context) in per_dataset[name]
        ]
        if entries:
            macro[str(context)] = _aggregate(entries)
    return per_dataset, macro, predictions


def _throughput(model, cfg: RunConfig, contexts: tuple[int, ...], cap_batch: int | None = None) -> dict:
    out: dict[str, dict] = {}
    for context in contexts:
        out[str(context)] = {}
        for batch in cfg.throughput_batches:
            if cap_batch is not None and batch > cap_batch:
                continue
            out[str(context)][str(batch)] = measure_latency(
                model, batch, context, cfg.horizon, cfg.repeats
            )
    return out


def _memory(model, cfg: RunConfig, contexts: tuple[int, ...]) -> dict:
    return {
        str(context): measure_peak_memory(model, cfg.mem_batch, context, cfg.horizon)
        for context in contexts
    }


def _weight_run(
    cfg: RunConfig,
    spec: WeightSpec,
    eval_data: dict[tuple[str, int], EvalData],
    contexts: tuple[int, ...],
    ref: dict[tuple[str, int], np.ndarray] | None,
    ref_bytes: int | None,
) -> dict:
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, spec)
    nbytes = weight_nbytes(model)
    print(f"[weights] {spec.name}: {nbytes / 1e6:.0f} MB", flush=True)
    acc, macro, _ = _accuracy(model, cfg, eval_data, contexts, ref)
    result = {
        "name": spec.name,
        "kind": spec.kind,
        "spec": dataclasses.asdict(spec),
        "effective_bits": logical_quant_bits(spec),
        "weight_bytes": nbytes,
        "weight_ratio": (ref_bytes / nbytes) if ref_bytes else None,
        "accuracy": acc,
        "macro": macro,
        "throughput": _throughput(model, cfg, cfg.throughput_contexts),
        "memory": _memory(model, cfg, cfg.throughput_contexts),
    }
    unload_model(model)
    return result


def _kv_run(
    cfg: RunConfig,
    name: str,
    weight: str,
    weight_spec: WeightSpec,
    eval_data: dict[tuple[str, int], EvalData],
    ref: dict[tuple[str, int], np.ndarray],
    contexts: tuple[int, ...],
) -> dict:
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, weight_spec)
    plan = make_kv_plan(name, len(model.transformer_stack.layers))
    for index, compressor in enumerate(plan):
        install_kv(model, compressor, scope="both", layers=[index])

    # The first forward at the probe shape traces the compiled graph and records
    # every K/V tensor shape. Later calls at the same shape reuse the graph.
    probe_context = max(contexts)
    for compressor in plan:
        compressor.stats.reset()
    probe = np.sin(np.linspace(0.0, 40.0, probe_context, dtype=np.float32))
    probe_batch = np.stack([probe] * cfg.mem_batch)
    mx.eval(model.decode(mx.array(probe_batch)[:, None, :], cfg.horizon))
    logical = sum(compressor.logical_bytes() for compressor in plan)
    fp16 = sum(compressor.fp16_bytes() for compressor in plan)
    probe_calls = sum(len(compressor.stats.calls) for compressor in plan)
    values = sum(
        math.prod(shape) for compressor in plan for _, shape in compressor.stats.calls
    )
    mixed = len({compressor.bits_k for compressor in plan}) > 1 or len(
        {compressor.bits_v for compressor in plan}
    ) > 1
    bits = logical * 8.0 / values if values else 0.0
    bits_k = bits if mixed else plan[0].bits_k
    bits_v = bits if mixed else plan[0].bits_v

    acc, macro, _ = _accuracy(
        model, cfg, eval_data, contexts, ref, batch_size=16 if "turbo" in name else None
    )
    display = name if weight == "fp32" else f"{weight}+{name}"
    result = {
        "name": display,
        "weight": weight,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "bits_per_value": (bits_k + bits_v) / 2,
        "accuracy": acc,
        "macro": macro,
        "throughput": _throughput(
            model, cfg, contexts, cap_batch=8 if "turbo" in name else None
        ),
        "memory": _memory(model, cfg, contexts),
        "kv_logical_bytes": logical,
        "kv_fp16_bytes": fp16,
        "kv_reduction": (fp16 / logical) if logical else None,
        "kv_calls": probe_calls,
        "kv_probe": {"batch": cfg.mem_batch, "context": probe_context},
    }
    print(
        f"[kv] {display}: K={bits_k:.2f} V={bits_v:.2f} bits "
        f"KV {logical / 1e6:.0f} MB vs fp16 {fp16 / 1e6:.0f} MB",
        flush=True,
    )
    unload_model(model)
    return result


def run(cfg: RunConfig) -> dict:
    started = time.time()
    needed = tuple(sorted(set(cfg.contexts) | set(cfg.kv_contexts) | set(cfg.throughput_contexts)))
    sets, eval_data = _load_data(cfg, needed)
    unavailable = sorted({c for c in needed if not any(k[1] == c for k in eval_data)})
    if unavailable:
        print(f"[data] contexts without data: {unavailable}", flush=True)

    # Always run the fp32 reference, even if it is not in the weight list.
    weight_names = list(cfg.weights)
    if "fp32" not in weight_names:
        weight_names.insert(0, "fp32")
    specs = weight_specs(weight_names)

    results: dict[str, Any] = {
        "meta": {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_s": None,
            "config": dataclasses.asdict(cfg),
        },
        "env": {},
        "datasets": {
            name: {
                "channels": len(series_set),
                "length": int(min(len(s) for s in series_set.series)),
                "freq": series_set.freq,
                "windows": {
                    str(c): (
                        len(eval_data[(name, c)].contexts) // max(1, len(series_set))
                        if (name, c) in eval_data
                        else 0
                    )
                    for c in needed
                },
            }
            for name, series_set in sets.items()
        },
        "weight_runs": [],
        "kv_runs": [],
    }

    ref_preds: dict[tuple[str, int], np.ndarray] | None = None
    ref_bytes: int | None = None
    for name, spec in specs.items():
        is_ref = name == "fp32"
        run_ref = ref_preds if not is_ref else None
        if is_ref:
            # Set env from the first loaded model, and cache reference outputs.
            model = load_model(cfg.checkpoint)
            apply_weight_spec(model, spec)
            results["env"] = env_info(cfg.checkpoint, model)
            results["env"]["param_count"] = int(
                sum(p.size for _, p in tree_flatten(model.parameters()))
            )
            ref_bytes = weight_nbytes(model)
            ref_contexts = tuple(sorted(set(cfg.contexts) | set(cfg.kv_contexts)))
            acc, macro, ref_preds = _accuracy(model, cfg, eval_data, cfg.contexts, None)
            for key, data in eval_data.items():
                if key[1] in ref_contexts and key not in ref_preds:
                    ref_preds[key] = predict(model, data.contexts, cfg.horizon, cfg.eval_batch)
            result = {
                "name": name,
                "kind": spec.kind,
                "spec": dataclasses.asdict(spec),
                "effective_bits": logical_quant_bits(spec),
                "weight_bytes": ref_bytes,
                "weight_ratio": 1.0,
                "accuracy": acc,
                "macro": macro,
                "throughput": _throughput(model, cfg, cfg.throughput_contexts),
                "memory": _memory(model, cfg, cfg.throughput_contexts),
            }
            unload_model(model)
            del model
        else:
            try:
                result = _weight_run(cfg, spec, eval_data, cfg.contexts, ref_preds, ref_bytes)
            except Exception as error:  # keep the grid running; record the failure
                gc.collect()
                mx.clear_cache()
                result = {"name": name, "error": f"{type(error).__name__}: {error}"}
        results["weight_runs"].append(result)
        _save(results, cfg.out)
        if results["env"]:
            print(
                f"[weights] {name} done ({results['env']['param_count'] / 1e6:.0f}M params)",
                flush=True,
            )

    if ref_preds is None:
        raise RuntimeError("fp32 reference was not produced")

    kv_specs = weight_specs(list(dict.fromkeys(["fp32", *cfg.kv_weights])))

    for weight_name, weight_spec in kv_specs.items():
        for name in cfg.kv:
            if name == "none":
                continue
            if name not in KV_NAMES:
                raise ValueError(f"unknown KV mode: {name}")
            display = name if weight_name == "fp32" else f"{weight_name}+{name}"
            try:
                result = _kv_run(
                    cfg,
                    name,
                    weight_name,
                    weight_spec,
                    eval_data,
                    ref_preds,
                    cfg.kv_contexts,
                )
            except Exception as error:
                gc.collect()
                mx.clear_cache()
                result = {"name": display, "error": f"{type(error).__name__}: {error}"}
            results["kv_runs"].append(result)
            _save(results, cfg.out)

    results["meta"]["elapsed_s"] = round(time.time() - started, 1)
    _save(results, cfg.out)
    print(f"wrote {cfg.out} in {results['meta']['elapsed_s']}s", flush=True)
    return results
