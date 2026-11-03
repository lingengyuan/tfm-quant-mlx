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
from .attention import install_kv_scopes, set_attention_backend
from .data import SeriesSet, build_multi_windows, build_windows, load_dataset, synthetic_set
from .kv import KV_NAMES, make_scope_plans
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
    # Group each time window's channels into one decode call, so variate
    # attention runs. Deltas stay relative to fp32 either way.
    multi: bool = False
    # 0 keeps every channel; Jena/Electricity can carry hundreds.
    max_channels: int = 0
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
    # Which attention blocks to compress. Multivariate runs need seq/var split
    # to attribute the error between the two.
    kv_scopes: tuple[str, ...] = ("both",)
    kv_contexts: tuple[int, ...] = (2048, 8192, 15360)
    # "manual" keeps the explicit matmul-softmax path; "sdpa" uses the fused
    # kernel. Not numerically identical with KV compression, so keep it opt-in.
    attention: str = "manual"
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


def _weight_order(names) -> list[str]:
    return ["fp32"] + [name for name in names if name != "fp32"]


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
        sets[name] = load_dataset(name, cfg.data_dir, cfg.max_channels)
    if cfg.synthetic:
        sets["synthetic"] = synthetic_set()
    eval_data: dict[tuple[str, int], EvalData] = {}
    builder = build_multi_windows if cfg.multi else build_windows
    for name, series_set in sets.items():
        for context in contexts:
            windows = builder(series_set, context, cfg.horizon, cfg.windows)
            if windows is None:
                continue
            x, y, s, _ = windows
            eval_data[(name, context)] = EvalData(x, y, s)
    return sets, eval_data


def _flatten(pred: np.ndarray, targets: np.ndarray, scales: np.ndarray):
    """Collapse a (windows, variates, ...) prediction to per-series rows."""
    if pred.ndim == 4:
        return (
            pred.reshape(-1, pred.shape[-2], pred.shape[-1]),
            targets.reshape(-1, targets.shape[-1]),
            scales.reshape(-1),
        )
    return pred, targets, scales


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
        flat_pred, flat_targets, flat_scales = _flatten(pred, data.targets, data.scales)
        if ref is None:
            m = metrics.evaluate(flat_pred, flat_targets, flat_scales).to_dict()
        else:
            flat_ref, _, _ = _flatten(ref[(name, context)], data.targets, data.scales)
            m = metrics.compare(flat_pred, flat_ref, flat_targets, flat_scales)
        m["window_pinball"] = [
            float(value)
            for value in metrics.window_pinball(flat_pred, flat_targets, flat_scales)
        ]
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


def _throughput(
    model, cfg: RunConfig, contexts: tuple[int, ...], cap_batch: int | None = None, variates: int = 1
) -> dict:
    out: dict[str, dict] = {}
    for context in contexts:
        out[str(context)] = {}
        for batch in cfg.throughput_batches:
            if cap_batch is not None and batch > cap_batch:
                continue
            out[str(context)][str(batch)] = measure_latency(
                model, batch, context, cfg.horizon, cfg.repeats, variates=variates
            )
    return out


def _memory(model, cfg: RunConfig, contexts: tuple[int, ...], variates: int = 1) -> dict:
    return {
        str(context): measure_peak_memory(
            model, cfg.mem_batch, context, cfg.horizon, variates=variates
        )
        for context in contexts
    }


def _weight_run(
    cfg: RunConfig,
    spec: WeightSpec,
    eval_data: dict[tuple[str, int], EvalData],
    contexts: tuple[int, ...],
    ref: dict[tuple[str, int], np.ndarray] | None,
    ref_bytes: int | None,
    variates: int = 1,
) -> dict:
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, spec)
    set_attention_backend(model, cfg.attention)
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
        "throughput": _throughput(model, cfg, cfg.throughput_contexts, variates=variates),
        "memory": _memory(model, cfg, cfg.throughput_contexts, variates=variates),
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
    variates: int = 1,
    scope: str = "both",
) -> dict:
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, weight_spec)
    plans = make_scope_plans(name, len(model.transformer_stack.layers), scope)
    for plan_scope, plan in plans:
        for index, compressor in enumerate(plan):
            assignments = (
                {"seq": compressor, "var": compressor}
                if plan_scope == "both"
                else {plan_scope: compressor}
            )
            install_kv_scopes(model, [index], assignments)
    set_attention_backend(model, cfg.attention)

    # The first forward at the probe shape traces the compiled graph and records
    # every K/V tensor shape. Later calls at the same shape reuse the graph.
    probe_context = max(contexts)
    compressors = [compressor for _, plan in plans for compressor in plan]
    for compressor in compressors:
        compressor.stats.reset()
    probe = np.sin(np.linspace(0.0, 40.0, probe_context, dtype=np.float32))
    if variates > 1:
        probe_batch = np.stack([probe + 0.05 * j for j in range(variates)])
        probe_input = mx.array(np.stack([probe_batch] * cfg.mem_batch))
    else:
        probe_batch = np.stack([probe] * cfg.mem_batch)
        probe_input = mx.array(probe_batch)[:, None, :]
    mx.eval(model.decode(probe_input, cfg.horizon))
    logical = sum(compressor.logical_bytes() for compressor in compressors)
    fp16 = sum(compressor.fp16_bytes() for compressor in compressors)
    probe_calls = sum(len(compressor.stats.calls) for compressor in compressors)
    values = sum(
        math.prod(shape)
        for compressor in compressors
        for _, shape in compressor.stats.calls
    )
    mixed = len({compressor.bits_k for compressor in compressors}) > 1 or len(
        {compressor.bits_v for compressor in compressors}
    ) > 1
    bits = logical * 8.0 / values if values else 0.0
    bits_k = bits if mixed else plan[0].bits_k
    bits_v = bits if mixed else plan[0].bits_v

    acc, macro, _ = _accuracy(
        model, cfg, eval_data, contexts, ref, batch_size=16 if "turbo" in name else None
    )
    display = name if weight == "fp32" else f"{weight}+{name}"
    if scope != "both":
        display += f"@{scope}"
    result = {
        "name": display,
        "weight": weight,
        "kv_scope": scope,
        "bits_k": bits_k,
        "bits_v": bits_v,
        "bits_per_value": (bits_k + bits_v) / 2,
        "accuracy": acc,
        "macro": macro,
        "throughput": _throughput(
            model, cfg, contexts, cap_batch=8 if "turbo" in name else None, variates=variates
        ),
        "memory": _memory(model, cfg, contexts, variates=variates),
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
    if cfg.attention not in ("manual", "sdpa"):
        raise ValueError(f"unknown attention backend: {cfg.attention}")
    needed = tuple(sorted(set(cfg.contexts) | set(cfg.kv_contexts) | set(cfg.throughput_contexts)))
    sets, eval_data = _load_data(cfg, needed)
    variates = 1
    if cfg.multi and eval_data:
        variates = int(next(iter(eval_data.values())).contexts.shape[1])
    unavailable = sorted({c for c in needed if not any(k[1] == c for k in eval_data)})
    if unavailable:
        print(f"[data] contexts without data: {unavailable}", flush=True)

    # Always run the fp32 reference, even if it is not in the weight list.
    # fp32 runs first regardless of position, so every later run has a reference.
    weight_names = _weight_order(cfg.weights)
    specs = weight_specs(weight_names)
    baselines: dict[str, dict[str, dict]] = {}
    for (name, context), data in eval_data.items():
        baselines.setdefault(name, {})[str(context)] = metrics.baseline(
            data.contexts, data.targets, data.scales
        ).to_dict()

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
                        len(eval_data[(name, c)].contexts)
                        if cfg.multi
                        else len(eval_data[(name, c)].contexts) // max(1, len(series_set))
                    )
                    if (name, c) in eval_data
                    else 0
                    for c in needed
                },
            }
            for name, series_set in sets.items()
        },
        "baselines": baselines,
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
            set_attention_backend(model, cfg.attention)
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
                "throughput": _throughput(
                    model, cfg, cfg.throughput_contexts, variates=variates
                ),
                "memory": _memory(model, cfg, cfg.throughput_contexts, variates=variates),
            }
            unload_model(model)
            del model
        else:
            try:
                result = _weight_run(
                    cfg, spec, eval_data, cfg.contexts, ref_preds, ref_bytes, variates
                )
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

    for scope in cfg.kv_scopes:
        if scope not in ("both", "seq", "var"):
            raise ValueError(f"unknown KV scope: {scope}")
        for weight_name, weight_spec in kv_specs.items():
            for name in cfg.kv:
                if name == "none":
                    continue
                if name not in KV_NAMES:
                    raise ValueError(f"unknown KV mode: {name}")
                display = name if weight_name == "fp32" else f"{weight_name}+{name}"
                if scope != "both":
                    display += f"@{scope}"
                try:
                    result = _kv_run(
                        cfg,
                        name,
                        weight_name,
                        weight_spec,
                        eval_data,
                        ref_preds,
                        cfg.kv_contexts,
                        variates,
                        scope,
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
