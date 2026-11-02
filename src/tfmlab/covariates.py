"""Quantization and KV compression with past-only and past-future covariates.

The benchmark grids feed only the target context. TimesFM 3.0 also accepts
observed covariates up to the origin and known future covariates. This probe
runs that path so quantization and KV results are not limited to target-only
inputs.

Run through ``tfmq covariates``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import mlx.core as mx
import numpy as np

from . import metrics
from .attention import install_kv_named
from .data import build_covariate_windows, load_ett
from .runtime import load_model, predict, predict_with_covariates, unload_model
from .weights import apply_weight_spec, weight_specs


@dataclasses.dataclass
class CovariateConfig:
    checkpoint: str = "models/timesfm-3.0"
    data_dir: str = "data/raw"
    datasets: tuple[str, ...] = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
    context: int = 15360
    horizon: int = 64
    windows: int = 2
    batch: int = 8
    out: str = "results/covariates.json"
    md: str = "results/covariates.md"


def _load(cfg: CovariateConfig):
    frames = []
    for name in cfg.datasets:
        built = build_covariate_windows(
            load_ett(name, cfg.data_dir), cfg.context, cfg.horizon, cfg.windows
        )
        if built is not None:
            frames.append(built)
    if not frames:
        raise ValueError("no covariate windows")
    return tuple(np.concatenate([f[i] for f in frames]) for i in range(5))


def _kv_bytes(model, compressors, cfg, contexts, past_only, past_future) -> tuple[int, int]:
    batch = min(cfg.batch, len(contexts))
    target = np.repeat(contexts[:1], batch, axis=0)
    po = np.repeat(past_only[:1], batch, axis=0)
    pf = np.repeat(past_future[:1], batch, axis=0)
    for compressor in compressors:
        compressor.stats.reset()
    mx.eval(
        model.decode(
            mx.array(target)[:, None, :],
            cfg.horizon,
            past_only_covariates=mx.array(po),
            past_future_covariates=mx.array(pf),
        )
    )
    return (
        sum(compressor.logical_bytes() for compressor in compressors),
        sum(compressor.fp16_bytes() for compressor in compressors),
    )


def run(cfg: CovariateConfig) -> dict:
    contexts, targets, scales, past_only, past_future = _load(cfg)
    print(
        f"[covariates] {len(contexts)} targets, {past_only.shape[1]} past-only, "
        f"{past_future.shape[1]} past-future, context {cfg.context}",
        flush=True,
    )

    model = load_model(cfg.checkpoint)
    reference = predict_with_covariates(
        model, contexts, past_only, past_future, cfg.horizon, cfg.batch
    )
    baseline = metrics.evaluate(reference, targets, scales)
    unload_model(model)
    del model

    runs = [
        ("fp32 无协变量", None, None, False),
        ("fp32", None, None, True),
        ("int8_g64", "int8_g64", None, True),
        ("int4_g64", "int4_g64", None, True),
        ("mix_l18_19_8", "mix_l18_19_8", None, True),
        ("affine4", None, "affine4", True),
        ("affine4_seq_var8", None, "affine4_seq_var8", True),
        ("affine4_l19fp16", None, "affine4_l19fp16", True),
    ]
    rows = []
    for name, weight, kv_name, with_covariates in runs:
        model = load_model(cfg.checkpoint)
        if weight is not None:
            apply_weight_spec(model, weight_specs([weight])[weight])
        compressors = []
        if kv_name is not None:
            compressors = install_kv_named(model, kv_name)
            logical, fp16 = _kv_bytes(
                model, compressors, cfg, contexts, past_only, past_future
            )
        else:
            logical = fp16 = None

        if with_covariates:
            pred = predict_with_covariates(
                model, contexts, past_only, past_future, cfg.horizon, cfg.batch
            )
        else:
            pred = predict(model, contexts, cfg.horizon, cfg.batch)
        score = metrics.compare(pred, reference, targets, scales)
        rows.append(
            {
                "name": name,
                "pinball": score["pinball"],
                "pinball_delta_pct": score["pinball_delta_pct"],
                "nmae_delta_pct": score["nmae_delta_pct"],
                "crossing_rate": score["crossing_rate"],
                "kv_logical_bytes": logical,
                "kv_fp16_bytes": fp16,
            }
        )
        print(
            f"  {name:15s} Δpinball {rows[-1]['pinball_delta_pct']:+7.3f}%",
            flush=True,
        )
        unload_model(model)
        del model

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": dataclasses.asdict(cfg),
        "series": int(len(contexts)),
        "past_only": int(past_only.shape[1]),
        "past_future": int(past_future.shape[1]),
        "baseline": {"pinball": baseline.pinball, "nmae": baseline.nmae},
        "runs": rows,
    }
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(result, f, indent=2)
    with open(cfg.md, "w") as f:
        f.write(render(result))
    print(f"wrote {cfg.out} and {cfg.md}", flush=True)
    return result


def render(result: dict) -> str:
    cfg = result["config"]
    lines = [
        "# 协变量下的量化与 KV",
        "",
        f"- 数据：{', '.join(cfg['datasets'])}；目标为最后一个通道，其余 "
        f"{result['past_only']} 个通道为 past-only；{result['past_future']} 个日历特征为 past-future",
        f"- ctx {cfg['context']}；每条序列 {cfg['windows']} 个窗口；共 {result['series']} 条目标序列",
        f"- fp32 pinball：{result['baseline']['pinball']:.4f}",
        "",
        "| 配置 | pinball | Δpinball | ΔnMAE | cross % | KV MB | reduction |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["runs"]:
        if row["kv_logical_bytes"]:
            kv = f"{row['kv_logical_bytes'] / 1e6:.0f}"
            reduction = f"{row['kv_fp16_bytes'] / row['kv_logical_bytes']:.2f}×"
        else:
            kv = reduction = "—"
        lines.append(
            f"| {row['name']} | {row['pinball']:.4f} | {row['pinball_delta_pct']:+.3f}% | "
            f"{row['nmae_delta_pct']:+.3f}% | {row['crossing_rate'] * 100:.2f}% | {kv} | {reduction} |"
        )
    return "\n".join(lines) + "\n"
