"""Horizons beyond one output patch: stitching error under quantization.

A horizon above ``output_patch_len`` (64) is stitched from several output
patches. This probe repeats the same windows at 64, 256 and 1024 steps so the
quantization and KV cost of stitching is measured instead of assumed.

Run through ``tfmq horizon``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np

from . import metrics
from .attention import install_kv_named
from .data import eval_windows
from .runtime import load_model, measure_latency, predict, unload_model
from .weights import apply_weight_spec, weight_specs

RUNS = (
    ("fp32", None, None),
    ("int8_g64", "int8_g64", None),
    ("int4_g64", "int4_g64", None),
    ("mix_l18_19_8", "mix_l18_19_8", None),
    ("affine4", None, "affine4"),
)


@dataclasses.dataclass
class HorizonConfig:
    checkpoint: str = "models/timesfm-3.0"
    data_dir: str = "data/raw"
    datasets: tuple[str, ...] = ("ETTh1",)
    context: int = 2048
    horizons: tuple[int, ...] = (64, 256, 1024)
    windows: int = 4
    batch: int = 16
    repeats: int = 2
    out: str = "results/horizon.json"
    md: str = "results/horizon.md"


def run(cfg: HorizonConfig) -> dict:
    rows = []
    for horizon in cfg.horizons:
        contexts, targets, scales = eval_windows(
            cfg.data_dir, cfg.datasets, False, cfg.context, horizon, cfg.windows
        )
        model = load_model(cfg.checkpoint)
        reference = predict(model, contexts, horizon, cfg.batch)
        baseline = metrics.evaluate(reference, targets, scales)
        unload_model(model)
        del model

        for name, weight, kv_name in RUNS:
            model = load_model(cfg.checkpoint)
            if weight is not None:
                apply_weight_spec(model, weight_specs([weight])[weight])
            if kv_name is not None:
                install_kv_named(model, kv_name)
            pred = predict(model, contexts, horizon, cfg.batch)
            score = metrics.compare(pred, reference, targets, scales)
            latency = measure_latency(
                model, 8, cfg.context, horizon, repeats=cfg.repeats
            )
            rows.append(
                {
                    "horizon": horizon,
                    "name": name,
                    "pinball": score["pinball"],
                    "pinball_delta_pct": score["pinball_delta_pct"],
                    "nmae_delta_pct": score["nmae_delta_pct"],
                    "crossing_rate": score["crossing_rate"],
                    "latency_ms": latency["latency_ms_p50"],
                }
            )
            print(
                f"  h={horizon:<4} {name:16s} Δpinball {rows[-1]['pinball_delta_pct']:+7.3f}% "
                f"{rows[-1]['latency_ms']:7.1f} ms",
                flush=True,
            )
            unload_model(model)
            del model

        rows.append({"horizon": horizon, "name": "_baseline", "pinball": baseline.pinball})

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": dataclasses.asdict(cfg),
        "series": int(len(contexts)),
        "rows": rows,
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
        "# 更长 horizon 下的量化与 KV",
        "",
        f"- 数据：{', '.join(cfg['datasets'])}；ctx {cfg['context']}；每条序列 {cfg['windows']} 个窗口",
        f"- horizon：{', '.join(map(str, cfg['horizons']))}；延迟为 batch 8 的中位值",
        "",
        "| horizon | 配置 | Δpinball | ΔnMAE | cross % | ms |",
        "|---|---|---|---|---|---|",
    ]
    baseline = {
        row["horizon"]: row["pinball"]
        for row in result["rows"]
        if row["name"] == "_baseline"
    }
    for row in result["rows"]:
        if row["name"] == "_baseline":
            continue
        lines.append(
            f"| {row['horizon']} | {row['name']} | {row['pinball_delta_pct']:+.3f}% | "
            f"{row['nmae_delta_pct']:+.3f}% | {row['crossing_rate'] * 100:.2f}% | "
            f"{row['latency_ms']:.1f} |"
        )
    lines += ["", "fp32 基线的 pinball：" + "；".join(
        f"h={h} {p:.4f}" for h, p in sorted(baseline.items())
    )]
    return "\n".join(lines) + "\n"
