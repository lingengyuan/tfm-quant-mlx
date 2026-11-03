"""Layer-wise 4-bit sensitivity and mixed-precision probes.

The uniform int4 result hides a large spread: most transformer layers tolerate
4-bit, a few (notably the last layer) and the output head do not. This module
quantizes one component at a time and reports the pinball shift, so a mixed
configuration can spend 8-bit bits only where they matter.

Run through ``tfmq sensitivity``.
"""

from __future__ import annotations

import dataclasses
from typing import Callable
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from . import metrics
from .data import eval_windows
from .runtime import load_model, predict, unload_model
from .weights import weight_nbytes


@dataclasses.dataclass
class SensitivityConfig:
    checkpoint: str = "models/timesfm-3.0"
    data_dir: str = "data/raw"
    datasets: tuple[str, ...] = ("ETTh1",)
    synthetic: bool = True
    context: int = 2048
    horizon: int = 64
    windows: int = 2
    # Reserve this many recent windows for the evaluation grid.
    offset: int = 0
    batch: int = 32
    bits: int = 4
    group_size: int = 64
    out: str = "results/sensitivity.json"


def _quantize_matching(model, match, bits: int, group_size: int) -> None:
    def predicate(path: str, module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if not match(path):
            return False
        return module.weight.shape[-1] % group_size == 0

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    model._compiled_forward = None


def _load_eval(cfg: SensitivityConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return eval_windows(
        cfg.data_dir,
        cfg.datasets,
        cfg.synthetic,
        cfg.context,
        cfg.horizon,
        cfg.windows,
        cfg.offset,
    )


def _tests(bits: int) -> list[tuple[str, Callable[[str], bool], int]]:
    tests: list[tuple[str, Callable[[str], bool], int]] = []
    for index in range(20):
        tests.append(
            (
                f"layer{index:02d}",
                lambda path, index=index: path.startswith(f"transformer_stack.layers.{index}."),
                bits,
            )
        )
    tests += [
        ("all_attn", lambda path: "seq_attn" in path or "var_attn" in path, bits),
        ("all_ffn", lambda path: ".ff0" in path or ".ff1" in path, bits),
        ("resblock", lambda path: path.startswith("pre_transformer_resblock"), bits),
        ("head8", lambda path: path == "output_head", 8),
        ("head4", lambda path: path == "output_head", 4),
    ]
    return tests


def run(cfg: SensitivityConfig) -> dict:
    contexts, targets, scales = _load_eval(cfg)
    print(f"[sensitivity] {len(contexts)} windows, context {cfg.context}", flush=True)

    model = load_model(cfg.checkpoint)
    reference = predict(model, contexts, cfg.horizon, cfg.batch)
    baseline = metrics.evaluate(reference, targets, scales)
    unload_model(model)

    rows = []
    for name, match, bits in _tests(cfg.bits):
        model = load_model(cfg.checkpoint)
        _quantize_matching(model, match, bits, cfg.group_size)
        pred = predict(model, contexts, cfg.horizon, cfg.batch)
        score = metrics.evaluate(pred, targets, scales)
        rows.append(
            {
                "name": name,
                "bits": bits,
                "weight_bytes": weight_nbytes(model),
                "pinball": score.pinball,
                "pinball_delta_pct": 100.0 * (score.pinball - baseline.pinball) / baseline.pinball,
                "crossing_rate": score.crossing_rate,
            }
        )
        print(
            f"  {name:10s} {bits}-bit  Δpinball {rows[-1]['pinball_delta_pct']:+7.3f}%",
            flush=True,
        )
        unload_model(model)
        del model

    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": dataclasses.asdict(cfg),
        "baseline": {
            "pinball": baseline.pinball,
            "nmae": baseline.nmae,
            "crossing_rate": baseline.crossing_rate,
        },
        "tests": rows,
    }
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {cfg.out}", flush=True)
    return result


def render(path: str) -> str:
    with open(path) as f:
        data = json.load(f)
    lines = [
        "# 逐层 4-bit 敏感度",
        "",
        f"- 上下文：{data['config']['context']}；窗口：{data['config']['windows']}；"
        f"数据：{', '.join(data['config']['datasets'])}"
        + (" + synthetic" if data["config"]["synthetic"] else ""),
        f"- fp32 pinball：{data['baseline']['pinball']:.4f}",
        "",
        "| 组件 | bits | 权重 MB | Δpinball | 交叉率 |",
        "|---|---|---|---|---|",
    ]
    if data["config"].get("offset"):
        lines.insert(
            2,
            f"- 窗口偏移：{data['config']['offset']}"
            f"（为评测预留最近 {data['config']['offset']} 个窗口）",
        )
    for row in data["tests"]:
        lines.append(
            f"| {row['name']} | {row['bits']} | {row['weight_bytes'] / 1e6:.0f} | "
            f"{row['pinball_delta_pct']:+.3f}% | {row['crossing_rate'] * 100:.2f}% |"
        )
    return "\n".join(lines) + "\n"
