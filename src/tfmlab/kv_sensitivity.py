"""Which layers and heads carry the KV compression error.

Uniform KV compression hides where the error comes from. At 15,360 context the
sequence attention K/V working set dominates, so a mixed policy needs to know
which blocks and heads to keep at higher precision. This probe compresses one
component at a time and reports the pinball shift.

Only sequence attention is probed: the benchmark feeds one variate per series,
so variate attention takes its ``n == 1`` shortcut and never builds K/V.

Run through ``tfmq kv-sensitivity``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

from . import metrics
from .attention import install_kv
from .data import eval_windows
from .kv import make_kv
from .runtime import load_model, predict, unload_model


@dataclasses.dataclass
class KVSensitivityConfig:
    checkpoint: str = "models/timesfm-3.0"
    data_dir: str = "data/raw"
    datasets: tuple[str, ...] = ("ETTh1",)
    synthetic: bool = True
    context: int = 15360
    horizon: int = 64
    windows: int = 4
    batch: int = 32
    kv: str = "affine4"
    out: str = "results/kv_sensitivity.json"


def _score(pred, targets, scales, baseline) -> dict:
    score = metrics.evaluate(pred, targets, scales)
    return {
        "pinball": score.pinball,
        "pinball_delta_pct": 100.0 * (score.pinball - baseline.pinball) / baseline.pinball,
        "nmae": score.nmae,
        "nmae_delta_pct": 100.0 * (score.nmae - baseline.nmae) / baseline.nmae,
        "crossing_rate": score.crossing_rate,
    }


def run(cfg: KVSensitivityConfig) -> dict:
    contexts, targets, scales = eval_windows(
        cfg.data_dir, cfg.datasets, cfg.synthetic, cfg.context, cfg.horizon, cfg.windows
    )

    model = load_model(cfg.checkpoint)
    baseline = metrics.evaluate(predict(model, contexts, cfg.horizon, cfg.batch), targets, scales)
    blocks = len(model.transformer_stack.layers)
    heads = model.transformer_stack.layers[0].seq_attn.num_heads
    unload_model(model)
    del model
    print(f"[kv-sensitivity] {len(contexts)} windows, {blocks} layers, {heads} heads", flush=True)

    tests: list[tuple[str, list[int] | None, list[int] | None, int | None]] = [
        ("all", None, None, None)
    ]
    tests += [(f"layer{index:02d}", [index], None, None) for index in range(blocks)]
    tests += [(f"no_layer{index:02d}", None, None, index) for index in range(blocks)]
    tests += [(f"head{index:02d}", None, [index], None) for index in range(heads)]

    rows = []
    for name, layers, head_ids, skip in tests:
        model = load_model(cfg.checkpoint)
        kv = make_kv(cfg.kv)
        kv.heads = tuple(head_ids) if head_ids is not None else None
        install_kv(model, kv, "seq", layers)
        if skip is not None:
            install_kv(model, None, "seq", [skip])
        row = {"name": name, "layers": layers, "heads": head_ids, "skip": skip}
        row.update(_score(predict(model, contexts, cfg.horizon, cfg.batch), targets, scales, baseline))
        rows.append(row)
        print(
            f"  {name:10s} Δpinball {row['pinball_delta_pct']:+7.3f}% "
            f"cross {row['crossing_rate'] * 100:5.2f}%",
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
    cfg = data["config"]
    rows = data["tests"]
    overall = next(r for r in rows if r["name"] == "all")
    layers = [r for r in rows if r["layers"] is not None]
    kept = [r for r in rows if r["skip"] is not None]
    heads = sorted((r for r in rows if r["heads"] is not None), key=lambda r: -r["pinball_delta_pct"])

    lines = [
        "# KV 逐层与逐头敏感度",
        "",
        f"- 模式：{cfg['kv']}；上下文：{cfg['context']}；窗口：{cfg['windows']}；"
        f"数据：{', '.join(cfg['datasets'])}" + (" + synthetic" if cfg["synthetic"] else ""),
        f"- fp32 pinball：{data['baseline']['pinball']:.4f}；全量压缩 "
        f"Δpinball {overall['pinball_delta_pct']:+.3f}%",
        "- 只测序列注意力：variate 注意力在单变量输入下不产生 K/V。",
        "- 单层/单头行只压缩该组件；no_layer 行压缩除该层外的所有层，"
        "回收 = 全量 Δ − 该行 Δ，即保留该层不压缩消掉的 pinball 偏移。",
        "",
        "## 全量",
        "",
        "| 组件 | Δpinball | ΔnMAE | 交叉率 |",
        "|---|---|---|---|",
        f"| all | {overall['pinball_delta_pct']:+.3f}% | {overall['nmae_delta_pct']:+.3f}% | "
        f"{overall['crossing_rate'] * 100:.2f}% |",
        "",
        "## 逐层（只压缩该层）",
        "",
        "| 层 | Δpinball | ΔnMAE |",
        "|---|---|---|",
    ]
    for row in layers:
        lines.append(
            f"| {row['layers'][0]:02d} | {row['pinball_delta_pct']:+.3f}% | "
            f"{row['nmae_delta_pct']:+.3f}% |"
        )
    lines += [
        "",
        "## 逐层（保留该层不压缩）",
        "",
        "| 层 | Δpinball | 回收 | ΔnMAE |",
        "|---|---|---|---|",
    ]
    for row in kept:
        recovery = overall["pinball_delta_pct"] - row["pinball_delta_pct"]
        lines.append(
            f"| {row['skip']:02d} | {row['pinball_delta_pct']:+.3f}% | {recovery:+.3f} pp | "
            f"{row['nmae_delta_pct']:+.3f}% |"
        )
    lines += [
        "",
        "## 逐头（只压缩该头）",
        "",
        "| 头 | Δpinball | ΔnMAE |",
        "|---|---|---|",
    ]
    for row in heads:
        lines.append(
            f"| {row['heads'][0]:02d} | {row['pinball_delta_pct']:+.3f}% | "
            f"{row['nmae_delta_pct']:+.3f}% |"
        )
    layer_sum = sum(r["pinball_delta_pct"] for r in layers)
    head_sum = sum(r["pinball_delta_pct"] for r in heads)
    best = max(kept, key=lambda r: overall["pinball_delta_pct"] - r["pinball_delta_pct"])
    lines += [
        "",
        f"单层 Δ 合计 {layer_sum:+.3f}%，单头合计 {head_sum:+.3f}%，全量 "
        f"{overall['pinball_delta_pct']:+.3f}%。三项不可加，只用于看误差分布。",
        f"保留第 {best['skip']:02d} 层不压缩回收最多："
        f"{overall['pinball_delta_pct'] - best['pinball_delta_pct']:+.3f} pp。",
    ]
    return "\n".join(lines) + "\n"
