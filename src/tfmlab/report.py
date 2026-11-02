"""Render benchmark JSON as a Markdown report."""

from __future__ import annotations

import json
from typing import Any


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:+.{digits}f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _get(d: dict, *keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _weight_table(runs: list[dict], context: int) -> str:
    rows = []
    for run in runs:
        m = _get(run, "macro", str(context))
        if not m:
            continue
        mem = _get(run, "memory", str(context), "peak_mb")
        tp = _get(run, "throughput", str(context), "32", "series_per_s")
        if tp is None:
            tp = _get(run, "throughput", str(context), "8", "series_per_s")
        bits = run.get("effective_bits")
        if bits is None and run.get("kind") == "mixed":
            bits_text = "mix"
        elif bits is None:
            dtype = (run.get("spec") or {}).get("dtype") or "float32"
            bits_text = "32.00" if dtype == "float32" else "16.00"
        else:
            bits_text = f"{bits:.2f}"
        rows.append(
            [
                run["name"],
                f"{run['weight_bytes'] / 1e6:.0f}",
                f"{run.get('weight_ratio') or 1:.2f}×",
                bits_text,
                _fmt(m.get("nmae")),
                _pct(m.get("nmae_delta_pct")),
                _fmt(m.get("pinball")),
                _pct(m.get("pinball_delta_pct")),
                _fmt(m.get("coverage_error")),
                f"{m.get('crossing_rate', 0) * 100:.2f}",
                _fmt(tp, 0),
                _fmt(mem, 0),
            ]
        )
    if not rows:
        return "_no data_"
    return _table(
        [
            "config",
            "weights MB",
            "ratio",
            "bits/w",
            "nMAE",
            "ΔnMAE",
            "pinball",
            "Δpinball",
            "cov err",
            "cross %",
            "series/s",
            "peak MB",
        ],
        rows,
    )


def _kv_table(runs: list[dict], context: int) -> str:
    rows = []
    for run in runs:
        m = _get(run, "macro", str(context))
        if not m:
            continue
        mem = _get(run, "memory", str(context), "peak_mb")
        tp = _get(run, "throughput", str(context), "8", "series_per_s")
        if tp is None:
            tp = _get(run, "throughput", str(context), "1", "series_per_s")
        rows.append(
            [
                run["name"],
                f"{run['bits_k']:.2f}",
                f"{run['bits_v']:.2f}",
                f"{run['kv_logical_bytes'] / 1e6:.0f}",
                f"{run['kv_reduction']:.2f}×",
                _fmt(m.get("nmae")),
                _pct(m.get("nmae_delta_pct")),
                _pct(m.get("pinball_delta_pct")),
                _fmt(m.get("point_dev")),
                _fmt(m.get("mean_dev")),
                _fmt(m.get("coverage_error")),
                _fmt(tp, 0),
                _fmt(mem, 0),
            ]
        )
    if not rows:
        return "_no data_"
    return _table(
        [
            "KV mode",
            "K bits",
            "V bits",
            "KV MB",
            "reduction",
            "nMAE",
            "ΔnMAE",
            "Δpinball",
            "point dev",
            "mean dev",
            "cov err",
            "series/s",
            "peak MB",
        ],
        rows,
    )


def _quantile_table(runs: list[dict], context: int, label: str = "KV mode") -> str:
    taus = [f"{t:.1f}" for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    rows = []
    for run in runs:
        m = _get(run, "macro", str(context))
        if not m:
            continue
        dev = m.get("per_quantile_dev")
        if not dev:
            continue
        rows.append([run["name"]] + [f"{v:.4f}" for v in dev])
    if not rows:
        return "_no data_"
    return _table([label] + taus, rows)


def _context_sweep(runs: list[dict], contexts: list[int], kind: str) -> str:
    header = [kind] + [f"ctx {c}" for c in contexts]
    rows = []
    for run in runs:
        row = [run["name"]]
        for context in contexts:
            m = _get(run, "macro", str(context))
            if not m:
                row.append("—")
                continue
            delta = m.get("pinball_delta_pct")
            row.append(f"{m['nmae']:.4f} / {'0.00' if delta is None else f'{delta:+.2f}'}%")
        rows.append(row)
    if not rows:
        return "_no data_"
    return _table(header, rows)


def _throughput_table(runs: list[dict], context: int) -> str:
    header = ["config"]
    batches: list[str] = []
    for run in runs:
        for batch in _get(run, "throughput", str(context), default={}) or {}:
            if batch not in batches:
                batches.append(batch)
    batches.sort(key=int)
    header += [f"b{b} ms / series/s" for b in batches]
    rows = []
    for run in runs:
        row = [run["name"]]
        for batch in batches:
            entry = _get(run, "throughput", str(context), batch)
            if entry is None:
                row.append("—")
            else:
                row.append(f"{entry['latency_ms_p50']:.1f} / {entry['series_per_s']:.0f}")
        rows.append(row)
    return _table(header, rows)


def _findings(weight_runs: list[dict], kv_runs: list[dict], weight_ctx: int, kv_ctx: int) -> list[str]:
    lines = []
    fp32 = next((r for r in weight_runs if r["name"] == "fp32"), None)
    if fp32:
        for run in weight_runs:
            m = _get(run, "macro", str(weight_ctx))
            if not m or run["name"] == "fp32":
                continue
            if abs(m.get("pinball_delta_pct", 99)) <= 1.0:
                lines.append(
                    f"{run['name']} 在 {weight_ctx} 上下文把权重压到 "
                    f"{run['weight_bytes'] / 1e6:.0f} MB（{run['weight_ratio']:.2f}×），"
                    f"pinball 变化 {_pct(m.get('pinball_delta_pct'))}。"
                )
        worst = max(
            (r for r in weight_runs if r["name"] != "fp32"),
            key=lambda r: _get(r, "macro", str(weight_ctx), "pinball_delta_pct", default=-1e9),
            default=None,
        )
        if worst and len(weight_runs) > 2:
            m = _get(worst, "macro", str(weight_ctx))
            lines.append(
                f"{worst['name']} 是 {weight_ctx} 上下文的最差档：pinball "
                f"{_pct(m.get('pinball_delta_pct'))}，nMAE {_pct(m.get('nmae_delta_pct'))}。"
            )
        fp32_peak = _get(fp32, "memory", str(weight_ctx), "peak_mb")
        best = min(
            (r for r in weight_runs if r["name"].startswith("int4")),
            key=lambda r: _get(r, "weight_bytes", default=1e18),
            default=None,
        )
        if fp32_peak and best:
            best_peak = _get(best, "memory", str(weight_ctx), "peak_mb")
            if best_peak:
                lines.append(
                    f"权重 {fp32['weight_bytes'] / 1e6:.0f} → {best['weight_bytes'] / 1e6:.0f} MB 时，"
                    f"峰值显存 {fp32_peak:.0f} → {best_peak:.0f} MB。"
                )
        fp32_tp = _get(fp32, "throughput", str(weight_ctx), "32", "series_per_s")
        if fp32_tp:
            tps = [
                _get(r, "throughput", str(weight_ctx), "32", "series_per_s")
                for r in weight_runs
            ]
            tps = [t for t in tps if t]
            if tps and len(weight_runs) > 1:
                lines.append(
                    f"权重吞吐在 {weight_ctx}、batch 32 为 {min(tps):.0f}–{max(tps):.0f} series/s，"
                    "量化没有带来加速。"
                )
    for run in kv_runs:
        m = _get(run, "macro", str(kv_ctx))
        if not m:
            continue
        if abs(m.get("pinball_delta_pct", 99)) <= 2.0:
            lines.append(
                f"KV {run['name']} 在 {kv_ctx} 上下文压缩 {run['kv_reduction']:.2f}×，"
                f"pinball 变化 {_pct(m.get('pinball_delta_pct'))}。"
            )
    kv_peaks = []
    for run in kv_runs:
        if run.get("weight", "fp32") != "fp32":
            continue
        peak = _get(run, "memory", str(kv_ctx), "peak_mb")
        kv_peaks.append((run["name"], peak))
    if fp32 and kv_peaks:
        fp32_peak = _get(fp32, "memory", str(kv_ctx), "peak_mb")
        if fp32_peak:
            lines.append(
                f"KV 压缩后峰值显存为 {min(p for _, p in kv_peaks if p):.0f}–"
                f"{max(p for _, p in kv_peaks if p):.0f} MB，fp32 为 {fp32_peak:.0f} MB；"
                "解码回 fp32 抵消了逻辑 K/V 的缩减。"
            )
    bad = [
        r
        for r in kv_runs
        if abs(_get(r, "macro", str(kv_ctx), "pinball_delta_pct", default=0) or 0) > 20
    ]
    for run in bad:
        m = _get(run, "macro", str(kv_ctx))
        lines.append(
            f"KV {run['name']} 不可用于生产：pinball {_pct(m.get('pinball_delta_pct'))}。"
        )
    return lines


def generate(path: str) -> str:
    with open(path) as f:
        data = json.load(f)
    env = data.get("env", {})
    cfg = data.get("meta", {}).get("config", {})
    weight_runs = data.get("weight_runs", [])
    kv_runs = data.get("kv_runs", [])
    contexts = sorted({int(c) for run in weight_runs for c in run.get("macro", {})})
    kv_contexts = sorted({int(c) for run in kv_runs for c in run.get("macro", {})})
    headline = 2048 if 2048 in contexts else (contexts[0] if contexts else 512)
    long_ctx = max(contexts) if contexts else 15360
    kv_ctx = 15360 if 15360 in kv_contexts else (max(kv_contexts) if kv_contexts else long_ctx)

    parts = ["# TimesFM-MLX 量化与长上下文 KV 压缩报告", ""]
    parts += [
        f"- 日期：{data.get('meta', {}).get('created')}",
        f"- 主机：{env.get('host')} / {env.get('machine')} / {env.get('platform')}",
        f"- MLX {env.get('mlx')}，timesfm {env.get('timesfm')}，Python {env.get('python')}",
        f"- checkpoint：`{env.get('checkpoint')}`，参数量 {env.get('param_count', 0) / 1e6:.0f}M",
        f"- 数据集：{', '.join(data.get('datasets', {}))}",
        f"- 上下文：{', '.join(map(str, contexts))}；horizon {cfg.get('horizon')}；"
        f"每条序列窗口数 ≤ {cfg.get('windows')}；评测 batch {cfg.get('eval_batch')}",
        f"- 总耗时：{data.get('meta', {}).get('elapsed_s')} s",
        "",
        "指标：nMAE 与 pinball 损失按每条序列上下文标准差归一化；Δ 为相对 fp32 基线的变化；"
        "point dev 为点预测相对 fp32 的平均绝对偏移（同样归一化）；cov err 为九个十分位覆盖率与名义值之差的平均。",
        "",
        f"## 权重量化（上下文 {headline}）",
        "",
        _weight_table(weight_runs, headline),
        "",
        f"## 权重量化（上下文 {long_ctx}）",
        "",
        _weight_table(weight_runs, long_ctx),
        "",
        "## 权重全上下文扫描（nMAE / Δpinball）",
        "",
        _context_sweep(weight_runs, contexts, "config"),
        "",
        f"## 权重吞吐（上下文 {long_ctx}）",
        "",
        _throughput_table(weight_runs, long_ctx),
        "",
    ]
    weight_quantiles = _quantile_table(weight_runs, long_ctx, "config")
    if weight_quantiles != "_no data_":
        parts += [
            f"## 权重每十分位偏移（上下文 {long_ctx}）",
            "",
            weight_quantiles,
            "",
        ]
    if kv_runs:
        parts += [
            "## KV 全上下文扫描（nMAE / Δpinball）",
            "",
            _context_sweep(kv_runs, kv_contexts, "KV mode"),
            "",
            f"## KV 压缩（上下文 {kv_ctx}）",
            "",
            f"逻辑 K/V 字节在 batch {cfg.get('mem_batch')}、上下文 {kv_ctx} 下测量。",
            "",
            _kv_table(kv_runs, kv_ctx),
            "",
            f"## KV 每十分位偏移（上下文 {kv_ctx}）",
            "",
            _quantile_table(kv_runs, kv_ctx),
            "",
        ]
    findings = _findings(weight_runs, kv_runs, headline, kv_ctx)
    if findings:
        parts += ["## 观察", ""] + [f"- {line}" for line in findings] + [""]
    notes = [
        "## 说明",
        "",
        "- fp32 基线为同一 checkpoint、同一编译路径、同一批窗口。",
        "- bits、逻辑 KV 字节为按张量形状计算的理论值；peak MB 是含常驻权重的 MLX 峰值显存。",
        "- 吞吐为 batch 1/8/32 的中位延迟。",
        "- int4_g128 跳过输入维不整除 128 的层（192 维残差块），这些层保持 fp32。",
    ]
    if kv_runs:
        notes += [
            "- TimesFM 3.0 MLX 为单次全序列前向，没有跨调用 KV cache；此处 KV 指每层注意力的 K/V 工作集。",
            "- turbo 模式在长上下文只测到 batch 8。",
            "- 压缩后 K/V 会解码回 fp32 再参与注意力，因此本次测量中峰值显存没有随逻辑 K/V 下降。",
        ]
    if cfg.get("multi"):
        notes.append(
            "- 多变量网格每次 decode 含该数据集全部通道（ETT 为 7 个 variate），"
            "KV 字节按全部 variate 统计，绝对值不能与单变量表直接比较。"
        )
    parts += notes
    return "\n".join(parts) + "\n"


def write(path: str, out: str) -> str:
    text = generate(path)
    with open(out, "w") as f:
        f.write(text)
    return text
