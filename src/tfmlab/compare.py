"""Paired comparison of one run against another on the same windows.

Both runs share every evaluation window, so the pinball difference is paired
by forecast origin and the interval covers the candidate minus the base, not
their separate distances from fp32. Used to test whether a KV strategy that
helps on one dataset still helps on the others.

Run through ``tfmq compare``.
"""

from __future__ import annotations

import json

import numpy as np

from . import metrics


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _runs(data: dict) -> dict[str, dict]:
    runs = data.get("weight_runs", []) + data.get("kv_runs", [])
    return {run["name"]: run for run in runs if "error" not in run}


def _entry(run: dict, dataset: str, context: str) -> dict | None:
    return run.get("accuracy", {}).get(dataset, {}).get(context)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _pct(value: float) -> str:
    return f"{value:+.2f}%"


def render(path: str, base_name: str, draws: int = 2000) -> str:
    data = _load(path)
    runs = _runs(data)
    if base_name not in runs:
        raise ValueError(f"no run named {base_name}")
    base = runs[base_name]
    rng = np.random.default_rng(0)

    rows: list[list[str]] = []
    verdicts: list[str] = []
    anchors: list[str] = []
    for name, run in runs.items():
        if name == base_name:
            continue
        points: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        domains: list[str] = []
        for dataset, by_context in run.get("accuracy", {}).items():
            for context in sorted(by_context, key=int):
                current = by_context[context]
                reference = _entry(base, dataset, context)
                if reference is None:
                    continue
                candidate_values = current.get("window_pinball")
                reference_values = reference.get("window_pinball")
                if not candidate_values or not reference_values:
                    continue
                if len(candidate_values) != len(reference_values):
                    continue
                point, samples = metrics.paired_draws(
                    candidate_values,
                    reference_values,
                    current.get("window_group"),
                    draws,
                    rng,
                )
                low, high = (float(v) for v in np.percentile(samples, [2.5, 97.5]))
                points.append(point)
                lows.append(low)
                highs.append(high)
                domains.append(dataset)
                rows.append(
                    [
                        name,
                        dataset,
                        context,
                        _fmt(reference.get("pinball")),
                        _fmt(current.get("pinball")),
                        _pct(point),
                        f"[{low:+.2f}%, {high:+.2f}%]",
                        "是" if high < 0 or low > 0 else "否",
                    ]
                )

        if not points:
            continue
        worst = max(range(len(points)), key=lambda i: points[i])
        if "kv_scope" not in run:
            anchors.append(
                f"- {name} 不在判定范围：它是本轮的参照（不压缩）。"
                f"相对 {base_name} 平均 {np.mean(points):+.2f}%，即 {base_name} 的压缩代价。"
            )
            continue
        better = sum(1 for high in highs if high < 0)
        if all(high < 0 for high in highs):
            verdict = "通用推荐"
        elif all(low > 0 for low in lows):
            verdict = "不推荐"
        else:
            verdict = "条件推荐"
        verdicts.append(
            f"- {name}：平均 {np.mean(points):+.2f}%，最差 {points[worst]:+.2f}%（{domains[worst]}）；"
            f"{better}/{len(points)} 个数据集的 95% CI 上界低于 0。判定：{verdict}。"
        )

    lines = [
        "# 配对对比",
        "",
        f"- 结果文件：`{path}`",
        f"- 基线：`{base_name}`",
        "- 候选与基线使用同一批窗口。pinball 为绝对损失；候选−基线 与 CI 为相对变化，"
        "按预测时点重采样 2000 次，负值表示候选更好。",
        "",
    ]
    if rows:
        lines += [
            "| 候选 | 数据集 | 上下文 | 基线 pinball | 候选 pinball | 候选−基线 | 95% CI | CI 不含 0 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        lines += ["", "## 判定", ""]
        lines += anchors + verdicts
        lines += ["", "通用推荐要求全部 CI 上界低于 0，不推荐要求全部下界高于 0。"]
    else:
        lines += ["_no paired windows_"]
    return "\n".join(lines) + "\n"


def write(path: str, base_name: str, out: str) -> str:
    text = render(path, base_name)
    with open(out, "w") as f:
        f.write(text)
    return text
