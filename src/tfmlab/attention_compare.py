"""Compare two runs that differ only in the attention backend.

The fused SDPA path is not bit-identical to the manual matmul-softmax path,
and KV compression changes the gap. This loads the manual and SDPA result
files and reports the per-dataset differences so the switch is not treated as
free.

Run through ``tfmq attention-compare``.
"""

from __future__ import annotations

import json


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _runs(data: dict) -> dict[str, dict]:
    runs = data.get("weight_runs", []) + data.get("kv_runs", [])
    return {run["name"]: run for run in runs if "error" not in run}


def _delta(a: float, b: float) -> float:
    return 100.0 * (b - a) / max(abs(a), 1e-12)


_CONFIG_KEYS = (
    "checkpoint",
    "datasets",
    "synthetic",
    "max_channels",
    "contexts",
    "windows",
    "window_mode",
    "horizon",
    "weights",
    "kv",
    "kv_contexts",
    "kv_weights",
    "kv_scopes",
    "multi",
    "eval_batch",
)


def _check_compatible(manual: dict, sdpa: dict) -> None:
    """Reject comparisons where something other than the backend changed."""
    left = manual.get("meta", {}).get("config", {})
    right = sdpa.get("meta", {}).get("config", {})
    problems = [key for key in _CONFIG_KEYS if left.get(key) != right.get(key)]
    if left.get("attention", "manual") != "manual":
        problems.append("manual file attention")
    if right.get("attention", "manual") != "sdpa":
        problems.append("sdpa file attention")
    if problems:
        raise ValueError(
            "results differ beyond the attention backend: " + ", ".join(problems)
        )
    left_runs, right_runs = _runs(manual), _runs(sdpa)
    if set(left_runs) != set(right_runs):
        raise ValueError("the two result files contain different runs")
    for name, run in left_runs.items():
        for dataset, by_context in run.get("accuracy", {}).items():
            for context, entry in by_context.items():
                other = _get(right_runs[name], dataset, context)
                counts = [entry.get("window_pinball"), (other or {}).get("window_pinball")]
                if all(counts) and len(counts[0]) != len(counts[1]):
                    raise ValueError(
                        f"window counts differ for {name}/{dataset}/{context}"
                    )


def _get(data: dict, dataset: str, context: str):
    accuracy = data.get("accuracy", {})
    if dataset not in accuracy:
        return None
    return accuracy[dataset].get(context)


def render(manual_path: str, sdpa_path: str) -> str:
    manual = _load(manual_path)
    sdpa = _load(sdpa_path)
    _check_compatible(manual, sdpa)
    left, right = _runs(manual), _runs(sdpa)
    names = [name for name in left if name in right]
    if not names:
        raise ValueError("the two result files share no run names")

    rows = []
    for name in names:
        for dataset, by_context in left[name].get("accuracy", {}).items():
            for context in sorted(by_context, key=int):
                if context not in right[name].get("accuracy", {}).get(dataset, {}):
                    continue
                a = by_context[context]
                b = right[name]["accuracy"][dataset][context]
                rows.append(
                    [
                        name,
                        dataset,
                        context,
                        f"{a['pinball']:.4f}",
                        f"{b['pinball']:.4f}",
                        f"{_delta(a['pinball'], b['pinball']):+.2f}%",
                        f"{a['nmae']:.4f}",
                        f"{b['nmae']:.4f}",
                        f"{_delta(a['nmae'], b['nmae']):+.2f}%",
                    ]
                )

    lines = [
        "# 注意力后端对比（manual vs SDPA）",
        "",
        f"- manual：`{manual_path}`",
        f"- sdpa：`{sdpa_path}`",
        "- Δ 为 SDPA 相对 manual 的变化，正值表示 SDPA 更差。",
        "",
        "| 配置 | 数据集 | 上下文 | manual pinball | sdpa pinball | Δpinball | manual nMAE | sdpa nMAE | ΔnMAE |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def write(manual_path: str, sdpa_path: str, out: str) -> str:
    text = render(manual_path, sdpa_path)
    with open(out, "w") as f:
        f.write(text)
    return text
