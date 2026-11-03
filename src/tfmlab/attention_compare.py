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


def render(manual_path: str, sdpa_path: str) -> str:
    manual = _load(manual_path)
    sdpa = _load(sdpa_path)
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
