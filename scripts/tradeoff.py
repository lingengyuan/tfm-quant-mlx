"""Render the peak-memory / forecast-error scatter from the result files.

Reads the context-15360 macro and memory entries so the figure follows the
grids instead of a hand-copied table.

Usage: ``python scripts/tradeoff.py`` writes ``results/tradeoff.svg``.
"""

from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT = "15360"
SOURCES = (
    "results/full.json",
    "results/mixed.json",
    "results/clip.json",
    "results/combo.json",
)
# Points the README calls out. Everything else stays grey context.
LABELS = {
    "fp32": ("fp32", -8, 4, "end"),
    "int8_g64": ("int8_g64", 8, 4, "start"),
    "mix_l18_19_8+affine4_l19fp16": ("mix_l18_19_8 + affine4_l19fp16", 8, -18, "start"),
    "int4_g64": ("int4_g64", 8, 4, "start"),
}
WIDTH, HEIGHT = 800, 450
LEFT, RIGHT, TOP, BOTTOM = 72, 780, 46, 386
X_MIN, X_MAX = 1650.0, 2850.0
Y_MIN, Y_MAX = -1.2, 13.0
ACCENT = "#1f6feb"
GREY = "#b6bcc4"
INK = "#24292f"


def load_points() -> list[tuple[str, float, float]]:
    points: dict[str, tuple[str, float, float]] = {}
    for source in SOURCES:
        path = os.path.join(ROOT, source)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for run in data.get("weight_runs", []) + data.get("kv_runs", []):
            if "error" in run or run["name"] in points:
                continue
            # KV on fp32 weights keeps peak memory flat, which says nothing
            # about the memory/error trade-off this figure is about.
            if run.get("kv_scope") and run.get("weight", "fp32") == "fp32":
                continue
            macro = run.get("macro", {}).get(CONTEXT)
            memory = run.get("memory", {}).get(CONTEXT, {})
            if not macro or memory.get("peak_mb") is None:
                continue
            delta = macro.get("pinball_delta_pct")
            points[run["name"]] = (run["name"], memory["peak_mb"], delta or 0.0)
    return list(points.values())


def _x(memory: float) -> float:
    return LEFT + (memory - X_MIN) / (X_MAX - X_MIN) * (RIGHT - LEFT)


def _y(delta: float) -> float:
    return BOTTOM - (delta - Y_MIN) / (Y_MAX - Y_MIN) * (BOTTOM - TOP)


def render(points: list[tuple[str, float, float]]) -> str:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        f'<text x="{LEFT}" y="30" font-size="13" fill="{INK}">'
        "Peak memory vs forecast error, context 15,360</text>",
    ]
    for tick in range(0, 13, 2):
        y = _y(tick)
        out.append(
            f'<line x1="{LEFT}" y1="{y:.1f}" x2="{RIGHT}" y2="{y:.1f}" '
            f'stroke="{"#c9ced6" if tick == 0 else "#eef0f3"}" stroke-width="1"/>'
        )
        out.append(
            f'<text x="{LEFT - 8}" y="{y + 4:.1f}" font-size="11" fill="#57606a" '
            f'text-anchor="end">{tick}</text>'
        )
    for tick in range(1700, 2900, 200):
        x = _x(tick)
        out.append(
            f'<text x="{x:.1f}" y="{BOTTOM + 18}" font-size="11" fill="#57606a" '
            f'text-anchor="middle">{tick}</text>'
        )
    out += [
        f'<line x1="{LEFT}" y1="{BOTTOM}" x2="{RIGHT}" y2="{BOTTOM}" stroke="#8c959f"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{BOTTOM}" stroke="#8c959f"/>',
        f'<text x="{(LEFT + RIGHT) / 2:.0f}" y="{HEIGHT - 8}" font-size="12" '
        f'fill="{INK}" text-anchor="middle">Peak memory (MB)</text>',
        f'<text x="18" y="{(TOP + BOTTOM) / 2:.0f}" font-size="12" fill="{INK}" '
        f'text-anchor="middle" transform="rotate(-90 18 {(TOP + BOTTOM) / 2:.0f})">'
        "Δpinball vs fp32 (%)</text>",
    ]

    for name, memory, delta in sorted(points, key=lambda p: p[0] not in LABELS):
        x, y = _x(memory), _y(delta)
        if name in LABELS:
            continue
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{GREY}"/>')

    for name, memory, delta in points:
        if name not in LABELS:
            continue
        text, dx, dy, anchor = LABELS[name]
        x, y = _x(memory), _y(delta)
        out.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{ACCENT}" '
            f'stroke="#ffffff" stroke-width="1.5"/>'
        )
        out.append(
            f'<text x="{x + dx:.1f}" y="{y + dy:.1f}" font-size="12" fill="{INK}" '
            f'text-anchor="{anchor}">{text}</text>'
        )

    out += [
        f'<circle cx="{LEFT + 4}" cy="{BOTTOM + 46}" r="3.5" fill="{GREY}"/>',
        f'<text x="{LEFT + 14}" y="{BOTTOM + 50}" font-size="11" fill="#57606a">'
        "other weight and KV configurations</text>",
        "</svg>",
    ]
    return "\n".join(out) + "\n"


def main() -> None:
    points = load_points()
    path = os.path.join(ROOT, "results", "tradeoff.svg")
    with open(path, "w") as f:
        f.write(render(points))
    print(f"wrote {path} ({len(points)} points)")


if __name__ == "__main__":
    main()
