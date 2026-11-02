"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from . import report
from .runner import PRESETS, RunConfig, config_from_preset, run


def _load_config(args) -> RunConfig:
    if args.config:
        with open(args.config) as f:
            data = yaml.safe_load(f)
        if args.preset and args.preset not in PRESETS:
            raise ValueError(f"unknown preset: {args.preset}")
        if args.preset:
            merged = dict(PRESETS[args.preset])
            merged.update(data)
            data = merged
        return RunConfig.from_dict(data)
    cfg = config_from_preset(args.preset or "quick")
    for key in ("checkpoint", "out", "windows", "repeats", "eval_batch"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _cmd_run(args) -> int:
    cfg = _load_config(args)
    run(cfg)
    print(f"report: tfmq report {cfg.out}")
    return 0


def _cmd_report(args) -> int:
    text = report.write(args.results, args.out)
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


def _cmd_sanity(args) -> int:
    import mlx.core as mx
    import numpy as np

    from .attention import _ORIGINAL, install_kv, patch_class
    from .kv import make_kv
    from .runtime import load_model

    patch_class()
    model = load_model(args.checkpoint)
    x = mx.array(np.sin(np.linspace(0, 40, 256, dtype=np.float32))[None, None, :])
    ref = np.array(model.decode(x, 64))
    model._compiled_forward = None
    install_kv(model, None)
    patched = np.array(model.decode(x, 64))
    print(f"attention patch, no KV: max|diff| = {np.abs(patched - ref).max():.2e}")
    for name in ("affine8", "affine4", "turbo3_mse", "turbo3_prod", "qjl1"):
        kv = make_kv(name)
        vec = mx.array(np.random.default_rng(0).normal(size=(256, 80)).astype(np.float32))
        out = np.array(kv._roundtrip(vec))
        print(f"{name:12s} roundtrip rel err = {np.linalg.norm(out - np.array(vec)) / np.linalg.norm(np.array(vec)):.3f}")
    return 0


def _cmd_sensitivity(args) -> int:
    from .sensitivity import SensitivityConfig, render, run as run_sensitivity

    cfg = SensitivityConfig(
        checkpoint=args.checkpoint,
        context=args.context,
        out=args.out,
    )
    run_sensitivity(cfg)
    text = render(args.out)
    with open(args.md, "w") as f:
        f.write(text)
    print(f"wrote {args.md}")
    return 0


def _cmd_kv_sensitivity(args) -> int:
    from .kv_sensitivity import KVSensitivityConfig, render, run as run_kv_sensitivity

    cfg = KVSensitivityConfig(
        checkpoint=args.checkpoint,
        context=args.context,
        kv=args.kv,
        windows=args.windows,
        out=args.out,
    )
    run_kv_sensitivity(cfg)
    text = render(args.out)
    with open(args.md, "w") as f:
        f.write(text)
    print(f"wrote {args.md}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tfmq")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run a benchmark")
    run_p.add_argument("--preset", choices=sorted(PRESETS), default="quick")
    run_p.add_argument("--config", help="YAML config file")
    run_p.add_argument("--checkpoint", default=None)
    run_p.add_argument("--out", default=None)
    run_p.add_argument("--windows", type=int, default=None)
    run_p.add_argument("--repeats", type=int, default=None)
    run_p.add_argument("--eval-batch", type=int, dest="eval_batch", default=None)
    run_p.set_defaults(func=_cmd_run)

    rep_p = sub.add_parser("report", help="render a Markdown report")
    rep_p.add_argument("results")
    rep_p.add_argument("-o", "--out", default="results/report.md")
    rep_p.set_defaults(func=_cmd_report)

    sens_p = sub.add_parser("sensitivity", help="layer-wise 4-bit sensitivity probe")
    sens_p.add_argument("--checkpoint", default="models/timesfm-3.0")
    sens_p.add_argument("--context", type=int, default=2048)
    sens_p.add_argument("--out", default="results/sensitivity.json")
    sens_p.add_argument("--md", default="results/sensitivity.md")
    sens_p.set_defaults(func=_cmd_sensitivity)

    san_p = sub.add_parser("sanity", help="check patch parity and KV roundtrip error")
    san_p.add_argument("--checkpoint", default="models/timesfm-3.0")
    san_p.set_defaults(func=_cmd_sanity)

    kvs_p = sub.add_parser("kv-sensitivity", help="layer and head sensitivity of KV compression")
    kvs_p.add_argument("--checkpoint", default="models/timesfm-3.0")
    kvs_p.add_argument("--context", type=int, default=15360)
    kvs_p.add_argument("--kv", default="affine4")
    kvs_p.add_argument("--windows", type=int, default=4)
    kvs_p.add_argument("--out", default="results/kv_sensitivity.json")
    kvs_p.add_argument("--md", default="results/kv_sensitivity.md")
    kvs_p.set_defaults(func=_cmd_kv_sensitivity)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
