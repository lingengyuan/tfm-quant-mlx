"""Cross-backend parity: PyTorch reference versus MLX fp32.

Every delta in this repo is relative to the MLX fp32 output. This probe runs the
same windows through the PyTorch backend of the same checkpoint, so the
absolute reference is measured instead of assumed.

Run through ``tfmq parity``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np

from . import metrics
from .data import eval_windows


@dataclasses.dataclass
class ParityConfig:
    checkpoint: str = "models/timesfm-3.0"
    data_dir: str = "data/raw"
    datasets: tuple[str, ...] = ("ETTh1",)
    context: int = 15360
    horizon: int = 64
    windows: int = 2
    batch: int = 8
    device: str = "cpu"
    out: str = "results/parity.json"


def _mlx_predictions(cfg: ParityConfig, contexts: np.ndarray) -> np.ndarray:
    from .runtime import load_model, predict, unload_model

    model = load_model(cfg.checkpoint)
    pred = predict(model, contexts, cfg.horizon, cfg.batch)
    unload_model(model)
    del model
    return pred


def _torch_predictions(cfg: ParityConfig, contexts: np.ndarray) -> np.ndarray:
    try:
        from timesfm3.torch.timesfm3_forecaster import TimesFM3Forecaster
    except ModuleNotFoundError as error:
        if error.name == "torch":
            raise RuntimeError("parity needs the torch backend: uv pip install torch") from error
        raise

    forecaster = TimesFM3Forecaster.from_pretrained(cfg.checkpoint, device=cfg.device)
    outputs = list(
        forecaster.predict_batch(
            [context for context in contexts],
            cfg.horizon,
            return_quantiles=True,
            sort_quantiles=False,
        )
    )
    if any(output.quantiles is None for output in outputs):
        raise RuntimeError("torch backend returned no quantiles")
    return np.stack([output.quantiles for output in outputs], axis=0)


def compare(mlx: np.ndarray, torch: np.ndarray, targets: np.ndarray, scales: np.ndarray) -> dict:
    dev = np.abs(mlx - torch) / scales[:, None, None]
    point = np.abs(mlx[:, :, metrics.MEDIAN_INDEX] - torch[:, :, metrics.MEDIAN_INDEX])
    pinball_mlx = metrics.evaluate(mlx, targets, scales).pinball
    pinball_torch = metrics.evaluate(torch, targets, scales).pinball
    return {
        "max_abs_dev": float(dev.max()),
        "mean_abs_dev": float(dev.mean()),
        "p95_abs_dev": float(np.quantile(dev, 0.95)),
        "point_mean_abs_dev": float((point / scales[:, None]).mean()),
        "per_quantile_dev": [float(v) for v in dev.mean(axis=(0, 1))],
        "pinball_mlx": pinball_mlx,
        "pinball_torch": pinball_torch,
        "pinball_delta_pct": 100.0 * (pinball_mlx - pinball_torch) / pinball_torch,
        "correlation": float(np.corrcoef(mlx.ravel(), torch.ravel())[0, 1]),
    }


def run(cfg: ParityConfig) -> dict:
    contexts, targets, scales = eval_windows(
        cfg.data_dir, cfg.datasets, False, cfg.context, cfg.horizon, cfg.windows
    )
    print(
        f"[parity] {len(contexts)} series, context {cfg.context}, torch device {cfg.device}",
        flush=True,
    )

    started = time.time()
    mlx = _mlx_predictions(cfg, contexts)
    torch = _torch_predictions(cfg, contexts)
    result = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": dataclasses.asdict(cfg),
        "series": int(len(contexts)),
        "elapsed_s": round(time.time() - started, 1),
        "mlx": {"shape": list(mlx.shape)},
        "torch": {"shape": list(torch.shape)},
        "metrics": compare(mlx, torch, targets, scales),
    }
    os.makedirs(os.path.dirname(cfg.out) or ".", exist_ok=True)
    with open(cfg.out, "w") as f:
        json.dump(result, f, indent=2)

    m = result["metrics"]
    print(
        f"  max dev {m['max_abs_dev']:.4f}  mean {m['mean_abs_dev']:.4f}  "
        f"corr {m['correlation']:.6f}  pinball Δ {m['pinball_delta_pct']:+.3f}%",
        flush=True,
    )
    print(f"wrote {cfg.out}", flush=True)
    return result
