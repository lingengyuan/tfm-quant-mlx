"""Export a quantized model and load it without re-quantizing.

Every prediction otherwise loads 1.3 GB of fp32 weights and quantizes them
again. A converted directory stores the packed tensors plus the spec and the
architecture, and ``tfmq`` loads it when the checkpoint path contains
``tfmlab-quant.json``.

Run through ``tfmq convert``.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from .runtime import load_model, synthetic_input, unload_model
from .weights import WeightSpec, apply_weight_spec, weight_nbytes, weight_specs

META_NAME = "tfmlab-quant.json"
WEIGHTS_NAME = "model.safetensors"


@dataclasses.dataclass
class ConvertConfig:
    checkpoint: str = "models/timesfm-3.0"
    weights: str = "int8_g64"
    out: str = "models/timesfm-int8"
    context: int = 512
    verify: bool = True


def is_converted(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, META_NAME))


def save_quantized(model, spec: WeightSpec, out: str, source: str) -> dict:
    os.makedirs(out, exist_ok=True)
    arrays = dict(tree_flatten(model.parameters()))
    weights_path = os.path.join(out, WEIGHTS_NAME)
    mx.save_safetensors(weights_path, arrays)
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": source,
        "weight_spec": dataclasses.asdict(spec),
        "model_config": dataclasses.asdict(model.config),
        "weights_file": WEIGHTS_NAME,
        "weight_bytes": weight_nbytes(model),
    }
    with open(os.path.join(out, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_quantized(path: str):
    """Rebuild the quantized module tree randomly, then load the packed tensors."""
    from timesfm3.mlx.configs import TimesFM3MlxConfig
    from timesfm3.mlx.model import TimesFM3Mlx

    with open(os.path.join(path, META_NAME)) as f:
        meta = json.load(f)
    spec = WeightSpec(**meta["weight_spec"])
    model = TimesFM3Mlx(TimesFM3MlxConfig(**meta["model_config"]))
    apply_weight_spec(model, spec)
    model.load_safetensors(os.path.join(path, meta["weights_file"]))
    model._compiled_forward = None
    mx.eval(model.parameters())
    # Marks the model so a second quantization is rejected instead of silently
    # quantizing the packed integers.
    model._tfmlab_spec_name = spec.name
    return model


def _probe(model, context: int) -> np.ndarray:
    out = model.decode(synthetic_input(2, context), 64)
    mx.eval(out)
    return np.array(out)


def run(cfg: ConvertConfig) -> dict:
    spec = weight_specs([cfg.weights])[cfg.weights]
    model = load_model(cfg.checkpoint)
    apply_weight_spec(model, spec)
    reference = _probe(model, cfg.context) if cfg.verify else None
    meta = save_quantized(model, spec, cfg.out, cfg.checkpoint)
    unload_model(model)
    del model

    loaded = load_quantized(cfg.out)
    if cfg.verify:
        out = _probe(loaded, cfg.context)
        diff = float(np.abs(out - reference).max())
        meta["verification"] = {
            "context": cfg.context,
            "max_abs_diff": diff,
        }
        with open(os.path.join(cfg.out, META_NAME), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[convert] max|diff| = {diff:.3e}", flush=True)
    unload_model(loaded)
    print(
        f"wrote {cfg.out}: {meta['weight_bytes'] / 1e6:.0f} MB, spec {spec.name}",
        flush=True,
    )
    return meta
