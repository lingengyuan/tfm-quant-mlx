"""Weight-level precision and quantization.

The checkpoint is fp32. Three families are supported:

* dtype cast: fp32, fp16, bf16 (the whole module tree changes dtype);
* affine weight-only quantization: N-bit, group size 32/64/128;
* skipping the output head, which carries the nine quantile logits and is
  disproportionately sensitive per parameter.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


@dataclasses.dataclass(frozen=True)
class WeightSpec:
    name: str
    dtype: str | None = None
    bits: int = 0
    group_size: int = 64
    quantize_head: bool = False
    skip_prefixes: tuple[str, ...] = ()
    # (path prefix, bits) overrides applied on top of ``bits``. Empty for a
    # uniform configuration.
    overrides: tuple[tuple[str, int], ...] = ()
    # Per-group clipping fraction of the group max before quantization; None
    # keeps RTN. Clipping trades outlier error for a smaller step size.
    clip: float | None = None

    @property
    def kind(self) -> str:
        if self.overrides:
            return "mixed"
        return "quant" if self.bits else "dtype"


def weight_specs(names: Iterable[str] | None = None) -> dict[str, WeightSpec]:
    specs = {
        "fp32": WeightSpec("fp32", dtype="float32"),
        "fp16": WeightSpec("fp16", dtype="float16"),
        "bf16": WeightSpec("bf16", dtype="bfloat16"),
        "int8_g64": WeightSpec("int8_g64", bits=8, group_size=64),
        "int4_g64": WeightSpec("int4_g64", bits=4, group_size=64),
        "int4_g32": WeightSpec("int4_g32", bits=4, group_size=32),
        "int4_g128": WeightSpec("int4_g128", bits=4, group_size=128),
        # RTN with per-group clipping. The fraction is searched on ETT + Jena.
        "int4c90_g64": WeightSpec("int4c90_g64", bits=4, group_size=64, clip=0.90),
        "int4c95_g64": WeightSpec("int4c95_g64", bits=4, group_size=64, clip=0.95),
        "int4c98_g64": WeightSpec("int4c98_g64", bits=4, group_size=64, clip=0.98),
        # Mixed configs keep the sensitive last layer and the input residual
        # block at 8-bit; everything else stays at 4-bit.
        "mix_l19_8": WeightSpec(
            "mix_l19_8",
            bits=4,
            overrides=(("transformer_stack.layers.19.", 8), ("pre_transformer_resblock", 8)),
        ),
        "mix_l18_19_8": WeightSpec(
            "mix_l18_19_8",
            bits=4,
            overrides=(
                ("transformer_stack.layers.18.", 8),
                ("transformer_stack.layers.19.", 8),
                ("pre_transformer_resblock", 8),
            ),
        ),
        "mix_top3_8": WeightSpec(
            "mix_top3_8",
            bits=4,
            overrides=(
                ("transformer_stack.layers.3.", 8),
                ("transformer_stack.layers.18.", 8),
                ("transformer_stack.layers.19.", 8),
                ("pre_transformer_resblock", 8),
            ),
        ),
        "mix_l18_19_8c95": WeightSpec(
            "mix_l18_19_8c95",
            bits=4,
            clip=0.95,
            overrides=(
                ("transformer_stack.layers.18.", 8),
                ("transformer_stack.layers.19.", 8),
                ("pre_transformer_resblock", 8),
            ),
        ),
    }
    if names is None:
        return specs
    missing = set(names) - set(specs)
    if missing:
        raise ValueError(f"unknown weight specs: {sorted(missing)}")
    return {name: specs[name] for name in names}


def _clip_groups(weight, group_size: int, fraction: float):
    """Clip each input-dim group to ``fraction`` of its max absolute value."""
    out_dims, in_dims = weight.shape
    groups = weight.reshape(out_dims, in_dims // group_size, group_size)
    limit = fraction * mx.max(mx.abs(groups), axis=-1, keepdims=True)
    return mx.clip(groups, -limit, limit).reshape(out_dims, in_dims)


def apply_weight_spec(model, spec: WeightSpec) -> None:
    """Mutate ``model`` in place. The caller must invalidate ``_compiled_forward``."""
    stored = getattr(model, "_tfmlab_spec_name", None)
    if stored is not None:
        if spec.name != stored:
            raise ValueError(
                f"converted checkpoint stores {stored} weights; pass --weights {stored}"
            )
        return
    if spec.bits:
        prefixes = spec.skip_prefixes
        head = spec.quantize_head

        def bits_for(path: str) -> int:
            bits = spec.bits
            for prefix, override in spec.overrides:
                if path.startswith(prefix):
                    bits = override
            return bits

        def predicate(path: str, module) -> bool | dict:
            if not isinstance(module, nn.Linear):
                return False
            if path == "output_head" and not head:
                return False
            if any(path.startswith(p) for p in prefixes):
                return False
            # Sequential layers (e.g. the 192->1280 residual block) may not be
            # divisible by group_size=128; leave those in fp32.
            if module.weight.shape[-1] % spec.group_size:
                return False
            bits = bits_for(path)
            if bits == spec.bits:
                return True
            return {"group_size": spec.group_size, "bits": bits}

        if spec.clip is not None:
            for path, module in model.named_modules():
                if not isinstance(module, nn.Linear) or not predicate(path, module):
                    continue
                if bits_for(path) != spec.bits:
                    continue
                module.weight = _clip_groups(module.weight, spec.group_size, spec.clip)

        nn.quantize(
            model,
            group_size=spec.group_size,
            bits=spec.bits,
            class_predicate=predicate,
        )
    else:
        dtype = DTYPES[spec.dtype or "float32"]
        model.set_dtype(dtype)
        model._compute_dtype = dtype
    model._compiled_forward = None
    mx.eval(model.parameters())


def weight_nbytes(model) -> int:
    return int(sum(p.nbytes for _, p in tree_flatten(model.parameters())))


def logical_quant_bits(spec: WeightSpec) -> float | None:
    """Effective bits per weight, including affine scales and biases in fp32.

    Mixed specs have no single bit width; report the measured weight bytes instead.
    """
    if not spec.bits or spec.overrides:
        return None
    return spec.bits + 64.0 / spec.group_size
