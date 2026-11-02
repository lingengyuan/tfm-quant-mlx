"""Attention patch that inserts K/V compression before the softmax.

The installed TimesFM 3.0.2 MLX attention computes post-RoPE, post-RMSNorm K/V
inline. Re-implementing that body is the only way to intercept the tensors
without changing the checkpoint format. The no-compressor path delegates to the
original method, so unmodified runs are bit-identical; a test asserts this.

The patch targets TimesFM 3.0.x. If an upgrade changes the attention body, the
parity test fails instead of silently benchmarking different arithmetic.
"""

from __future__ import annotations

import math

import mlx.core as mx

import timesfm3.mlx.transformer as tfm_transformer
from timesfm3.mlx.transformer import rope

from .kv import KVCompressor

_KV_ATTR = "_tfmlab_kv"
_ORIGINAL = tfm_transformer.MultiHeadAttention.__call__


def _attention_with_kv(self, x: mx.array, patch_mask: mx.array) -> mx.array:
    kv: KVCompressor | None = getattr(self, _KV_ATTR, None)
    if kv is None:
        return _ORIGINAL(self, x, patch_mask)

    b, n, _ = x.shape
    if n == 1:
        return self.out_proj(self.value_proj(x))
    h, hd = self.num_heads, self.head_dim
    q = self.query_proj(x).reshape(b, n, h, hd)
    k = self.key_proj(x).reshape(b, n, h, hd)
    v = self.value_proj(x).reshape(b, n, h, hd)
    if self.use_rope:
        pos = mx.broadcast_to(mx.arange(n)[None, :], (b, n))
        q = rope(q, pos)
        k = rope(k, pos)
    q = self.query_ln(q)
    k = self.key_ln(k)
    q = self.per_dim_scale(q)

    k = kv.apply(k, "k")
    v = kv.apply(v, "v")

    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    kv_valid = (~patch_mask)[:, None, None, :]
    if self.causal:
        qi = mx.arange(n)[None, None, :, None]
        ki = mx.arange(n)[None, None, None, :]
        attend = (qi >= ki) & kv_valid
    else:
        attend = mx.broadcast_to(kv_valid, (b, 1, n, n))
    bias = mx.where(attend, 0.0, -1e9)
    q = q * math.sqrt(hd)
    logits = (q @ k.transpose(0, 1, 3, 2)) + bias
    w = mx.softmax(logits, axis=-1)
    out = w @ v
    out = out.transpose(0, 2, 1, 3).reshape(b, n, h * hd)
    return self.out_proj(out)


def install_kv(model, kv: KVCompressor | None, scope: str = "both", layers=None) -> None:
    """Attach ``kv`` to sequence and/or variate attention of the selected layers.

    ``layers`` is an iterable of block indices; ``None`` covers every block.
    """
    if scope not in ("both", "seq", "var"):
        raise ValueError(f"unknown KV scope: {scope}")
    blocks = model.transformer_stack.layers
    indices = range(len(blocks)) if layers is None else layers
    for index in indices:
        layer = blocks[index]
        for attr, part in (("seq_attn", "seq"), ("var_attn", "var")):
            module = getattr(layer, attr, None)
            if module is None:
                continue
            setattr(module, _KV_ATTR, kv if scope in ("both", part) else None)
    model._compiled_forward = None


def install_kv_scopes(model, layers, assignments: dict[str, KVCompressor | None]) -> None:
    """Set compressors for the given scopes only; other scopes stay untouched.

    ``install_kv`` clears the scope it is not installing, which would undo a
    per-scope plan that keeps sequence and variate attention on different
    compressors.
    """
    blocks = model.transformer_stack.layers
    indices = range(len(blocks)) if layers is None else layers
    for index in indices:
        layer = blocks[index]
        for attr, part in (("seq_attn", "seq"), ("var_attn", "var")):
            if part not in assignments:
                continue
            module = getattr(layer, attr, None)
            if module is not None:
                setattr(module, _KV_ATTR, assignments[part])
    model._compiled_forward = None


def patch_class() -> None:
    tfm_transformer.MultiHeadAttention.__call__ = _attention_with_kv


patch_class()
