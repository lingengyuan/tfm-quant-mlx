"""KV working-set compression for the TimesFM 3.0 MLX attention path.

TimesFM 3.0 MLX is non-autoregressive: one forward pass covers context and
horizon patches, so the model has no cache that survives a call. What exists
per call is the K/V working set of every attention block. At a 15,360-point
context this working set dominates activations, and it is what this module
compresses.

Implementations:

* ``affine``: MLX affine quantization of K and V, ``bits`` in {4, 8}. Heads are
  flattened before quantization because head_dim=80 is not divisible by the
  supported group sizes.
* ``turbo``: PolarQuant (``mse``) and PolarQuant + QJL residual (``prod``) from
  arXiv:2504.19874, following the MLX port in lingengyuan/qjl-mlx (MIT).
  Vectors are padded to the next power of two per head.
* ``qjl``: 1-bit sign estimator from arXiv:2406.03482. Applied to K only; V
  stays at fp16, because 1-bit reconstruction of values is not meaningful.

All compressors decode before attention, so they change quality and the logical
size of K/V but do not by themselves speed up arithmetic. Throughput results
show that cost honestly.
"""

from __future__ import annotations

import dataclasses
import math

import mlx.core as mx
import numpy as np

_CODEBOOK_CACHE: dict[tuple[int, int], tuple[list[np.ndarray], list[np.ndarray]]] = {}
_ROTATION_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _orthogonal(dim: int, seed: int) -> np.ndarray:
    key = (dim, seed)
    if key not in _ROTATION_CACHE:
        rng = np.random.default_rng(seed)
        sample = rng.standard_normal((dim, dim), dtype=np.float32)
        q, r = np.linalg.qr(sample)
        q = q * np.sign(np.diag(r))[None, :]
        _ROTATION_CACHE[key] = np.ascontiguousarray(q, dtype=np.float32)
    return _ROTATION_CACHE[key]


def _padded_dim(dim: int) -> int:
    return 1 << (max(1, dim) - 1).bit_length()


def _level_sizes(padded_dim: int) -> tuple[int, ...]:
    sizes = []
    current = padded_dim
    while current > 1:
        current //= 2
        sizes.append(current)
    return tuple(sizes)


def _analytic_codebook(level: int, bins: int, grid_size: int = 8001) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd-Max codebook for the PolarQuant level density sin(2x)^(2^(level-1)-1)."""
    key = (level, bins)
    if key in _CODEBOOK_CACHE:
        return _CODEBOOK_CACHE[key]
    grid = np.linspace(0.0, math.pi / 2, grid_size, dtype=np.float64)
    delta = grid[1] - grid[0]
    pdf = np.sin(2.0 * grid) ** ((2 ** (level - 1)) - 1)
    pdf = pdf / (np.trapezoid(pdf, grid) + 1e-30)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) * 0.5 * delta)])
    cdf /= cdf[-1]
    boundaries = np.interp(np.linspace(0.0, 1.0, bins + 1), cdf, grid)
    centers = np.zeros(bins, dtype=np.float64)
    for _ in range(60):
        for i in range(bins):
            mask = (grid >= boundaries[i]) & (grid <= boundaries[i + 1])
            denom = np.trapezoid(pdf[mask], grid[mask])
            numer = np.trapezoid(grid[mask] * pdf[mask], grid[mask])
            centers[i] = numer / denom if denom > 1e-30 else (boundaries[i] + boundaries[i + 1]) / 2
        new = np.empty_like(boundaries)
        new[0], new[-1] = 0.0, math.pi / 2
        new[1:-1] = (centers[:-1] + centers[1:]) / 2
        if np.max(np.abs(new - boundaries)) < 1e-9:
            boundaries = new
            break
        boundaries = new
    result = (boundaries.astype(np.float32), centers.astype(np.float32))
    _CODEBOOK_CACHE[key] = result
    return result


def _uniform_codebook(level: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    upper = 2 * math.pi if level == 1 else math.pi / 2
    edges = np.linspace(0.0, upper, bins + 1, dtype=np.float32)
    return edges, ((edges[:-1] + edges[1:]) / 2).astype(np.float32)


@dataclasses.dataclass
class KVStats:
    """Per-forward K/V shapes. Populated once per compiled shape."""

    calls: list[tuple[str, tuple[int, ...]]]

    def reset(self) -> None:
        self.calls.clear()


class KVCompressor:
    name = "none"
    parts: tuple[str, ...] = ("k", "v")
    bits_k: float = 16.0
    bits_v: float = 16.0
    # None compresses every head. A tuple limits compression to those head
    # indices; the reader still sees the other heads at full precision, which is
    # how a mixed KV policy is expressed.
    heads: tuple[int, ...] | None = None
    # Use the fused MLX attention kernel after decompression.
    fast: bool = False

    def __init__(self) -> None:
        self.stats = KVStats([])

    def apply(self, x: mx.array, part: str) -> mx.array:
        self.stats.calls.append((part, tuple(x.shape)))
        if part not in self.parts:
            return x
        out = self._roundtrip(x)
        if self.heads is None:
            return out
        mask = np.zeros(x.shape[-2], dtype=bool)
        mask[list(self.heads)] = True
        return mx.where(mx.array(mask)[None, None, :, None], out, x)

    def _roundtrip(self, x: mx.array) -> mx.array:
        raise NotImplementedError

    def bits_per_value(self, part: str) -> float:
        return self.bits_k if part == "k" else self.bits_v

    def logical_bytes(self) -> int:
        total = 0.0
        for part, shape in self.stats.calls:
            values = math.prod(shape)
            if self.heads is not None:
                values = values * len(self.heads) / shape[-2]
            total += values * self.bits_per_value(part) / 8.0
        return int(round(total))

    def fp16_bytes(self) -> int:
        return int(sum(math.prod(shape) * 2 for _, shape in self.stats.calls))


class AffineKV(KVCompressor):
    def __init__(self, bits: int = 8, group_size: int = 64) -> None:
        super().__init__()
        self.bits = bits
        self.group_size = group_size
        self.name = f"affine{bits}_g{group_size}"
        self.bits_k = self.bits_v = bits + 64.0 / group_size

    def _roundtrip(self, x: mx.array) -> mx.array:
        shape = x.shape
        d = shape[-2] * shape[-1]
        flat = x.reshape(-1, d).astype(mx.float32)
        wq, scales, biases = mx.quantize(flat, self.group_size, self.bits)
        out = mx.dequantize(wq, scales, biases, self.group_size, self.bits)
        return out.reshape(shape).astype(x.dtype)


class TurboKV(KVCompressor):
    """PolarQuant + optional QJL residual, one rotation and codebook per head_dim."""

    def __init__(self, bits: int = 3, mode: str = "mse", seed: int = 42) -> None:
        super().__init__()
        if bits < 2:
            raise ValueError("turbo KV needs total_bits >= 2")
        if mode not in ("mse", "prod"):
            raise ValueError(f"unknown turbo mode: {mode}")
        self.bits = bits
        self.mode = mode
        self.name = f"turbo{mode}{bits}"
        self.angle_bits = bits if mode == "mse" else bits - 1
        self.bins = 2 ** self.angle_bits
        self._dim: int | None = None
        self._padded: int | None = None
        self._levels: tuple[int, ...] = ()
        self._S = None
        self._ST = None
        self._Sq = None
        self._codebooks: list[tuple[mx.array, mx.array]] = []
        self._seed = seed
        self.bits_k = self.bits_v = 0.0

    def _prepare(self, dim: int) -> None:
        if self._dim == dim:
            return
        self._dim = dim
        self._padded = _padded_dim(dim)
        self._levels = _level_sizes(self._padded)
        self._S = mx.array(_orthogonal(self._padded, self._seed))
        self._ST = mx.transpose(self._S)
        self._codebooks = []
        for level in range(1, len(self._levels) + 1):
            edges, centers = (
                _analytic_codebook(level, self.bins) if level > 1 else _uniform_codebook(level, self.bins)
            )
            self._codebooks.append((mx.array(edges), mx.array(centers)))
        self.bits_k = self.bits_v = (
            math.fsum(self._levels) * self.angle_bits + 32
        ) / self._padded
        if self.mode == "prod":
            self._Sq = mx.array(_orthogonal(self._padded, self._seed + 1))
            self.bits_k += 1.0
            self.bits_v += 1.0

    def _roundtrip(self, x: mx.array) -> mx.array:
        shape = x.shape
        d = shape[-1]
        self._prepare(d)
        flat = x.reshape(-1, d).astype(mx.float32)
        if self._padded > d:
            flat = mx.pad(flat, ((0, 0), (0, self._padded - d)))
        rotated = flat @ self._ST
        codes, norms = self._encode_angles(rotated)
        recon = self._decode_angles(codes, norms)
        if self.mode == "prod":
            residual = rotated - recon
            signs = mx.where(residual @ mx.transpose(self._Sq) >= 0, 1.0, -1.0)
            recon = recon + (math.sqrt(math.pi / 2) / self._padded) * (signs @ self._Sq)
        out = recon @ self._S
        out = out[:, :d].reshape(shape)
        return out.astype(x.dtype)

    def _encode_angles(self, rotated: mx.array) -> tuple[mx.array, mx.array]:
        current = rotated
        codes = []
        for level, size in enumerate(self._levels, start=1):
            pairs = current.reshape((current.shape[0], size, 2))
            radii = mx.sqrt(mx.sum(pairs * pairs, axis=2))
            angles = mx.arctan2(pairs[:, :, 1], pairs[:, :, 0])
            if level == 1:
                angles = mx.where(angles < 0, angles + 2 * math.pi, angles)
            edges, _ = self._codebooks[level - 1]
            code = mx.sum((angles[..., None] >= edges[1:-1]).astype(mx.int32), axis=-1)
            codes.append(code.reshape(code.shape[0], -1))
            current = radii
        return mx.concatenate(codes, axis=1), current.reshape((current.shape[0],))

    def _decode_angles(self, codes: mx.array, norms: mx.array) -> mx.array:
        current = norms[:, None]
        offset = int(math.fsum(self._levels))
        for index in reversed(range(len(self._levels))):
            level = index + 1
            size = self._levels[index]
            offset -= size
            _, centers = self._codebooks[index]
            angles = mx.take(centers, codes[:, offset : offset + size].astype(mx.int32))
            current = mx.stack([current * mx.cos(angles), current * mx.sin(angles)], axis=-1)
            current = current.reshape((codes.shape[0], size * 2))
        return current[:, : self._padded]


class QJLKV(KVCompressor):
    """1-bit QJL for keys; values are left at fp16."""

    parts = ("k",)

    def __init__(self, seed: int = 42) -> None:
        super().__init__()
        self.name = "qjl1"
        self.bits_k = 0.0
        self.bits_v = 16.0
        self._seed = seed
        self._dim: int | None = None
        self._S = None
        self._padded = None

    def _prepare(self, dim: int) -> None:
        if self._dim == dim:
            return
        self._dim = dim
        self._padded = _padded_dim(dim)
        self._S = mx.array(_orthogonal(self._padded, self._seed))
        self.bits_k = self._padded / dim + 32.0 / dim

    def _roundtrip(self, x: mx.array) -> mx.array:
        shape = x.shape
        d = shape[-1]
        self._prepare(d)
        flat = x.reshape(-1, d).astype(mx.float32)
        if self._padded > d:
            flat = mx.pad(flat, ((0, 0), (0, self._padded - d)))
        norms = mx.linalg.norm(flat, axis=-1)
        signs = mx.where((flat @ mx.transpose(self._S)) >= 0, 1.0, -1.0)
        scale = math.sqrt(math.pi / 2) / self._padded
        out = (signs @ self._S) * (norms * scale)[:, None]
        out = out[:, :d].reshape(shape)
        return out.astype(x.dtype)


class PassThroughKV(KVCompressor):
    """Records K/V shapes without changing them; used for layers kept at fp16."""

    name = "fp16"
    parts: tuple[str, ...] = ()


def make_kv(name: str) -> KVCompressor | None:
    if name in ("none", "off", ""):
        return None
    if name == "affine8":
        return AffineKV(8, 64)
    if name == "affine4":
        return AffineKV(4, 64)
    if name == "turbo3_mse":
        return TurboKV(3, "mse")
    if name == "turbo2_mse":
        return TurboKV(2, "mse")
    if name == "turbo3_prod":
        return TurboKV(3, "prod")
    if name == "qjl1":
        return QJLKV()
    if name in KV_LAYER_PLANS or name in KV_SCOPE_PLANS:
        raise ValueError(f"mixed KV mode needs make_kv_plan: {name}")
    raise ValueError(f"unknown KV compressor: {name}")


# Modes that use different compressors per attention scope. The runner installs
# each plan on its own scope and sums bytes across both.
KV_SCOPE_PLANS: dict[str, dict[str, str | None]] = {
    "affine4_seq_var8": {"seq": "affine4", "var": "affine8"},
    "affine4_seq_varfp16": {"seq": "affine4", "var": None},
}


# Base mode for every layer plus per-layer overrides; ``None`` keeps that layer
# at fp16. Named after the sensitivity result: the last block carries most of
# the KV error, so it is the only layer worth spending 16 bits on.
KV_LAYER_PLANS: dict[str, tuple[str, dict[int, str | None]]] = {
    "affine4_l19fp16": ("affine4", {19: None}),
    "turbo3_l19fp16": ("turbo3_mse", {19: None}),
}


def make_kv_plan(name: str, layers: int) -> list[KVCompressor]:
    """One compressor per transformer block, so bytes can be summed per layer."""
    base, overrides = KV_LAYER_PLANS.get(name, (name, {}))
    for index in overrides:
        if index >= layers:
            raise ValueError(f"{name} needs layer {index}, model has {layers}")
    plan = []
    for index in range(layers):
        spec = overrides.get(index, base)
        plan.append(PassThroughKV() if spec in (None, "none") else make_kv(spec))
    return plan


def make_scope_plans(
    name: str, layers: int, scope: str = "both"
) -> list[tuple[str, list[KVCompressor]]]:
    """Per-scope compressor plans. Normal modes reuse one plan for ``scope``."""
    if name in KV_SCOPE_PLANS:
        if scope != "both":
            raise ValueError(f"{name} fixes its own scopes; drop kv_scopes")
        return [
            (part, make_kv_plan(mode or "none", layers))
            for part, mode in KV_SCOPE_PLANS[name].items()
        ]
    return [(scope, make_kv_plan(name, layers))]


KV_NAMES = (
    "none",
    "affine8",
    "affine4",
    "turbo3_mse",
    "turbo2_mse",
    "turbo3_prod",
    "qjl1",
    *KV_LAYER_PLANS,
    *KV_SCOPE_PLANS,
)
