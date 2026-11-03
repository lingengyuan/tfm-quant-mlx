"""Forecast accuracy metrics for point forecasts and nine deciles."""

from __future__ import annotations

import dataclasses

import numpy as np

TAUS = np.asarray((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), dtype=np.float32)
MEDIAN_INDEX = 4


@dataclasses.dataclass
class Metrics:
    nmae: float
    nrmse: float
    pinball: float
    per_quantile_pinball: list[float]
    coverage: list[float]
    coverage_error: float
    crossing_rate: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _pinball(target: np.ndarray, quantiles: np.ndarray, taus: np.ndarray) -> np.ndarray:
    # target (N, H); quantiles (N, H, T)
    diff = target[..., None] - quantiles
    return np.maximum(taus * diff, (taus - 1.0) * diff)


def evaluate(pred: np.ndarray, target: np.ndarray, scales: np.ndarray) -> Metrics:
    """``pred`` is (N, H, 9), unsorted. Outputs are normalized per series by ``scales``.

    Sorting matches the default ``predict(..., sort_quantiles=True)`` behavior.
    Crossing is measured on the raw output, before sorting.
    """
    if not np.isfinite(pred).all():
        raise ValueError("non-finite forecast")
    raw = pred
    crossing = float(np.mean(raw[..., 1:] < raw[..., :-1]))
    pred = np.sort(pred, axis=-1)
    scale = scales[:, None]
    point = pred[..., MEDIAN_INDEX]
    err = (point - target) / scale
    nmae = float(np.mean(np.abs(err)))
    nrmse = float(np.sqrt(np.mean(err**2)))
    loss = _pinball(target, pred, TAUS) / scale[..., None]
    per_q = np.mean(loss, axis=(0, 1))
    coverage = np.mean(target[..., None] <= pred, axis=(0, 1))
    return Metrics(
        nmae=nmae,
        nrmse=nrmse,
        pinball=float(np.mean(per_q)),
        per_quantile_pinball=[float(v) for v in per_q],
        coverage=[float(v) for v in coverage],
        coverage_error=float(np.mean(np.abs(coverage - TAUS))),
        crossing_rate=crossing,
    )


def window_pinball(pred: np.ndarray, target: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Per-series pinball loss, for paired comparisons across runs."""
    ordered = np.sort(np.asarray(pred), axis=-1)
    loss = _pinball(target, ordered, TAUS) / scales[:, None, None]
    return loss.mean(axis=(1, 2))


def baseline(contexts: np.ndarray, target: np.ndarray, scales: np.ndarray) -> Metrics:
    """Forecast that repeats the last context value for every step.

    The nine quantiles collapse to that point, so the pinball loss is the
    constant-forecast loss. Accepts ``(N, L)`` or ``(W, V, L)`` contexts.
    """
    point = np.asarray(contexts, dtype=np.float32)[..., -1].reshape(-1)
    flat_target = target.reshape(-1, target.shape[-1])
    flat_scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    pred = np.repeat(point[:, None, None], flat_target.shape[1], axis=1)
    pred = np.repeat(pred, len(TAUS), axis=2)
    return evaluate(pred, flat_target, flat_scales)


def deviations(pred: np.ndarray, ref: np.ndarray, scales: np.ndarray) -> dict:
    """Forecast shift from the fp32 reference, normalized by each series' context scale."""
    scale = scales[:, None, None]
    diff = (pred - ref) / scale
    per_q = np.mean(np.abs(diff), axis=(0, 1))
    bias_q = np.mean(diff, axis=(0, 1))
    return {
        "point_dev": float(np.mean(np.abs(diff[..., MEDIAN_INDEX]))),
        "mean_dev": float(np.mean(np.abs(diff))),
        "max_dev": float(np.max(np.abs(diff))),
        "per_quantile_dev": [float(v) for v in per_q],
        "per_quantile_bias": [float(v) for v in bias_q],
    }


def paired_draws(
    candidate: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray | None = None,
    draws: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """Relative change of two window losses, in percent, plus its draws.

    ``groups`` holds each window's forecast origin; windows of one origin are
    resampled together because they are correlated. Each draw recomputes both
    means and keeps the sampled origins' window counts, so the statistic stays
    a window mean.
    """
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if groups is None or len(groups) != len(candidate):
        groups = np.arange(len(candidate))
    _, origin_index = np.unique(np.asarray(groups), return_inverse=True)
    origin_count = int(origin_index.max()) + 1
    counts = np.bincount(origin_index, minlength=origin_count).astype(float)
    candidate_sum = np.bincount(origin_index, weights=candidate, minlength=origin_count)
    reference_sum = np.bincount(origin_index, weights=reference, minlength=origin_count)
    base_mean = float(reference.mean())
    point = 100.0 * (float(candidate.mean()) - base_mean) / max(base_mean, 1e-12)
    rng = rng if rng is not None else np.random.default_rng(0)
    picked = rng.integers(0, origin_count, (draws, origin_count))
    drawn_count = counts[picked].sum(axis=1)
    drawn_candidate = candidate_sum[picked].sum(axis=1) / drawn_count
    drawn_reference = reference_sum[picked].sum(axis=1) / drawn_count
    samples = (
        100.0 * (drawn_candidate - drawn_reference) / np.maximum(drawn_reference, 1e-12)
    )
    return point, samples


def compare(pred: np.ndarray, ref: np.ndarray, target: np.ndarray, scales: np.ndarray) -> dict:
    out = evaluate(pred, target, scales).to_dict()
    out.update(deviations(pred, ref, scales))
    ref_metrics = evaluate(ref, target, scales)
    out["ref_nmae"] = ref_metrics.nmae
    out["ref_pinball"] = ref_metrics.pinball
    out["nmae_delta_pct"] = 100.0 * (out["nmae"] - ref_metrics.nmae) / max(ref_metrics.nmae, 1e-12)
    out["pinball_delta_pct"] = (
        100.0 * (out["pinball"] - ref_metrics.pinball) / max(ref_metrics.pinball, 1e-12)
    )
    return out
