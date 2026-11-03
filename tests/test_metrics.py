import numpy as np
import pytest

from tfmlab.metrics import compare, evaluate, window_pinball


def test_perfect_forecast_has_zero_error():
    target = np.zeros((2, 8), dtype=np.float32)
    pred = np.zeros((2, 8, 9), dtype=np.float32)
    scales = np.ones(2, dtype=np.float32)
    m = evaluate(pred, target, scales)
    assert m.nmae == 0.0
    assert m.pinball == 0.0


def test_point_error_and_deviation_normalization():
    target = np.zeros((1, 4), dtype=np.float32)
    pred = np.zeros((1, 4, 9), dtype=np.float32)
    pred[...] = 1.0
    ref = np.zeros_like(pred)
    scales = np.array([2.0], dtype=np.float32)
    out = compare(pred, ref, target, scales)
    assert out["nmae"] == 0.5
    assert out["point_dev"] == 0.5
    assert out["nmae_delta_pct"] > 0


def test_crossing_detection_uses_raw_output():
    target = np.zeros((1, 2), dtype=np.float32)
    pred = np.zeros((1, 2, 9), dtype=np.float32)
    pred[..., 1] = -1.0  # sorted later; raw order crosses
    scales = np.ones(1, dtype=np.float32)
    m = evaluate(pred, target, scales)
    assert m.crossing_rate > 0


def test_window_pinball_averages_to_the_global_metric():
    rng = np.random.default_rng(0)
    target = rng.normal(size=(3, 4)).astype(np.float32)
    pred = rng.normal(size=(3, 4, 9)).astype(np.float32)
    scales = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    windows = window_pinball(pred, target, scales)
    assert windows.shape == (3,)
    assert windows.mean() == pytest.approx(evaluate(pred, target, scales).pinball)
