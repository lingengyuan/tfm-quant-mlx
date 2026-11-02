import numpy as np

from tfmlab.parity import compare


def _arrays():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(3, 8, 9)).astype(np.float32)
    targets = rng.normal(size=(3, 8)).astype(np.float32)
    scales = np.asarray([1.0, 2.0, 0.5], dtype=np.float32)
    return pred, targets, scales


def test_identical_backends_are_zero():
    pred, targets, scales = _arrays()
    m = compare(pred, pred.copy(), targets, scales)
    assert m["max_abs_dev"] == 0.0
    assert m["mean_abs_dev"] == 0.0
    assert m["pinball_delta_pct"] == 0.0
    assert m["correlation"] == 1.0


def test_offset_shows_deviation_after_scaling():
    pred, targets, scales = _arrays()
    shifted = pred + scales[:, None, None]
    m = compare(pred, shifted, targets, scales)
    assert abs(m["mean_abs_dev"] - 1.0) < 1e-6
    assert m["correlation"] < 1.0
