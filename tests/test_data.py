import numpy as np

from tfmlab.data import build_multi_windows, build_windows, eval_windows, synthetic_set


def test_window_shapes_and_targets():
    ss = synthetic_set(length=4000)
    out = build_windows(ss, context=512, horizon=64, num_windows=3)
    assert out is not None
    contexts, targets, scales, channels = out
    assert contexts.shape[1] == 512
    assert targets.shape[1] == 64
    assert contexts.shape[0] == targets.shape[0] == len(scales) == len(channels)
    assert np.isfinite(contexts).all() and np.isfinite(targets).all()
    assert (scales > 0).all()


def test_too_long_context_is_skipped():
    ss = synthetic_set(length=500)
    assert build_windows(ss, context=512, horizon=64, num_windows=2) is None


def test_windows_do_not_overrun_series():
    ss = synthetic_set(length=1000)
    contexts, targets, _, _ = build_windows(ss, context=256, horizon=64, num_windows=4)
    for frame in list(contexts) + list(targets):
        assert np.isfinite(frame).all()
    # The last target must exist; build_windows slices s[end:end+horizon].
    assert targets.shape[1] == 64


def test_eval_windows_stacks_synthetic_sets():
    contexts, targets, scales = eval_windows("data/raw", (), True, 512, 64, 2)
    assert contexts.shape[1] == 512
    assert targets.shape[1] == 64
    assert len(contexts) == len(targets) == len(scales) == 16


def test_multi_windows_keep_variates_together():
    series_set = synthetic_set(length=4000)
    out = build_multi_windows(series_set, 512, 64, 2)
    assert out is not None
    contexts, targets, scales, channels = out
    assert contexts.shape == (2, len(channels), 512)
    assert targets.shape == (2, len(channels), 64)
    assert scales.shape == (2, len(channels))
    assert (scales > 0).all()
