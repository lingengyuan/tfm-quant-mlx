import numpy as np

from tfmlab.data import (
    build_covariate_windows,
    build_horizon_windows,
    build_multi_windows,
    build_windows,
    eval_windows,
    synthetic_set,
)


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


def test_covariate_windows_split_target_and_covariates():
    series_set = synthetic_set(length=4000)
    out = build_covariate_windows(series_set, 512, 64, 2)
    assert out is not None
    contexts, targets, scales, past_only, past_future = out
    assert contexts.shape == (2, 512)
    assert targets.shape == (2, 64)
    assert past_only.shape == (2, len(series_set) - 1, 512)
    assert past_future.shape == (2, 2, 512 + 64)
    assert (scales > 0).all()


def test_horizon_windows_share_contexts():
    series_set = synthetic_set(length=6000)
    out = build_horizon_windows(series_set, 512, (64, 256), 2)
    assert out is not None
    contexts, targets, scales, _ = out
    assert targets[64].shape == (contexts.shape[0], 64)
    assert targets[256].shape == (contexts.shape[0], 256)
    assert len(scales) == contexts.shape[0]


def test_window_offset_moves_origins_earlier():
    series_set = synthetic_set(length=1000)
    base = build_windows(series_set, 256, 64, 2)
    shifted = build_windows(series_set, 256, 64, 2, offset=2)
    assert shifted is not None
    assert base[0].shape == shifted[0].shape
    assert not np.array_equal(base[0], shifted[0])
    channel = shifted[3][-1]
    source = series_set.series[channel]
    end = len(source) - 64 - 2 * 64
    np.testing.assert_allclose(shifted[0][-1], source[end - 256 : end])
    np.testing.assert_allclose(shifted[1][-1], source[end : end + 64])
