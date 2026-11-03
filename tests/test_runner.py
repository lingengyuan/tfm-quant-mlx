import numpy as np

from tfmlab.data import SYNTHETIC_LENGTH, synthetic_set
from tfmlab.runner import RunConfig, _aggregate, _load_data, _stacked_groups, _weight_order


def test_fp32_moves_to_the_front():
    assert _weight_order(("int8_g64", "fp32", "int4_g64")) == [
        "fp32",
        "int8_g64",
        "int4_g64",
    ]


def test_fp32_is_inserted_when_absent():
    assert _weight_order(("int8_g64",)) == ["fp32", "int8_g64"]


def test_aggregate_skips_per_window_arrays_with_different_lengths():
    entries = [
        {"nmae": 0.4, "window_pinball": [0.1] * 32, "window_group": [0] * 32},
        {"nmae": 0.6, "window_pinball": [0.2] * 28, "window_group": [0] * 28},
    ]
    out = _aggregate(entries)
    assert out["nmae"] == 0.5
    assert "window_pinball" not in out
    assert "window_group" not in out


def _cfg(**overrides) -> RunConfig:
    base = dict(datasets=(), synthetic=True, contexts=(256,), windows=2, horizon=64)
    base.update(overrides)
    return RunConfig(**base)


def test_multi_evaluation_follows_tail_windows():
    _, eval_data = _load_data(_cfg(multi=True, window_mode="tail"), (256,))
    data = eval_data[("synthetic", 256)]
    series = synthetic_set()
    ends = [SYNTHETIC_LENGTH - 128, SYNTHETIC_LENGTH - 64]
    assert data.groups.tolist() == ends
    np.testing.assert_allclose(
        data.contexts[-1, 0], series.series[0][ends[-1] - 256 : ends[-1]]
    )


def test_offset_reserves_recent_windows():
    _, eval_data = _load_data(_cfg(window_mode="tail", offset=2), (256,))
    data = eval_data[("synthetic", 256)]
    reserved = SYNTHETIC_LENGTH - 64 - 2 * 64
    assert sorted(set(data.groups.tolist())) == [reserved - 64, reserved]


def test_group_ids_repeat_over_variates():
    _, eval_data = _load_data(_cfg(multi=True), (256,))
    data = eval_data[("synthetic", 256)]
    groups = _stacked_groups(data)
    variates = data.contexts.shape[1]
    assert groups.tolist() == np.repeat(data.groups, variates).tolist()
    assert len(np.unique(groups)) == len(data.groups)
