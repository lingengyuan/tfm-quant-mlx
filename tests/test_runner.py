from tfmlab.runner import _aggregate, _weight_order


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
