from tfmlab.runner import _weight_order


def test_fp32_moves_to_the_front():
    assert _weight_order(("int8_g64", "fp32", "int4_g64")) == [
        "fp32",
        "int8_g64",
        "int4_g64",
    ]


def test_fp32_is_inserted_when_absent():
    assert _weight_order(("int8_g64",)) == ["fp32", "int8_g64"]
