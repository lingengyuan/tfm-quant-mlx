import pytest

from tfmlab.weights import logical_quant_bits, weight_specs


def test_uniform_specs_have_bits():
    specs = weight_specs(["fp16", "int8_g64", "int4_g128"])
    assert specs["fp16"].kind == "dtype"
    assert logical_quant_bits(specs["int8_g64"]) == pytest.approx(9.0)
    assert logical_quant_bits(specs["int4_g128"]) == pytest.approx(4.5)


def test_mixed_specs_report_no_single_bit_width():
    for name in ("mix_l19_8", "mix_l18_19_8", "mix_top3_8"):
        spec = weight_specs([name])[name]
        assert spec.kind == "mixed"
        assert logical_quant_bits(spec) is None
        assert spec.overrides


def test_unknown_spec_is_rejected():
    with pytest.raises(ValueError):
        weight_specs(["int3"])
