import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from tfmlab.kv import AffineKV, PassThroughKV, make_kv, make_kv_plan, make_scope_plans


@pytest.mark.parametrize(
    "name,limit",
    [
        ("affine8", 0.02),
        ("affine4", 0.18),
        ("turbo3_mse", 0.35),
        ("turbo3_prod", 0.40),
    ],
)
def test_roundtrip_error(name, limit):
    rng = np.random.default_rng(0)
    # Vectors shaped like attention heads: (batch, time, heads, head_dim).
    x = mx.array(rng.normal(size=(4, 16, 16, 80)).astype(np.float32))
    kv = make_kv(name)
    out = np.array(kv._roundtrip(x))
    reference = np.array(x)
    rel = np.linalg.norm(out - reference) / np.linalg.norm(reference)
    assert rel < limit, f"{name} relative error {rel:.3f}"


def test_qjl_inner_products_are_correlated():
    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(size=(512, 80)).astype(np.float32))
    kv = make_kv("qjl1")
    estimated = np.array(kv._roundtrip(x))
    queries = rng.normal(size=(200, 80)).astype(np.float32)
    true = queries @ np.array(x).T
    approx = queries @ estimated.T
    corr = np.corrcoef(true.ravel(), approx.ravel())[0, 1]
    # 1-bit QJL is an estimator, not a codec; only correlation is meaningful.
    assert corr > 0.5, f"correlation {corr:.3f}"


def test_affine_bits_include_metadata():
    assert make_kv("affine8").bits_k == pytest.approx(9.0)
    assert make_kv("affine4").bits_k == pytest.approx(5.0)


def test_qjl_leaves_values_alone():
    kv = make_kv("qjl1")
    assert kv.parts == ("k",)
    assert kv.bits_v == 16.0


def test_head_mask_keeps_other_heads_exact():
    rng = np.random.default_rng(2)
    x = mx.array(rng.normal(size=(2, 8, 4, 80)).astype(np.float32))
    kv = make_kv("affine4")
    kv.heads = (1,)
    out = np.array(kv.apply(x, "k"))
    reference = np.array(x)
    np.testing.assert_array_equal(out[:, :, [0, 2, 3]], reference[:, :, [0, 2, 3]])
    assert np.abs(out[:, :, 1] - reference[:, :, 1]).max() > 1e-4


def test_logical_bytes_scale_with_head_subset():
    x = mx.array(np.zeros((1, 4, 8, 80), dtype=np.float32))
    full = make_kv("affine4")
    full.apply(x, "k")
    part = make_kv("affine4")
    part.heads = (0, 1)
    part.apply(x, "k")
    assert part.logical_bytes() * 4 == full.logical_bytes()


def test_layer_plan_keeps_last_layer_fp16():
    plan = make_kv_plan("affine4_l19fp16", 20)
    assert len(plan) == 20
    assert all(isinstance(kv, PassThroughKV) == (i == 19) for i, kv in enumerate(plan))

    x = mx.array(np.zeros((1, 4, 16, 80), dtype=np.float32))
    for kv in plan:
        kv.apply(x, "k")
        kv.apply(x, "v")
    # Per layer: 4*16*80 values per part, two parts.
    assert sum(kv.logical_bytes() for kv in plan) == 19 * 6400 + 20480
    assert sum(kv.fp16_bytes() for kv in plan) == 20 * 20480


def test_mixed_kv_name_needs_a_plan():
    with pytest.raises(ValueError):
        make_kv("affine4_l19fp16")


def test_scope_plan_splits_attention_modes():
    plans = make_scope_plans("affine4_seq_var8", 20)
    assert [scope for scope, _ in plans] == ["seq", "var"]
    assert all(isinstance(kv, AffineKV) and kv.bits == 4 for kv in plans[0][1])
    assert all(isinstance(kv, AffineKV) and kv.bits == 8 for kv in plans[1][1])

    pass_through = make_scope_plans("affine4_seq_varfp16", 20)[1][1]
    assert all(isinstance(kv, PassThroughKV) for kv in pass_through)

    single = make_scope_plans("affine4", 20, "seq")
    assert [scope for scope, _ in single] == ["seq"]


def test_turbo_bits_count_the_padded_encoding():
    kv = make_kv("turbo3_mse")
    x = mx.array(np.zeros((1, 4, 2, 80), dtype=np.float32))
    kv.apply(x, "k")
    assert kv.bits_k == pytest.approx((127 * 3 + 32) / 80)
    assert kv.logical_bytes() == (1 * 4 * 2) * (127 * 3 + 32) / 8


def test_qjl_inner_products_are_unbiased():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(512, 80)).astype(np.float32)
    estimate = np.array(make_kv("qjl1")._roundtrip(mx.array(x)))
    queries = rng.normal(size=(200, 80)).astype(np.float32)
    true = (queries @ x.T).ravel()
    approx = (queries @ estimate.T).ravel()
    slope = float(np.cov(true, approx)[0, 1] / np.var(true))
    assert 0.9 < slope < 1.1, f"slope {slope:.3f}"


def test_turbo_prod_maps_zero_to_zero():
    out = np.array(
        make_kv("turbo3_prod")._roundtrip(mx.array(np.zeros((2, 80), dtype=np.float32)))
    )
    assert np.abs(out).max() < 1e-6
