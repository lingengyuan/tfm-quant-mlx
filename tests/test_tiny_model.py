"""Patch parity and multivariate coverage without the 1.2 GB checkpoint.

A two-layer random model exercises the same attention code as the real one, so
CI can catch a broken patch even though the checkpoint is unavailable there.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("timesfm3.mlx")

from timesfm3.mlx import TimesFM3Mlx
from timesfm3.mlx.configs import TimesFM3MlxConfig


def _model():
    from tfmlab.attention import patch_class

    patch_class()
    cfg = TimesFM3MlxConfig(model_dims=64, hidden_dims=64, num_layers=2, num_heads=4)
    return TimesFM3Mlx(cfg, compile=False)


def _series(variates, context=256):
    values = np.sin(np.linspace(0.0, 40.0, variates * context, dtype=np.float32))
    return values.reshape(variates, context)


def test_pass_through_patch_matches_original_attention():
    from tfmlab.attention import install_kv
    from tfmlab.kv import PassThroughKV

    model = _model()
    x = mx.array(_series(1)[None])
    reference = np.array(model.decode(x, 64))
    install_kv(model, PassThroughKV())
    patched = np.array(model.decode(x, 64))
    np.testing.assert_array_equal(patched, reference)


def test_fast_attention_matches_manual_attention():
    from tfmlab.attention import install_kv
    from tfmlab.kv import PassThroughKV

    model = _model()
    x = mx.array(_series(1)[None])
    reference = np.array(model.decode(x, 64))
    kv = PassThroughKV()
    kv.fast = True
    install_kv(model, kv)
    fused = np.array(model.decode(x, 64))
    assert np.abs(fused - reference).max() < 5e-4


def test_int4_clipping_quantizes_tiny_model():
    from tfmlab.weights import apply_weight_spec, weight_specs

    model = _model()
    apply_weight_spec(model, weight_specs(["int4c95_g64"])["int4c95_g64"])
    out = np.array(model.decode(mx.array(_series(1)[None]), 64))
    assert out.shape == (1, 1, 64, 9)
    assert np.isfinite(out).all()


def test_install_kv_named_splits_scope_plan():
    from tfmlab.attention import _KV_ATTR, install_kv_named

    model = _model()
    compressors = install_kv_named(model, "affine4_seq_var8")
    layers = model.transformer_stack.layers
    assert len(compressors) == 2 * len(layers)
    assert getattr(layers[0].seq_attn, _KV_ATTR).bits == 4
    assert getattr(layers[0].var_attn, _KV_ATTR).bits == 8


def test_covariate_decode_returns_target_rows():
    from tfmlab.runtime import predict_with_covariates

    model = _model()
    series = _series(1)
    target = mx.array(series[None])
    past_only = mx.array(np.stack([series[0], series[0]])[None])
    past_future = mx.zeros((1, 2, 256 + 64), dtype=mx.float32)
    out = np.array(
        model.decode(
            target, 64, past_only_covariates=past_only, past_future_covariates=past_future
        )
    )
    # One target plus two past-only and two past-future covariates.
    assert out.shape == (1, 5, 64, 9)

    pred = predict_with_covariates(
        model,
        series,
        np.stack([series[0], series[0]])[None],
        np.zeros((1, 2, 256 + 64), dtype=np.float32),
        64,
        1,
    )
    assert pred.shape == (1, 64, 9)


def test_multivariate_decode_builds_variate_attention_kv():
    from tfmlab.attention import install_kv
    from tfmlab.kv import make_kv

    model = _model()
    kv = make_kv("affine8")
    install_kv(model, kv)
    out = np.array(model.decode(mx.array(_series(3)[None]), 64))
    assert out.shape == (1, 3, 64, 9)
    assert np.isfinite(out).all()

    shapes = {shape for _, shape in kv.stats.calls}
    # Sequence attention folds variates into batch; variate attention does the
    # reverse. Both must be present once the input has more than one variate.
    assert any(shape[0] == 3 and shape[1] > 1 for shape in shapes)
    assert any(shape[1] == 3 and shape[0] > 1 for shape in shapes)


def test_predict_accepts_three_dimensional_inputs():
    from tfmlab.runtime import predict

    model = _model()
    contexts = np.stack([_series(3), _series(3)])
    out = predict(model, contexts, 64, 2)
    assert out.shape == (2, 3, 64, 9)
