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


def test_masked_covariate_decode_stays_finite():
    from tfmlab.runtime import predict_with_covariates

    model = _model()
    series = _series(1)[0]
    target = np.nan_to_num(series)
    target_mask = np.zeros(len(series), dtype=bool)
    target_mask[10] = True
    past_only = np.stack([series, series])
    past_only_mask = np.zeros_like(past_only, dtype=bool)
    past_only_mask[0, 5] = True
    index = np.arange(len(series) + 64)
    past_future = np.stack(
        [np.sin(index / 12.0), np.cos(index / 12.0)]
    ).astype(np.float32)
    pred = predict_with_covariates(
        model,
        target[None],
        past_only[None],
        past_future[None],
        64,
        1,
        target_mask=target_mask[None],
        past_only_mask=past_only_mask[None],
    )
    assert pred.shape == (1, 64, 9)
    assert np.isfinite(pred).all()


def test_masked_covariates_accept_a_batch_with_an_odd_context():
    from tfmlab.runtime import predict_with_covariates

    model = _model()
    context = 65
    series = np.sin(np.linspace(0.0, 40.0, context, dtype=np.float32))
    targets = np.stack([series, series])
    past_only = np.stack([np.stack([series, series])] * 2)
    past_future = np.zeros((2, 2, context + 8), dtype=np.float32)
    masks = np.zeros((2, context), dtype=bool)
    pred = predict_with_covariates(
        model,
        targets,
        past_only,
        past_future,
        8,
        2,
        target_mask=masks,
        past_only_mask=np.zeros_like(past_only, dtype=bool),
    )
    assert pred.shape == (2, 8, 9)


def test_bfloat16_export_roundtrip(tmp_path):
    from tfmlab.convert import load_quantized, save_quantized
    from tfmlab.weights import apply_weight_spec, weight_specs

    model = _model()
    spec = weight_specs(["bf16"])["bf16"]
    apply_weight_spec(model, spec)
    save_quantized(model, spec, str(tmp_path / "q"), "tiny")
    loaded = load_quantized(str(tmp_path / "q"))
    out = np.array(loaded.decode(mx.array(_series(1)[None]), 64))
    assert np.isfinite(out).all()


def test_quantized_export_roundtrip(tmp_path):
    from tfmlab.convert import load_quantized, save_quantized
    from tfmlab.weights import apply_weight_spec, weight_specs

    model = _model()
    spec = weight_specs(["int8_g64"])["int8_g64"]
    apply_weight_spec(model, spec)
    x = mx.array(_series(1)[None])
    reference = np.array(model.decode(x, 64))
    save_quantized(model, spec, str(tmp_path / "q"), "tiny")
    loaded = load_quantized(str(tmp_path / "q"))
    out = np.array(loaded.decode(x, 64))
    np.testing.assert_array_equal(out, reference)


def test_converted_model_rejects_a_different_spec(tmp_path):
    from tfmlab.convert import is_converted, load_quantized, save_quantized
    from tfmlab.weights import apply_weight_spec, weight_specs

    model = _model()
    spec = weight_specs(["int8_g64"])["int8_g64"]
    apply_weight_spec(model, spec)
    save_quantized(model, spec, str(tmp_path / "q"), "tiny")
    loaded = load_quantized(str(tmp_path / "q"))
    assert is_converted(str(tmp_path / "q"))
    apply_weight_spec(loaded, spec)
    with pytest.raises(ValueError):
        apply_weight_spec(loaded, weight_specs(["int4_g64"])["int4_g64"])


def test_predict_accepts_three_dimensional_inputs():
    from tfmlab.runtime import predict

    model = _model()
    contexts = np.stack([_series(3), _series(3)])
    out = predict(model, contexts, 64, 2)
    assert out.shape == (2, 3, 64, 9)
