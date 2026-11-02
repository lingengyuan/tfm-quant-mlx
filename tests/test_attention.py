import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

CHECKPOINT = os.environ.get("TFMQ_CHECKPOINT", "models/timesfm-3.0")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT), reason="local TimesFM checkpoint is required"
)


def test_no_kv_patch_is_bit_identical():
    from tfmlab.attention import install_kv, patch_class
    from tfmlab.runtime import load_model

    patch_class()
    model = load_model(CHECKPOINT)
    x = mx.array(np.sin(np.linspace(0, 40, 256, dtype=np.float32))[None, None, :])
    reference = np.array(model.decode(x, 64))
    model._compiled_forward = None
    install_kv(model, None)
    patched = np.array(model.decode(x, 64))
    np.testing.assert_array_equal(patched, reference)


def test_affine8_kv_is_finite_and_close():
    from tfmlab.attention import install_kv
    from tfmlab.kv import make_kv
    from tfmlab.runtime import load_model

    model = load_model(CHECKPOINT)
    x = mx.array(np.sin(np.linspace(0, 40, 256, dtype=np.float32))[None, None, :])
    reference = np.array(model.decode(x, 64))
    install_kv(model, make_kv("affine8"))
    out = np.array(model.decode(x, 64))
    assert np.isfinite(out).all()
    assert np.abs(out - reference).max() < 0.05


def test_install_kv_restricts_layers_and_scope():
    from tfmlab.attention import _KV_ATTR, install_kv
    from tfmlab.kv import make_kv
    from tfmlab.runtime import load_model

    model = load_model(CHECKPOINT)
    kv = make_kv("affine4")
    install_kv(model, kv, "seq", layers=[3])
    layers = model.transformer_stack.layers
    assert getattr(layers[3].seq_attn, _KV_ATTR, None) is kv
    assert getattr(layers[2].seq_attn, _KV_ATTR, None) is None
    assert getattr(layers[3].var_attn, _KV_ATTR, None) is None


def test_install_kv_scopes_leaves_other_scope_intact():
    from tfmlab.attention import _KV_ATTR, install_kv_scopes
    from tfmlab.kv import make_kv
    from tfmlab.runtime import load_model

    model = load_model(CHECKPOINT)
    kv_seq = make_kv("affine4")
    kv_var = make_kv("affine8")
    install_kv_scopes(model, [2], {"seq": kv_seq})
    install_kv_scopes(model, [2], {"var": kv_var})
    layer = model.transformer_stack.layers[2]
    assert getattr(layer.seq_attn, _KV_ATTR) is kv_seq
    assert getattr(layer.var_attn, _KV_ATTR) is kv_var


def test_multivariate_decode_builds_variate_attention_kv():
    from tfmlab.attention import install_kv
    from tfmlab.kv import make_kv
    from tfmlab.runtime import load_model, predict

    model = load_model(CHECKPOINT)
    kv = make_kv("affine8")
    install_kv(model, kv)
    series = np.sin(np.linspace(0, 40, 3 * 256, dtype=np.float32)).reshape(3, 256)
    x = mx.array(series[None])
    out = np.array(model.decode(x, 64))
    assert out.shape == (1, 3, 64, 9)
    assert np.isfinite(out).all()
    shapes = {shape for _, shape in kv.stats.calls}
    # Sequence attention folds variates into the batch axis; variate attention
    # uses patches as the batch axis and the variates as the sequence.
    assert any(shape[0] == 3 and shape[1] > 1 for shape in shapes)
    assert any(shape[1] == 3 and shape[0] > 1 for shape in shapes)

    batched = predict(model, np.stack([series, series]), 64, 2)
    assert batched.shape == (2, 3, 64, 9)
