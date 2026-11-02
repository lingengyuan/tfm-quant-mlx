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
