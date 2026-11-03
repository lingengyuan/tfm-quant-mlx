import numpy as np
import pytest
from types import SimpleNamespace

from tfmlab import metrics
from tfmlab.attention import _KV_ATTR, set_attention_backend
from tfmlab.kv import PassThroughKV, make_kv


def test_baseline_repeats_last_value():
    contexts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    targets = np.array([[4.0, 5.0]], dtype=np.float32)
    scales = np.array([1.0], dtype=np.float32)
    score = metrics.baseline(contexts, targets, scales)
    assert score.nmae == pytest.approx(1.5)
    assert score.crossing_rate == 0.0


def _stub_model():
    layers = [SimpleNamespace(seq_attn=SimpleNamespace(), var_attn=SimpleNamespace())]
    return SimpleNamespace(transformer_stack=SimpleNamespace(layers=layers))


def test_sdpa_backend_installs_pass_through_when_uncompressed():
    model = _stub_model()
    set_attention_backend(model, "sdpa")
    kv = getattr(model.transformer_stack.layers[0].seq_attn, _KV_ATTR)
    assert isinstance(kv, PassThroughKV) and kv.fast


def test_manual_backend_keeps_the_original_path():
    model = _stub_model()
    set_attention_backend(model, "manual")
    assert getattr(model.transformer_stack.layers[0].seq_attn, _KV_ATTR, None) is None


def test_backend_switches_installed_compressor():
    model = _stub_model()
    kv = make_kv("affine4")
    setattr(model.transformer_stack.layers[0].seq_attn, _KV_ATTR, kv)
    set_attention_backend(model, "sdpa")
    assert kv.fast


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError):
        set_attention_backend(_stub_model(), "flash")
