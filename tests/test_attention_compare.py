import json

import pytest

from tfmlab import attention_compare


def _run(name: str, pinball: float, nmae: float) -> dict:
    entry = {
        "nmae": nmae,
        "pinball": pinball,
        "nmae_delta_pct": 0.0,
        "pinball_delta_pct": 0.0,
        "window_pinball": [pinball] * 4,
    }
    return {
        "name": name,
        "accuracy": {"ETTh1": {"15360": entry}},
        "macro": {"15360": entry},
    }


def _file(attention: str, windows: int = 4, pinball: float = 0.10) -> dict:
    return {
        "meta": {
            "config": {
                "checkpoint": "models/x",
                "datasets": ["ETTh1"],
                "synthetic": False,
                "max_channels": 0,
                "contexts": [15360],
                "windows": windows,
                "horizon": 64,
                "weights": ["fp32"],
                "kv": ["affine4"],
                "kv_contexts": [15360],
                "kv_weights": ["fp32"],
                "kv_scopes": ["both"],
                "multi": False,
                "eval_batch": 16,
                "attention": attention,
            }
        },
        "weight_runs": [_run("fp32", pinball, 0.30)],
    }


def test_compare_reports_relative_change(tmp_path):
    manual = tmp_path / "manual.json"
    sdpa = tmp_path / "sdpa.json"
    manual.write_text(json.dumps(_file("manual")))
    sdpa.write_text(json.dumps(_file("sdpa", pinball=0.11)))
    text = attention_compare.render(str(manual), str(sdpa))
    assert "| fp32 | ETTh1 | 15360 | 0.1000 | 0.1100 | +10.00% |" in text


def test_compare_rejects_different_windows(tmp_path):
    manual = tmp_path / "manual.json"
    sdpa = tmp_path / "sdpa.json"
    manual.write_text(json.dumps(_file("manual", windows=6)))
    sdpa.write_text(json.dumps(_file("sdpa", windows=4)))
    with pytest.raises(ValueError):
        attention_compare.render(str(manual), str(sdpa))


def test_compare_rejects_two_manual_files(tmp_path):
    manual = tmp_path / "manual.json"
    other = tmp_path / "other.json"
    manual.write_text(json.dumps(_file("manual")))
    other.write_text(json.dumps(_file("manual", pinball=0.11)))
    with pytest.raises(ValueError):
        attention_compare.render(str(manual), str(other))
