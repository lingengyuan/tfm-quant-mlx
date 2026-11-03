import json

import pytest

from tfmlab import compare


def _entry(pinballs, pinball):
    return {
        "pinball": pinball,
        "window_pinball": pinballs,
        "window_group": [0, 0, 1, 1],
    }


def _run(name, accuracy, kv=True):
    run = {"name": name, "accuracy": accuracy}
    if kv:
        run["kv_scope"] = "both"
    return run


def _data() -> dict:
    return {
        "weight_runs": [],
        "kv_runs": [
            _run("affine4", {"ETTh1": {"15360": _entry([0.10] * 4, 0.10)}}),
            _run("l19", {"ETTh1": {"15360": _entry([0.09] * 4, 0.09)}}),
        ],
    }


def test_compare_reports_the_paired_delta(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_data()))
    text = compare.render(str(path), "affine4")
    assert "| l19 | ETTh1 | 15360 | 0.1000 | 0.0900 | -10.00% |" in text
    assert "通用推荐" in text


def test_compare_degrades_to_conditional_when_one_domain_worsens(tmp_path):
    data = _data()
    data["kv_runs"][1]["accuracy"]["ETTh2"] = {
        "15360": _entry([0.21] * 4, 0.21)
    }
    data["kv_runs"][0]["accuracy"]["ETTh2"] = {
        "15360": _entry([0.20] * 4, 0.20)
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data))
    text = compare.render(str(path), "affine4")
    assert "1/2 个数据集" in text
    assert "条件推荐" in text


def test_compare_skips_the_verdict_for_a_weight_run(tmp_path):
    data = _data()
    data["weight_runs"] = [_run("fp32", {"ETTh1": {"15360": _entry([0.098] * 4, 0.098)}}, kv=False)]
    path = tmp_path / "r.json"
    path.write_text(json.dumps(data))
    text = compare.render(str(path), "affine4")
    assert "fp32 不在判定范围" in text


def test_compare_rejects_an_unknown_base(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(_data()))
    with pytest.raises(ValueError):
        compare.render(str(path), "missing")
