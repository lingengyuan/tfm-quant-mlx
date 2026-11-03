import json

from tfmlab import attention_compare


def _run(name: str, pinball: float, nmae: float) -> dict:
    entry = {
        "nmae": nmae,
        "pinball": pinball,
        "nmae_delta_pct": 0.0,
        "pinball_delta_pct": 0.0,
    }
    return {
        "name": name,
        "accuracy": {"ETTh1": {"15360": entry}},
        "macro": {"15360": entry},
    }


def test_compare_reports_relative_change(tmp_path):
    manual = tmp_path / "manual.json"
    sdpa = tmp_path / "sdpa.json"
    manual.write_text(json.dumps({"weight_runs": [_run("fp32", 0.10, 0.30)]}))
    sdpa.write_text(json.dumps({"weight_runs": [_run("fp32", 0.11, 0.30)]}))
    text = attention_compare.render(str(manual), str(sdpa))
    assert "| fp32 | ETTh1 | 15360 | 0.1000 | 0.1100 | +10.00% |" in text
    assert "0.30" in text
