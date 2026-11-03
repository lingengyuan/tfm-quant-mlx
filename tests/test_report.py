import json

from tfmlab import report


def _results() -> dict:
    macro = {
        "nmae": 0.4,
        "nmae_delta_pct": 0.0,
        "pinball": 0.16,
        "pinball_delta_pct": 0.0,
        "coverage_error": 0.04,
        "crossing_rate": 0.004,
        "per_quantile_dev": [0.1] * 9,
    }
    fp32 = {
        "name": "fp32",
        "kind": "dtype",
        "effective_bits": None,
        "spec": {"dtype": "float32"},
        "weight_bytes": 1323e6,
        "weight_ratio": 1.0,
        "accuracy": {"ETTh1": {"512": macro}},
        "macro": {"512": macro},
        "throughput": {"512": {"8": {"latency_ms_p50": 100.0, "series_per_s": 10.0}}},
        "memory": {"512": {"peak_mb": 2000.0}},
    }
    return {
        "meta": {
            "created": "2026-01-01T00:00:00",
            "elapsed_s": 1.0,
            "config": {"horizon": 64, "windows": 2, "eval_batch": 16, "mem_batch": 8},
        },
        "env": {
            "host": "local",
            "machine": "arm64",
            "platform": "test",
            "mlx": "0",
            "timesfm": "0",
            "python": "3.12",
            "checkpoint": "models/x",
            "param_count": 1,
        },
        "datasets": {"ETTh1": {"channels": 7, "length": 1000, "freq": "1h", "windows": {"512": 1}}},
        "weight_runs": [fp32, {"name": "int4_g64", "error": "boom"}],
        "kv_runs": [{"name": "affine4", "error": "boom"}],
        "baselines": {
            "ETTh1": {
                "512": {
                    "nmae": 0.5,
                    "pinball": 0.18,
                    "crossing_rate": 0.0,
                }
            }
        },
    }


def test_report_survives_failed_runs(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_results()))
    text = report.generate(str(path))
    assert "失败运行" in text
    assert "int4_g64: boom" in text


def test_report_includes_naive_baseline(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_results()))
    text = report.generate(str(path))
    assert "朴素基线" in text
    assert "| ETTh1 | 512 | 0.4000 | 0.5000 | -20.00% | 0.1600 | 0.1800 | -11.11% |" in text
