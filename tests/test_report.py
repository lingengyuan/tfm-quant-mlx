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


def test_report_computes_paired_bootstrap_interval(tmp_path):
    data = _results()
    fp32 = data["weight_runs"][0]
    fp32["accuracy"]["ETTh1"]["512"]["window_pinball"] = [0.16, 0.17, 0.15]
    quantized = json.loads(json.dumps(fp32))
    quantized.update(
        {
            "name": "int8_g64",
            "kind": "quant",
            "weight_bytes": 375e6,
            "weight_ratio": 3.5,
            "accuracy": {
                "ETTh1": {
                    "512": {
                        **fp32["accuracy"]["ETTh1"]["512"],
                        "window_pinball": [0.17, 0.18, 0.16],
                    }
                }
            },
        }
    )
    data["weight_runs"] = [fp32, quantized]
    data["kv_runs"] = []
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    text = report.generate(str(path))
    assert "配对差异" in text
    assert "int8_g64" in text
    assert "95% CI" in text


def _paired_data() -> dict:
    def entry(pinballs, groups):
        metrics = {
            "nmae": 0.4,
            "nmae_delta_pct": 0.0,
            "pinball": sum(pinballs) / len(pinballs),
            "pinball_delta_pct": 0.0,
            "coverage_error": 0.04,
            "crossing_rate": 0.0,
            "per_quantile_dev": [0.1] * 9,
            "window_pinball": pinballs,
            "window_group": groups,
        }
        return metrics

    reference = {"ETTh1": {}, "ETTh2": {}}
    quantized = {"ETTh1": {}, "ETTh2": {}}
    for context in ("512",):
        reference["ETTh1"][context] = entry([0.1] * 4, [0, 0, 1, 1])
        reference["ETTh2"][context] = entry([0.1] * 4, [0, 0, 1, 1])
        # ETTh1: +10%; ETTh2: -1%; the macro delta is +4.5%.
        quantized["ETTh1"][context] = entry([0.11] * 4, [0, 0, 1, 1])
        quantized["ETTh2"][context] = entry([0.099] * 4, [0, 0, 1, 1])

    def run(name, accuracy):
        return {
            "name": name,
            "kind": "quant",
            "spec": {"dtype": "float32"},
            "weight_bytes": 1.3e9,
            "weight_ratio": 1.0,
            "accuracy": accuracy,
            "macro": {},
            "throughput": {},
            "memory": {},
        }

    return {
        "meta": {"created": "x", "elapsed_s": 1, "config": {"horizon": 64}},
        "env": {},
        "datasets": {},
        "weight_runs": [run("fp32", reference), run("int8_g64", quantized)],
        "kv_runs": [],
        "baselines": {},
    }


def test_paired_ci_matches_the_macro_delta(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(_paired_data()))
    text = report.generate(str(path))
    row = next(line for line in text.splitlines() if "| int8_g64 | 512 |" in line)
    assert "+4.50%" in row
    assert "[+4.50%, +4.50%]" in row


def test_paired_ci_resamples_origins_not_channels(tmp_path):
    data = _paired_data()
    quantized = data["weight_runs"][1]
    # One origin is 10% worse, the other is unchanged; two channels per origin.
    quantized["accuracy"] = {
        "ETTh1": {
            "512": {
                **quantized["accuracy"]["ETTh1"]["512"],
                "window_pinball": [0.11, 0.11, 0.10, 0.10],
            }
        },
        "ETTh2": {
            "512": {
                **quantized["accuracy"]["ETTh2"]["512"],
                "window_pinball": [0.11, 0.11, 0.10, 0.10],
            }
        },
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    text = report.generate(str(path))
    row = next(line for line in text.splitlines() if "| int8_g64 | 512 |" in line)
    assert "+5.00%" in row
    assert "[+0.00%, +10.00%]" in row


def test_paired_ci_weights_origins_by_window_count(tmp_path):
    data = _paired_data()
    quantized = data["weight_runs"][1]
    # Origin 0 keeps three channels and origin 1 keeps one. The window mean is
    # +25%; an origin mean would report +50%.
    quantized["accuracy"] = {
        "ETTh2": {},
        "ETTh1": {
            "512": {
                **quantized["accuracy"]["ETTh1"]["512"],
                "window_pinball": [0.10, 0.10, 0.10, 0.20],
                "window_group": [0, 0, 0, 1],
                "pinball": 0.125,
            }
        },
    }
    reference = data["weight_runs"][0]
    reference["accuracy"] = {
        "ETTh2": {},
        "ETTh1": {
            "512": {
                **reference["accuracy"]["ETTh1"]["512"],
                "window_pinball": [0.10, 0.10, 0.10, 0.10],
                "window_group": [0, 0, 0, 1],
            }
        },
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    text = report.generate(str(path))
    row = next(line for line in text.splitlines() if "| int8_g64 | 512 |" in line)
    assert "| +25.00% |" in row


def test_paired_ci_resamples_the_reference_mean(tmp_path):
    data = _paired_data()
    reference = data["weight_runs"][0]
    quantized = data["weight_runs"][1]
    # Origin 0 sits at 0.10 and origin 1 at 0.20; only origin 0 changes. A drawn
    # pair must divide by the sampled reference mean, not the fixed one.
    for entry, pinballs in (
        (reference, [0.10, 0.10, 0.20, 0.20]),
        (quantized, [0.11, 0.11, 0.20, 0.20]),
    ):
        both = {"ETTh1": {"512": {**entry["accuracy"]["ETTh1"]["512"], "window_pinball": pinballs}}}
        both["ETTh2"] = both["ETTh1"]
        entry["accuracy"] = both
    path = tmp_path / "results.json"
    path.write_text(json.dumps(data))
    text = report.generate(str(path))
    row = next(line for line in text.splitlines() if "| int8_g64 | 512 |" in line)
    assert "| +3.33% |" in row
    assert "[+0.00%, +10.00%]" in row
