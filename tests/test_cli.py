import argparse

from tfmlab.cli import _load_config


def test_config_file_honors_cli_overrides(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "datasets: [ETTh1]\n"
        "weights: [fp32]\n"
        "contexts: [1024]\n"
        "windows: 2\n"
        "out: results/from_yaml.json\n"
    )
    args = argparse.Namespace(
        config=str(path),
        preset="quick",
        checkpoint=None,
        out="results/from_cli.json",
        windows=5,
        repeats=None,
        eval_batch=None,
    )
    cfg = _load_config(args)
    assert cfg.out == "results/from_cli.json"
    assert cfg.windows == 5
    assert cfg.contexts == (1024,)
