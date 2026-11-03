import warnings

import numpy as np
import pandas as pd
import pytest

from tfmlab.predict import format_forecast, series_from_frame, stacked_series


def test_series_from_frame_takes_tail_and_drops_nan():
    frame = pd.DataFrame(
        {
            "date": list("abcde"),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [5.0, np.nan, 7.0, 8.0, 9.0],
        }
    )
    series = series_from_frame(frame, context=2)
    assert set(series) == {"a", "b"}
    assert series["a"].tolist() == [4.0, 5.0]
    assert series["b"].tolist() == [8.0, 9.0]


def test_stacked_series_drops_rows_across_all_columns():
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [0.0, np.nan, 3.0, 4.0, 5.0]}
    )
    names, contexts, length = stacked_series(frame, context=8)
    assert names == ["a", "b"]
    assert length == 4
    assert contexts.shape == (2, 4)
    # Rows 0, 2, 3, 4 survive; no column-local shifting.
    assert contexts[0].tolist() == [1.0, 3.0, 4.0, 5.0]
    assert contexts[1].tolist() == [0.0, 3.0, 4.0, 5.0]


def test_stacked_series_drops_non_finite_and_overflowed_rows():
    frame = pd.DataFrame(
        {
            "a": [1.0, np.inf, 3.0, 1e40, 5.0],
            "b": [0.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        names, contexts, length = stacked_series(frame, context=8)
    assert names == ["a", "b"]
    # inf and 1e40 (finite in float64, inf in float32) are dropped.
    assert length == 3
    assert contexts[0].tolist() == [1.0, 3.0, 5.0]


def test_format_forecast_rejects_non_finite_predictions():
    pred = np.zeros((2, 9), dtype=np.float32)
    pred[0, 0] = np.nan
    with pytest.raises(ValueError):
        format_forecast("s", pred)


def test_batched_forecasts_group_by_length(monkeypatch):
    import tfmlab.predict as predict_mod

    calls = []

    def fake(model, contexts, horizon, batch):
        calls.append((contexts.shape, batch))
        return np.zeros((len(contexts), horizon, 9), dtype=np.float32)

    monkeypatch.setattr(predict_mod, "predict", fake)
    series = {
        "a": np.arange(4, dtype=np.float32),
        "b": np.arange(5, dtype=np.float32),
        "c": np.arange(4, dtype=np.float32),
    }
    frames = predict_mod._batched_forecasts(None, series, 2, 8)
    assert [frame["series"].iloc[0] for frame in frames] == ["a", "b", "c"]
    assert calls == [((2, 4), 8), ((1, 5), 8)]


def test_format_forecast_sorts_quantiles():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(4, 9)).astype(np.float32)
    frame = format_forecast("s", pred)
    expected = np.sort(pred, axis=-1)
    quantiles = [f"q{tau}" for tau in ("0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9")]
    assert list(frame.columns) == ["series", "step", "point", *quantiles]
    assert frame["step"].tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(frame[quantiles].to_numpy(), expected)
    np.testing.assert_allclose(frame["point"].to_numpy(), expected[:, 4])
