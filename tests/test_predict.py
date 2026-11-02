import numpy as np
import pandas as pd

from tfmlab.predict import format_forecast, series_from_frame


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
