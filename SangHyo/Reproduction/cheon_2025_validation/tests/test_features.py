import numpy as np
import pandas as pd

from cheon.features import clock_hours, parse_series


def test_parse_series_drops_trailing_separator():
    a = parse_series("1.2/0.9/1/")
    assert a.tolist() == [1.2, 0.9, 1.0]
    assert np.isnan(parse_series("1//2/")[1])


def test_clock_hours_local_kst():
    h = clock_hours(pd.Series(["2020-10-19T05:10:28+09:00", "2020-10-19T23:59:00+09:00"]))
    assert abs(h.iloc[0] - (5 + 10 / 60 + 28 / 3600)) < 1e-9 and abs(h.iloc[1] - 23.983333) < 1e-5
