import pytest
import pandas as pd
from src.validations import robust_outlier_rate

def test_one_outlier_among_normal_values():
    series = pd.Series([100, 102, 98, 101, 99, 100, 103, 97, 100, 40])  # 40 is an outlier
    outlier_rate = robust_outlier_rate(series)
    assert outlier_rate == 0.1  # 1 out of 10 values is an outlier

def test_no_outliers():
    series = pd.Series([98, 99, 100, 101, 102])  # No outliers
    outlier_rate = robust_outlier_rate(series)
    assert outlier_rate == 0.0

def test_outliers_on_both_ends():
    series = pd.Series([-50, 98, 99, 100, 101, 102, 250])  # Both ends have outliers
    outlier_rate = robust_outlier_rate(series)
    assert outlier_rate == pytest.approx(2/7)  # 2 out of 7 values are outliers

def test_text_and_missing_values():
    series = pd.Series([100, 102, 98, 101, 99, "abc", None, 40])  # Non-numeric and missing values
    outlier_rate = robust_outlier_rate(series)
    assert outlier_rate == pytest.approx(1/6)  # 1 out of 6 numeric values is an outlier

def test_empty_series():
    series = pd.Series([])  # Empty series
    outlier_rate = robust_outlier_rate(series)
    assert outlier_rate == 0.0

@pytest.mark.xfail(reason="MAD is 0 when more than half the values are identical, so outliers are missed", strict=True)
def test_mad_zero():
    series = pd.Series([5, 5, 5, 5, 500])  # MAD is zero but 500 is an outlier
    outlier_rate = robust_outlier_rate(series)
    assert outlier_rate == pytest.approx(1/5)  # 1 out of 5 values is an outlier, function returns 0.0 due to MAD being zero