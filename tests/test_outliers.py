from src.data import makeDataSet
from src.config import MAX_MISSING_RATE, OUTLIER_Z_THRESH, numerical_cols, MAX_OUTLIER_RATE
from src.validations import robust_outlier_rate

def test_outlier_rate_under_threshold_for_numeric_column():
    df = makeDataSet(seed=3, missing_rate=0.05, outlier_rate=0.01)
    numerical_cols_to_check = [ c for c in numerical_cols if c != "age" ]

    for col in numerical_cols_to_check:
        rate = robust_outlier_rate(df[col], z_thresh=OUTLIER_Z_THRESH)
        assert rate <= MAX_OUTLIER_RATE , (
            f"Outlier rate too high in '{col}': {rate:.3f}"
            f"(threshold={MAX_OUTLIER_RATE}, z_thresh={OUTLIER_Z_THRESH})"
        )

def test_outlier_gate_triggers_when_outliers_spike():
    df = makeDataSet(seed=3, outlier_rate=0.10)
    cols = [ c for c in numerical_cols if c != "age" ]

    assert any(
        robust_outlier_rate(df[col], z_thresh=OUTLIER_Z_THRESH) > MAX_OUTLIER_RATE
        for col in cols
    ), "Expected outlier gate to trigger but it did not."