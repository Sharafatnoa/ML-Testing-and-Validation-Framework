from src.data import makeDataSet
from src.config import MAX_MISSING_RATE

def missing_rate_under_threshold():
    df = makeDataSet(seed=2)
    max_missing = df.isna().mean().max()
    assert max_missing <= MAX_MISSING_RATE, f"Missing Rate is Too high: {max_missing:.3f}"
