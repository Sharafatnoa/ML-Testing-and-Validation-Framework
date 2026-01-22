from __future__ import annotations
import numpy as np
import pandas as pd
def robust_outlier_rate(series: pd.Series, z_thresh=6.0) -> float:
    """
    Return the fraction of values considered outliers using a robust z-score (MAD-based).

    Why MAD?
    - Mean/std can be distorted by outliers.
    - Median/MAD stays stable, so our gate is less flaky.

    Outlier rule:
        robust_z = 0.6745 * (x - median) / MAD
        outlier if |robust_z| > z_thresh

    Parameters
    ----------
    series : pd.Series
        Numeric series to evaluate.
    z_thresh : float
        Robust z-score threshold. 6.0 is a common conservative default.

    Returns
    -------
    float
        Outlier rate between 0.0 and 1.0
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if x.size == 0:
        return 0.00
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    if mad == 0:
        return 0.00
    robust_z = 0.6745 * (x - mad) / mad
    return float((np.abs(robust_z) > z_thresh ).mean())

