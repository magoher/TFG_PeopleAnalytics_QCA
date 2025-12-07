# core/fuzzy_logic.py
"""
Fuzzy Logic Utilities for QCA
-----------------------------
This module implements core fuzzy-set operations used in
Qualitative Comparative Analysis (QCA), including:

• Three-threshold calibration (Ragin 2008)
• Logistic calibration
• Complement / negation (~X)
• Automatic threshold extraction
• Utility helpers for fuzzy membership data

Author: TFG Marla — People Analytics QCA Platform
"""

import numpy as np
import pandas as pd





# ======================================================
# BASIC UTILITIES
# ======================================================

def clip01(values):
    """
    Ensures all fuzzy membership values stay within [0,1].
    """
    return np.clip(values, 0, 1)


def negate(series):
    """
    Fuzzy negation (~X):
    1 → 0
    0 → 1
    """
    return 1 - series.astype(float)


# ======================================================
# THREE-THRESHOLD CALIBRATION (Ragin 2008)
# ======================================================

def calibrate_three_thresholds(series, low, crossover, high):
    """
    Calibrates a raw variable into a fuzzy set using three
    qualitative anchors:

    • low       → full non-membership (≈0.0)
    • crossover → point of maximum ambiguity (0.5)
    • high      → full membership (≈1.0)

    Parameters
    ----------
    series : pd.Series
    low : float
    crossover : float
    high : float

    Returns
    -------
    np.ndarray (values between 0 and 1)
    """
    x = series.astype(float).values
    result = np.zeros_like(x, dtype=float)

    # below minimum → 0
    result[x <= low] = 0

    # above maximum → 1
    result[x >= high] = 1

    # between low and crossover
    mask_low = (x > low) & (x < crossover)
    result[mask_low] = (x[mask_low] - low) / (crossover - low)

    # between crossover and high
    mask_high = (x >= crossover) & (x < high)
    result[mask_high] = 0.5 + 0.5 * ((x[mask_high] - crossover) / (high - crossover))

    return clip01(result)


# ======================================================
# LOGISTIC CALIBRATION (continuous → fuzzy)
# ======================================================

def calibrate_logistic(series, crossover, steepness=1.5):
    """
    Logistic calibration (S-curve). Useful when:
    • the variable should grow smoothly
    • no discrete anchors exist

    Parameters
    ----------
    series : pd.Series
    crossover : float
    steepness : float, optional (default=1.5)
    """
    x = series.astype(float)
    scores = 1 / (1 + np.exp(-steepness * (x - crossover)))
    return clip01(scores)


# ======================================================
# AUTOMATIC THRESHOLD CALIBRATION
# ======================================================

def auto_calibrate(series, percentiles=(10, 50, 90)):
    """
    Automatically assigns thresholds using percentiles
    (default: 10th, 50th, 90th).

    Returns a fuzzy membership vector using three-threshold calibration.
    """
    low, cross, high = np.percentile(series.dropna(), percentiles)
    return calibrate_three_thresholds(series, low, cross, high)


# ======================================================
# UNIVERSAL CALIBRATION WRAPPER
# ======================================================

def calibrate_fuzzy(series, thresholds=None, method="three", steepness=1.5):
    """
    General fuzzy calibration function.

    Parameters
    ----------
    series : pd.Series
    thresholds : tuple or None
        If provided → (low, crossover, high)
        If None → automatic calibration is used
    method : str
        "three"     → three-threshold method
        "logistic"  → logistic S-curve
    steepness : float
        Only used for logistic calibration
    """
    series = series.astype(float)

    if method == "logistic":
        if thresholds is None:
            crossover = np.percentile(series, 50)
        else:
            crossover = thresholds[1]
        return calibrate_logistic(series, crossover, steepness)

    # Default: three-threshold calibration
    if thresholds is None:
        return auto_calibrate(series)

    low, cross, high = thresholds
    return calibrate_three_thresholds(series, low, cross, high)


# ======================================================
# HELPER: CHECK IF A COLUMN IS FUZZY
# ======================================================

def is_fuzzy(series):
    """
    Returns True if a column appears to represent fuzzy memberships.
    """
    return (
        series.min() >= 0
        and series.max() <= 1
        and series.dtype in [float, int]
    )
