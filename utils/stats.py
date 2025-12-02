import numpy as np, pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.signal import correlate

'''
Functions to run generic data transformations -> normalization and calculations of data
'''
def min_max_norm(value_arr):
    """
    Generic min-max norm function to perform on array'

    Parameters:
    -----------
    value_arr : array-like
        Data to normalize 

    Returns:
    --------
    min_max_normalized : np.ndarray
        Z-scored input array.    

    """
    value_arr = np.asarray(value_arr)
    min_val, max_val = np.min(value_arr), np.max(value_arr)
    min_max_normalized = (value_arr - min_val) / (max_val - min_val)
    return min_max_normalized

def cross_correlation(a, b, fs_hz, max_lag_s=10, axis=1):
    """
    Normalized cross-correlation and lags (seconds).
    a, b must be same length and sampled at fs_hz.
    """
    a = zscore_robust(a, axis=axis)
    b = zscore_robust(b, axis=axis)
    n = len(a)
    xcorr_full = correlate(a, b, mode='full') / n
    lags = np.arange(-n+1, n) / fs_hz

    # Keep only |lag| <= max_lag_s
    keep = np.abs(lags) <= max_lag_s
    return lags[keep], xcorr_full[keep]

def zscore_robust(value_arr, axis=1):
    """
    Compute robust z-scores using the median and MAD (median absolute deviation).

    Parameters
    ----------
    value_arr : array-like
        Input data (2D or 1D).
    axis : int, default=1
        Axis along which to compute the z-score.
        - axis=1 → row-wise (default)
        - axis=0 → column-wise

    Returns
    -------
    z_score : np.ndarray
        Robustly z-scored array, same shape as value_arr.
    """
    a = np.asarray(value_arr, float)
    median = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(a - median), axis=axis, keepdims=True) + 1e-12
    z_score = (a - median) / (1.4826 * mad)
    return z_score

def movquant(signal, q=0.1, window=300, omitnan=True, truncate=True):
    """
    Python implementation of MATLAB's 'movquant' function. Compute a moving quantile over a 1D signal.

    Parameters:
    -----------
    signal : array-like
        Input 1D signal.
    q : float
        Quantile to compute (between 0 and 1).
    window : int
        Window length in samples.
    omitnan : bool
        Whether to ignore NaNs (True = skip NaNs).
    truncate : bool
        Whether to truncate edges (True = pad edges with NaNs).

    Returns:
    --------
    filtered : np.ndarray
        Output signal after moving quantile filtering.
    """
    series = pd.Series(signal)
    if omitnan:
        quant = series.rolling(window=window, center=True, min_periods=1).quantile(q)
    else:
        # this will propagate NaNs if present in the window
        quant = series.rolling(window=window, center=True).apply(
            lambda x: np.nan if np.any(np.isnan(x)) else np.quantile(x, q), raw=False)
    if not truncate:
        quant = quant.fillna(method='bfill').fillna(method='ffill')
    return quant.to_numpy()

def compare_to_normal_distribution(data):
    '''
    Plot data histogram and QQPlot to test normality/distribution of data

    '''
    # 1. Visual Inspection: Histogram
    plt.hist(data, bins=50, density=True, alpha=0.6, color='g')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

    # 1. Visual Inspection: Q-Q Plot
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    plt.show()

    # 2. Statistical Test: Shapiro-Wilk Test
    shapiro_statistic, shapiro_pvalue = stats.shapiro(data)
    print(f"Shapiro-Wilk Test: Statistic={shapiro_statistic:.3f}, p-value={shapiro_pvalue:.3f}")

    # 3. Descriptive Statistics: Skewness and Kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data, fisher=True) # Fisher's definition
    print(f"Skewness: {skewness:.3f}")
    print(f"Kurtosis: {kurtosis:.3f}")

    