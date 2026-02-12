import os, numpy as np, traceback
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from lfp.readLFP import *
# def load_lfp(lfp_source):
#     """lfp_source: np.ndarray (T,C) or path to .lfp (memmap)"""
#     if isinstance(lfp_source, str):
#         # When you saved: shape was (out_frames, n_sel)
#         # We don’t know n_sel ahead of time, so we must be passed an array instead for memmap.
#         # If you know shape, adapt here. Otherwise just np.load if you saved .npy
#         raise ValueError("Pass arrays directly or adapt this loader with known shape.")
#     else:
#         arr = np.asarray(lfp_source)
#         if arr.ndim != 2:
#             raise ValueError("LFP must be 2D (time, channels).")
#         return arr

# either str (filepath ending in )
# return array if provided as array, else try to load provided .lfp file
    
def channel_reduce(arr, method="median"):
    """
    Reduce channels to a single 1D time-series.
    Accepts (T,), (T,C) or (C,T) and returns (T,).

    method: 'median' or 'mean'
    """
    x = np.asarray(arr)
    if method == "median":
        return np.median(x, axis=1)  # (T,)
    elif method == "mean":
        return np.mean(x, axis=1)    # (T,)
    else:
        raise ValueError("method must be 'median' or 'mean'")


def compute_power_spect_db(
        signal,
        fs,
        nperseg=2048,
        noverlap=1536,
        fmax=200,
        scaling='spectrum',
        window='hann',
        detrend=False
    ):
    """
    Compute a power spectrogram (in dB) for an LFP or motion signal.

    Parameters
    ----------
    signal : array-like
        1D voltage or motion signal.
    fs : float
        Sampling rate in Hz.
    nperseg : int
        Number of samples per FFT window.
    noverlap : int
        Number of overlapping samples between windows.
    fmax : float
        Maximum frequency to keep (Hz).
    scaling{ : ‘density’, ‘spectrum’ }, optional
        Whether to compute with PSD or PS
    window : str or tuple or array_like, optional
        Window function to use for computation
    detrend : bool
        Whether to detrend signal or not

    Returns
    -------
    freqs : array (Hz)
    times : array (s)
    power_db : 2D array (freqs × times), in dB
    """

    # Ensure correct shape
    #signal = np.asarray(signal).squeeze()

    # Compute spectrogram
    freqs, times, power = spectrogram(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling=scaling,
        detrend=detrend,
        window= window
    )

    # Convert to dB scale safely
    power_db = 10 * np.log10(np.maximum(power, 1e-20))

    # Limit to desired frequency range
    if fmax is not None:
        freq_mask = freqs <= fmax
        freqs = freqs[freq_mask]
        power_db = power_db[freq_mask, :]

    return freqs, times, power_db


def get_band(freq, low, high):
    freq = np.asarray(freq)
    return (freq >= low) & (freq <= high)


def _ensure_freq_first(Sdb, f):
    """
    Ensure Sdb is shaped (F, T), where F == len(f).
    If Sdb is (T, F), it will be transposed automatically.
    """
    Sdb = np.asarray(Sdb, dtype=float)
    f = np.asarray(f, dtype=float)

    if Sdb.ndim != 2:
        raise ValueError(f"Sdb must be 2D (F, T or T, F), got shape {Sdb.shape}")

    F = f.size
    if Sdb.shape[0] == F:
        # Already (F, T)
        return Sdb, f
    elif Sdb.shape[1] == F:
        # Likely (T, F) -> transpose
        return Sdb.T, f
    else:
        raise ValueError(
            f"None of Sdb's axes match len(f). Sdb.shape={Sdb.shape}, len(f)={F}"
        )


def normalize_per_freq(spect_power_db):
    """Z-score per frequency row (so color maps are comparable across days)."""
    Sdb = np.asarray(spect_power_db, dtype=float)

    # Replace ±inf with NaN so they don't break mean/std
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    mu = np.nanmean(Sdb, axis=1, keepdims=True)
    sd = np.nanstd(Sdb, axis=1, keepdims=True) + 1e-12
    return (Sdb - mu) / sd


def summarize_bands(
    spect_power_db,
    f,
    bands=(("delta", 0.5, 4),
           ("theta", 4, 8.5),
           ("alpha", 8.5, 13),
           ("beta", 15, 30),
           ("gamma", 30, 80),
           ("broadband", 0.5, 120)),
    normalize=True,
):
    """
    Summarize band power from a spectrogram.

    Parameters
    ----------
    spect_power_db : array-like, shape (F, T) or (T, F)
        Spectral power in dB.
    f : array-like, shape (F,)
        Frequency vector (Hz).
    bands : tuple
        Iterable of (name, low, high).
    normalize : bool
        If True, also compute per-band z-scored time series and store as '{band}_z'.

    Returns
    -------
    out : dict
        Keys: '{band}' -> band mean power (dB) over time (T,)
              and if normalize: '{band}_z' -> z-scored version over time (T,)
    """
    Sdb, f = _ensure_freq_first(spect_power_db, f)  # ensures (F, T)
    Sdb = np.asarray(Sdb, dtype=float)
    f = np.asarray(f, dtype=float)

    # Clean non-finite values (±inf -> NaN)
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    F, T = Sdb.shape
    out = {}

    for name, low, high in bands:
        m = get_band(f, low, high)

        if not np.any(m):
            band_ts = np.full(T, np.nan)
        else:
            band_slice = Sdb[m, :]  # (F_band, T)
            band_ts = np.nanmean(band_slice, axis=0) if not np.all(np.isnan(band_slice)) else np.full(T, np.nan)

        out[name] = band_ts

        if normalize:
            # z-score *within this band time series* (across time)
            mu = np.nanmean(band_ts)
            sd = np.nanstd(band_ts) + 1e-12
            out[f"{name}_z"] = (band_ts - mu) / sd

    return out


def overall_metrics(Sdb, f):
    """
    Return robust scalars for easy cross-day comparison.
    Accepts Sdb shaped (F, T) or (T, F); detects frequency axis.
    """
    Sdb, f = _ensure_freq_first(Sdb, f)  # Sdb -> (F, T)
    Sdb = np.where(np.isfinite(Sdb), Sdb, np.nan)

    def band_mean(low, high):
        m = get_band(f, low, high)
        if not np.any(m):
            return np.nan
        vals = Sdb[m, :]  # (F_band, T)
        if np.all(np.isnan(vals)):
            return np.nan
        return np.nanmean(vals)  # mean over freq + time

    bb   = band_mean(0.5, 120)   # broadband
    gam  = band_mean(30, 80)
    delt = band_mean(0.5, 4)
    thet = band_mean(4, 8.5)
    beta = band_mean(15,30)
    alpha = band_mean(8,12)

    # dB difference (≈ log10 gamma/delta ratio)
    gdr = gam - delt if np.isfinite(gam) and np.isfinite(delt) else np.nan

    # helper: convert dB → linear and compute ratio safely
    def ratio_from_db(num_db, den_db):
        if not (np.isfinite(num_db) and np.isfinite(den_db)):
            return np.nan
        num_lin = 10.0 ** (num_db / 10.0)
        den_lin = 10.0 ** (den_db / 10.0)
        if den_lin == 0:
            return np.nan
        return num_lin / den_lin

    delta_theta_ratio = ratio_from_db(delt, thet)
    delta_beta_ratio  = ratio_from_db(delt, beta)
    theta_gamma_ratio = ratio_from_db(thet, gam)

    return dict(
        broadband_db=bb,
        gamma_db=gam,
        delta_db=delt,
        theta_db=thet,
        beta_db=beta,
        alpha_db=alpha,
        gamma_delta_diff_db=gdr,
        delta_theta_ratio=delta_theta_ratio,
        delta_beta_ratio=delta_beta_ratio,
        theta_gamma_ratio=theta_gamma_ratio,)

def compute_spectrogram(lfp, fs, win_sec=4, overlap_sec=2, channel=0):
    """
    lfp: 1D (n_samples,) or 2D (n_samples, n_channels) or (n_channels, n_samples)
    fs : sampling rate
    """

    lfp = np.asarray(lfp)

    # --- ensure 1D signal ---
    if lfp.ndim == 2:
        # Decide which axis is samples vs channels
        # Heuristic: more samples than channels → samples axis is 0
        if lfp.shape[0] > lfp.shape[1]:
            # shape (n_samples, n_channels)
            sig = lfp[:, channel]
        else:
            # shape (n_channels, n_samples)
            sig = lfp[channel, :]
    elif lfp.ndim == 1:
        sig = lfp
    else:
        raise ValueError(f"lfp must be 1D or 2D, got shape {lfp.shape}")

    # --- window & overlap ---
    nperseg = int(win_sec * fs)
    if nperseg > len(sig):
        nperseg = len(sig)

    noverlap = int(overlap_sec * fs)
    if noverlap >= nperseg:
        noverlap = nperseg - 1

    # --- spectrogram ---
    f, t, Sxx = spectrogram(
        sig,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=None,         # let scipy choose
        scaling="density",
        mode="magnitude",
    )

    spec = Sxx.T  # time x freq
    return t, f, spec

import numpy as np

def db_to_linear_power(power_db):
    """
    Convert power from decibels (dB) to linear scale.

    Parameters
    ----------
    power_db : array-like or float
        Power in dB.

    Returns
    -------
    power_linear : array-like or float
        Power in linear units.
    """
    return 10.0 ** (power_db / 10.0)


def compute_linear_power_ratio(numerator_db, denominator_db, eps=1e-12):
    """
    Compute a power ratio in linear space given numerator and denominator in dB.

    Parameters
    ----------
    numerator_db : array-like
        Power in dB for the numerator band.
    denominator_db : array-like
        Power in dB for the denominator band.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    ratio_linear : array-like
        Linear power ratio (numerator / denominator).
    """
    numerator_linear = db_to_linear_power(numerator_db)
    denominator_linear = db_to_linear_power(denominator_db)

    return numerator_linear / (denominator_linear + eps)


def enforce_min_state_duration(state_sequence, minimum_duration=3):
    """
    Enforce a minimum duration for contiguous state segments.

    Short state bouts (< minimum_duration) are replaced with the
    preceding state if available, otherwise the following state.

    Parameters
    ----------
    state_sequence : array-like of int
        Discrete state labels over time.
    minimum_duration : int
        Minimum number of consecutive time bins required for a state.

    Returns
    -------
    cleaned_states : ndarray
        State sequence with short bouts removed.
    """
    cleaned_states = state_sequence.copy()
    num_timepoints = len(cleaned_states)

    start_idx = 0
    while start_idx < num_timepoints:
        end_idx = start_idx + 1

        # Find end of current contiguous state block
        while end_idx < num_timepoints and cleaned_states[end_idx] == cleaned_states[start_idx]:
            end_idx += 1

        bout_length = end_idx - start_idx

        # Replace short bouts
        if bout_length < minimum_duration:
            if start_idx > 0:
                replacement_state = cleaned_states[start_idx - 1]
            elif end_idx < num_timepoints:
                replacement_state = cleaned_states[end_idx]
            else:
                replacement_state = cleaned_states[start_idx]

            cleaned_states[start_idx:end_idx] = replacement_state

        start_idx = end_idx

    return cleaned_states

def resample_to_LFP(signal, signal_fs, n_lfp_samples, lfp_fs=1250):
    t_signal = np.arange(len(signal)) / signal_fs
    t_lfp    = np.arange(n_lfp_samples) / lfp_fs

    # interpolate signal onto LFP time grid
    signal_resampled = np.interp(t_lfp, t_signal, signal)

    return t_lfp, signal_resampled

def plv(phi):
    return np.abs(np.mean(np.exp(1j*phi)))

def rayleigh_p(phi):
    n = len(phi)
    R = n * plv(phi)
    # Small-sample corrected approximation
    z = (R**2)/n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    return max(min(p,1.0), 0.0)
