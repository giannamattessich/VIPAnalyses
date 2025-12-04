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
    return (freq >= low) & (freq <= high)

def normalize_per_freq(spect_power_db):
    """Z-score per frequency row (so color maps are comparable across days)."""
    mu = spect_power_db.mean(axis=1, keepdims=True)
    sd = spect_power_db.std(axis=1, keepdims=True) + 1e-12
    return (spect_power_db - mu)/sd

def summarize_bands(spect_power_db, f, bands=(("delta",1,4),("theta",6,10),("beta",15,30),("gamma",30,80))):
    out = {}
    for name, low, high in bands:
        m = get_band(f, low, high)
        out[name] = spect_power_db[m].mean(axis=0)   # time series of band power (dB)
    return out  # dict of name -> (T,)

def overall_metrics(Sdb, f):
    """Return robust scalars per day for easy cross-day lines/bars."""
    bb_mask = get_band(f, 1, 120)
    gamma_m = get_band(f, 30, 80)
    delta_m = get_band(f, 1, 4)
    theta_m = get_band(f, 4, 8)
    bb = Sdb[bb_mask].mean()                       # broadband dB
    gam = Sdb[gamma_m].mean()
    delt = Sdb[delta_m].mean()
    thet = Sdb[theta_m].mean()
    gdr = gam - delt                                # (≈ log10 gamma/delta ratio)
    return dict(broadband_db=bb, gamma_db=gam, delta_db=delt,
                gamma_delta_diff_db=gdr, theta_db=thet)

def plv(phi):
    return np.abs(np.mean(np.exp(1j*phi)))

def rayleigh_p(phi):
    n = len(phi)
    R = n * plv(phi)
    # Small-sample corrected approximation
    z = (R**2)/n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    return max(min(p,1.0), 0.0)

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

