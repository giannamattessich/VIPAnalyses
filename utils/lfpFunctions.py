import os, numpy as np, traceback
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from utils.readLFP import *
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
    

import numpy as np

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

# def compute_power_spect_db(signal, fs, nperseg=2048, noverlap=1536, fmax=None):
#     """
#     Compute a power spectrogram (in dB) for a 1D signal.
#     """
#     # Ensure nperseg does not exceed signal length
#     nperseg = min(nperseg, len(signal))

#     # Ensure noverlap < nperseg
#     if noverlap >= nperseg:
#         noverlap = int(0.75 * nperseg)  # 75% overlap default
#         # Ensure still strictly less than nperseg
#         noverlap = max(0, min(noverlap, nperseg - 1))
#     print(f"Using nperseg={nperseg}, noverlap={noverlap}, signal_length={len(signal)}")

#     freqs, times, spect_power = spectrogram(
#         signal,
#         fs=fs,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         scaling='spectrum',
#         detrend=False
#     )

#     # Convert to dB
#     spect_power_db = 10 * np.log10(np.maximum(spect_power, 1e-20))

#     # Limit frequency axis
#     if fmax is not None:
#         mask = freqs <= fmax
#         freqs = freqs[mask]
#         spect_power_db = spect_power_db[mask]

#     return freqs, times, spect_power_db


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

def compare_days_spectrograms(day2lfp, fs_lfp, chan_reduce="median",
                              nperseg=2048, noverlap=1536, fmax=200, show=True):
    """
    day2lfp: dict like {'day1': lfp_tc, 'day2': lfp_tc, ...} with LFP as (time, channels)
    fs_lfp : sampling rate of the LFP file (e.g., 1250)
    """
    # Prepare results
    days = list(day2lfp.keys())
    spec_norm = {}
    metrics = {}
    band_summaries = {}
    freqs = None

    # Compute per-day
    for d in days:
        lfp_tc = load_lfp(day2lfp[d])
        x = channel_reduce(lfp_tc, method=chan_reduce)          # (T,)
        f, t, power_spect_db = compute_power_spect_db(x, fs_lfp, nperseg, noverlap, fmax=fmax)
        if freqs is None: freqs = f
        Sdb_z = normalize_per_freq(power_spect_db)                         # (F,T) z per freq
        spec_norm[d] = (t, Sdb_z)
        band_summaries[d] = summarize_bands(power_spect_db, f)             # time series (dB)
        metrics[d] = overall_metrics(power_spect_db, f)                    # scalars

    # ---- FIGURE 1: per-day spectrograms (z-scored per freq) ----
    n = len(days)
    ncols = min(3, n)
    nrows = int(np.ceil(n/ncols))
    fig1, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.2*nrows), squeeze=False)
    #plt.subplots_adjust(hspace=0.8)
    fig1.tight_layout(h_pad=3, w_pad=1)
    vlim = 3.0
    for i, d in enumerate(days):
        ax = axes[i//ncols, i % ncols]
        t, Sdb_z = spec_norm[d]
        im = ax.imshow(Sdb_z, origin='lower', aspect='auto',
                       extent=[t[0], t[-1], freqs[0], freqs[-1]],
                       vmin=-vlim, vmax=vlim, cmap='viridis')
        ax.set_title(f"{d} (z per freq)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
        #ax.set_y_li 
    cbar = fig1.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Power (z)")

    # ---- FIGURE 2: overall metrics across days ----
    # Using robust comparisons: broadband dB and gamma–delta diff (proxy for activation)
    bb = [metrics[d]['broadband_db'] for d in days]
    gdr = [metrics[d]['gamma_delta_diff_db'] for d in days]


    fig2, ax2 = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax2[0].plot(days, bb, marker='o'); ax2[0].set_ylabel("Broadband power (dB)")
    ax2[0].set_title("Overall activity across days")
    ax2[1].plot(days, gdr, marker='o'); ax2[1].set_ylabel("Gamma − Delta (dB)")
    ax2[1].set_xlabel("Day")
    fig2.tight_layout()

    # ---- FIGURE 3: band power time courses (per day) ----
    fig3, ax3 = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    # gamma and delta examples
    for d in days:
        t, _ = spec_norm[d]
        ax3[0].plot(t, band_summaries[d]['gamma'], alpha=0.9, label=d)
        ax3[1].plot(t, band_summaries[d]['delta'], alpha=0.9, label=d)
    ax3[0].set_ylabel("Gamma power (dB)"); ax3[0].legend(ncol=min(3, len(days)))
    ax3[1].set_ylabel("Delta power (dB)"); ax3[1].set_xlabel("Time (s)")
    fig3.tight_layout()
    if show: plt.show()
    return dict(freqs=freqs, spec_norm=spec_norm, band_summaries=band_summaries, metrics=metrics)

def plv(phi):
    return np.abs(np.mean(np.exp(1j*phi)))

def rayleigh_p(phi):
    n = len(phi)
    R = n * plv(phi)
    # Small-sample corrected approximation
    z = (R**2)/n
    p = np.exp(-z) * (1 + (2*z - z**2)/(4*n) - (24*z - 132*z**2 + 76*z**3 - 9*z**4)/(288*n**2))
    return max(min(p,1.0), 0.0)

# def compute_spectrogram_db(
#     signal, 
#     fs_hz, 
#     nperseg=2048, 
#     noverlap=None, 
#     window='hann',
#     scaling='spectrum',
#     max_freq_hz=None
# ):
#     """
#     Compute a (power) spectrogram in dB.

#     Parameters
#     ----------
#     signal : 1D array
#         Time-domain signal (e.g., LFP trace).
#     fs_hz : float
#         Sampling rate in Hz.
#     nperseg : int, optional
#         FFT window length (samples). Controls time/frequency resolution.
#     noverlap : int or None, optional
#         Overlap between consecutive windows (samples).
#         If None, defaults to 75% overlap (int(0.75 * nperseg)).
#     window : str, optional
#         Spectrogram window type (scipy.signal-compatible).
#     scaling : {'density','spectrum'}, optional
#         'spectrum' gives power; 'density' gives power spectral density.
#     max_freq_hz : float or None, optional
#         If set, returns/plots only frequencies ≤ this value.

#     Returns
#     -------
#     freqs_hz : 1D array
#         Frequency axis (Hz), possibly truncated at max_freq_hz.
#     times_s : 1D array
#         Time axis (s).
#     power_db : 2D array [freq x time]
#         Power in dB (10*log10 of spectrogram power).
#     """
#     if noverlap is None:
#         noverlap = int(0.75 * nperseg)

#     freqs_hz, times_s, Sxx = spectrogram(
#         signal,
#         fs=fs_hz,
#         nperseg=nperseg,
#         noverlap=noverlap,
#         window=window,
#         detrend=False,
#         scaling=scaling
#     )

#     # Convert to dB safely
#     power_db = 10.0 * np.log10(np.maximum(Sxx, 1e-20))

#     # Optionally limit the frequency axis (e.g., to 0–200 Hz)
#     if max_freq_hz is not None:
#         keep = freqs_hz <= max_freq_hz
#         freqs_hz = freqs_hz[keep]
#         power_db = power_db[keep, :]

#     return freqs_hz, times_s, power_db

