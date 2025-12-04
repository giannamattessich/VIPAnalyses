import numpy as np, os
from utils.filtering import *
import matplotlib.pyplot as plt

def load_buzcode_emg_fromlfp(mat_path):
    """
    Load Buzcode EMGFromLFP struct from EMGFROMLFP.LFP.mat.

    """
    import scipy
    #md = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    emg_from_lfp_file = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "EMGFromLFP" not in emg_from_lfp_file:
        raise KeyError(f"'EMGFromLFP' not found in {mat_path}. "
                        f"Top-level keys: {list(emg_from_lfp_file.keys())}")
    emg_from_lfp = emg_from_lfp_file["EMGFromLFP"]
    emg_signal = np.asarray(emg_from_lfp.data).squeeze()
    timestamps = np.asarray(emg_from_lfp.timestamps).squeeze()
    return emg_signal, timestamps


def compute_emg_from_lfp(
    lfp,
    fs,
    lowcut=100.0,
    highcut=250.0,
    order=4,
    smooth_win=0.05,   # seconds for smoothing window (e.g. 50 ms)
    agg="median",      # how to combine channels: "median" or "mean"
):
    """
    Estimate EMG envelope from LFP.

    Parameters
    ----------
    lfp : array
        Shape (n_samples,) or (n_channels, n_samples).
    fs : float
        Sampling rate (Hz).
    lowcut, highcut : float
        EMG-ish band limits (Hz), e.g. 100–250.
    order : int
        Butterworth bandpass order.
    smooth_win : float
        Smoothing window length in seconds.
    agg : {"median", "mean"}
        How to combine channels into a single EMG trace.

    Returns
    -------
    t : array, shape (n_samples,)
        Time vector (s).
    emg : array, shape (n_samples,)
        EMG envelope estimate.
    emg_per_chan : array, shape (n_channels, n_samples)
        Per-channel EMG envelopes (before aggregation).
    """
    x = np.asarray(lfp, dtype=float)

    # Ensure shape (n_channels, n_samples)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"LFP must be 1D or 2D, got shape {x.shape}")

    n_channels, n_samples = x.shape

    # Clean NaNs / infs
    bad = np.isnan(x) | np.isinf(x)
    if bad.any():
        print(f"Interpolating {bad.sum()} NaN/Inf samples across channels...")
        t_idx = np.arange(n_samples)
        for ch in range(n_channels):
            m = bad[ch]
            if m.any() and (~m).any():
                x[ch, m] = np.interp(t_idx[m], t_idx[~m], x[ch, ~m])
            elif m.all():
                x[ch] = 0.0

    # High-frequency bandpass (EMG-ish band)
    x_filt = bandpass_filter_sos(x, fs, lowcut, highcut, order=order)

    # Rectify (envelope proxy)
    x_rect = np.abs(x_filt)

    # Smooth with moving average
    win_samples = max(1, int(round(smooth_win * fs)))
    kernel = np.ones(win_samples) / win_samples
    emg_per_chan = np.zeros_like(x_rect)
    for ch in range(n_channels):
        emg_per_chan[ch] = np.convolve(x_rect[ch], kernel, mode="same")

    # Aggregate across channels
    if agg == "median":
        emg = np.median(emg_per_chan, axis=0)
    elif agg == "mean":
        emg = np.mean(emg_per_chan, axis=0)
    else:
        raise ValueError("agg must be 'median' or 'mean'")

    t = np.arange(n_samples) / fs
    return t, emg, emg_per_chan

def plot_emg_from_lfp(
    lfp,
    fs,
    lowcut=100.0,
    highcut=250.0,
    smooth_win=0.05,
    agg="median",
    t_start=None,
    t_end=None,
    downsample=1,
    ax=None,
    color=None,
    title="EMG from LFP"
):
    """
    Compute and plot EMG estimate from LFP on an axis.
    """
    # Compute full EMG
    t, emg, _ = compute_emg_from_lfp(
        lfp,
        fs,
        lowcut=lowcut,
        highcut=highcut,
        smooth_win=smooth_win,
        agg=agg,
    )

    # Optional crop
    if t_start is not None or t_end is not None:
        if t_start is None:
            t_start = t[0]
        if t_end is None:
            t_end = t[-1]
        mask = (t >= t_start) & (t <= t_end)
        t = t[mask]
        emg = emg[mask]

    # Downsample for plotting
    if downsample > 1:
        t_plot = t[::downsample]
        emg_plot = emg[::downsample]
    else:
        t_plot = t
        emg_plot = emg

    # Axis handling
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    if color is None:
        color='black'
    ax.plot(t_plot, emg_plot, linewidth=0.8, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EMG (a.u.)")
    ax.set_title(title)

    return ax, t_plot, emg_plot


def emg_envelope(emg, fs, lowcut=40, highcut=300, smooth_ms=100):
    """
    EMG preprocessing:
    1) bandpass filter
    2) rectify
    3) smooth with moving average
    """
    # 1) bandpass
    emg_bp = bandpass_filter(emg, fs, lowcut=lowcut, highcut=highcut)
    # 2) rectify
    emg_rect = np.abs(emg_bp)
    # 3) smooth (moving average)
    win_samples = max(1, int(smooth_ms * 1e-3 * fs))
    win = np.ones(win_samples) / win_samples
    emg_env = np.convolve(emg_rect, win, mode="same")
    return emg_bp, emg_rect, emg_env

#provide channel as idx in lfp file, NOT from intan
def plot_emg_trace_from_lfpchan(basepath, n_channels, ch_idx=0,
                             lfp_fs=1250.0, axs=None, title=None,
                             lowcut=40, highcut=300, smooth_ms=100):
    from lfp.readLFP import load_lfp_channel
    day = os.path.basename(basepath)
    mat_path = os.path.join(basepath, f"{day}.EMGFromLFP.LFP.mat")
    if not os.path.exists(mat_path):
        raise ValueError(f'No {day}.SleepStateEpisodes.states.mat found in provided basepath')
    emg, timestamps = load_buzcode_emg_fromlfp(mat_path)
    emg_bp, emg_rect, emg_env = emg_envelope(emg, lfp_fs, lowcut=lowcut, highcut=highcut, smooth_ms=smooth_ms)
    if axs is None:
        fig, axs= plt.subplots(1, 1) 
    t = np.arange(emg_env.size) / lfp_fs
    axs.plot(t, emg_env, color='black')
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("EMG envelope")
    axs.set_title(title)
    return axs

def plot_emg_envelope(emg, fs, t_start=0, t_end=10, title=None):
    """
    Plot raw EMG + envelope for a time window [t_start, t_end] in seconds.
    """
    n = len(emg)
    t = np.arange(n) / fs

    # select window
    mask = (t >= t_start) & (t <= t_end)
    if not np.any(mask):
        raise ValueError("No samples in selected window")

    emg_bp, emg_rect, emg_env = emg_envelope(emg, fs)
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 4), height_ratios=[2, 1])

    # raw bandpassed EMG
    ax[0].plot(t[mask], emg_bp[mask])
    ax[0].set_ylabel("EMG (µV)")
    ax[0].set_title(title or f"EMG (bandpass) {t_start:.1f}–{t_end:.1f} s")

    # envelope
    ax[1].plot(t[mask], emg_env[mask])
    ax[1].set_ylabel("Envelope")
    ax[1].set_xlabel("Time (s)")

    plt.tight_layout()
    return fig, ax

def plot_emg_spectrogram(emg, fs, t_start=0, t_end=60):
    n = len(emg)
    t = np.arange(n) / fs
    mask = (t >= t_start) & (t <= t_end)
    emg_seg = emg[mask]

    fig, ax = plt.subplots(figsize=(10, 3))
    Pxx, freqs, bins, im = ax.specgram(emg_seg, NFFT=512, Fs=fs, noverlap=256)
    ax.set_ylim(0, 100)  # focus on EMG band
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Freq (Hz)")
    ax.set_title("EMG spectrogram")
    plt.tight_layout()

# def plot_buzcode_style_spectrogram(
#     to, fo, spec,
#     time_sigma=2,
#     freq_sigma=1,
#     vmin=-2, vmax=2,
#     axs=None,
#     t_start=None,
#     t_end=None,
#     cmap='jet_r',
#     colorbar=True
# ):
#     """
#     Plot a Buzcode-style smoothed spectrogram.
    
#     Parameters
#     ----------
#     to : time vector
#     fo : frequency vector
#     spec : 2D array (freq x time)
#     t_start, t_end : optional time range to display
#     """
#     # --- Optional time cropping ---
#     if t_start is not None or t_end is not None:
#         if t_start is None:
#             t_start = to[0]
#         if t_end is None:
#             t_end = to[-1]

#         mask = (to >= t_start) & (to <= t_end)

#         # Apply mask
#         to = to[mask]
#         spec = spec[:, mask]   # freq x time

#     # --- Log power ---
#     spec_log = np.log(spec + 1e-12)

#     # --- Smoothing ---
#     from scipy.ndimage import gaussian_filter
#     spec_smooth = gaussian_filter(spec_log, sigma=(time_sigma, freq_sigma))

#     # --- Plotting ---
#     if axs is not None:
#         im = axs.imshow(
#             spec_smooth.T,
#             aspect="auto",
#             origin="lower",
#             extent=[to[0], to[-1], fo[0], fo[-1]],
#             vmin=vmin,
#             vmax=vmax,
#             cmap=cmap,
#         )
#         axs.set_ylabel("Freq (Hz)")
#         axs.set_xlabel("Time (s)")
#         axs.set_ylim(0, 80)
#         axs.set_yticks([0, 5, 10, 15, 20])
#         if colorbar:
#             plt.colorbar(im, ax=axs, label="Power (log)")
#         return axs

#     # No axis passed → create figure
#     plt.figure(figsize=(10, 4))
#     im = plt.imshow(
#         spec_smooth.T,
#         aspect="auto",
#         origin="lower",
#         extent=[to[0], to[-1], fo[0], fo[-1]],
#         vmin=vmin,
#         vmax=vmax,
#         cmap="jet_r",
#     )
#     plt.ylabel("Freq (Hz)")
#     plt.xlabel("Time (s)")
#     plt.ylim(0, 80)
#     plt.yticks([0, 5, 10, 15, 20])
#     if colorbar:
#         plt.colorbar(im, label="Power (log)")
#     plt.tight_layout()
#     plt.show()


# def plot_delta_lfp(
#     lfp,
#     fs,
#     lowcut=0.5,
#     highcut=4.0,
#     t_start=None,
#     t_end=None,
#     downsample=1,
#     ax=None,
#     color=None,
#     title="Delta-band LFP (0.5–4 Hz)"
# ):

#     # --- Ensure 1D ---
#     lfp = np.asarray(lfp).reshape(-1).astype(float)

#     # --- Clean NaNs and infs ---
#     mask = np.isnan(lfp) | np.isinf(lfp)
#     if mask.any():
#         print(f"Interpolating {mask.sum()} bad samples...")
#         t_idx = np.arange(len(lfp))
#         lfp[mask] = np.interp(t_idx[mask], t_idx[~mask], lfp[~mask])

#     # Build time vector
#     n_samples = lfp.size
#     t = np.arange(n_samples) / fs

#     # Optional cropping
#     if t_start is not None or t_end is not None:
#         if t_start is None: t_start = t[0]
#         if t_end is None:   t_end = t[-1]
#         mask = (t >= t_start) & (t <= t_end)
#         lfp = lfp[mask]
#         t = t[mask]

#     # Filter to delta band
#     lfp_delta = bandpass_filter_sos(lfp, fs, lowcut=lowcut, highcut=highcut)

#     # Downsample for plotting clarity
#     t_plot = t[::downsample]
#     lfp_plot = lfp_delta[::downsample]

#     # Create axis if none provided
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 3))
#     if color is None:
#         color = 'black'
#     ax.plot(t_plot, lfp_plot, linewidth=0.8,color=color)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude (µV)")
#     ax.set_title(title)

#     return ax, t_plot, lfp_plot


# def plot_gamma_lfp(
#     lfp,
#     fs,
#     lowcut=30.0,
#     highcut=80.0,
#     t_start=None,
#     t_end=None,
#     downsample=1,
#     ax=None,
#     color='black',
#     title="Gamma-band LFP (30–80 Hz)"
# ):
#     """
#     Plot gamma-band filtered LFP using a stable SOS bandpass filter.
#     """

#     # --- Ensure 1D float array ---
#     lfp = np.asarray(lfp).reshape(-1).astype(float)

#     # Clean NaNs / infs
#     mask = np.isnan(lfp) | np.isinf(lfp)
#     if mask.any():
#         print(f"Interpolating {mask.sum()} NaN/Inf samples...")
#         t_idx = np.arange(len(lfp))
#         lfp[mask] = np.interp(t_idx[mask], t_idx[~mask], lfp[~mask])

#     # Time vector
#     n_samples = lfp.size
#     t = np.arange(n_samples) / fs

#     # Optional time cropping
#     if t_start is not None or t_end is not None:
#         if t_start is None:
#             t_start = t[0]
#         if t_end is None:
#             t_end = t[-1]

#         mask = (t >= t_start) & (t <= t_end)
#         lfp = lfp[mask]
#         t = t[mask]

#     # Gamma-band filter (stable)
#     lfp_gamma = bandpass_filter_sos(lfp, fs, lowcut, highcut)

#     # Downsample for plotting clarity
#     if downsample > 1:
#         t_plot = t[::downsample]
#         lfp_plot = lfp_gamma[::downsample]
#     else:
#         t_plot = t
#         lfp_plot = lfp_gamma

#     # Create axis if none passed
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(10, 3))

#     ax.plot(t_plot, lfp_plot, linewidth=0.8, color=color)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude (µV)")
#     ax.set_title(title)

#     return ax, t_plot, lfp_plot