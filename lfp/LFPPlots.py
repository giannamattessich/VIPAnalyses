import numpy as np
from lfp.readLFP import *
from lfp.lfpFunctions import *
from scipy.ndimage import gaussian_filter

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

def plot_buzcode_style_spectrogram(
    to, fo, spec,
    time_sigma=2,
    freq_sigma=1,
    vmin=-2, vmax=2,
    axs=None,
    t_start=None,
    t_end=None,
    cmap = 'jet_r',
    colorbar=True
):
    """
    Plot a Buzcode-style smoothed spectrogram.

    Parameters
    ----------
    to : 1D array
        Time vector (seconds), length T.
    fo : 1D array
        Frequency vector (Hz), length F.
    spec : 2D array
        Spectrogram. Shape may be (T, F) or (F, T).
    t_start, t_end : float or None
        Optional time limits for plotting (seconds).
    """

    to = np.asarray(to)
    fo = np.asarray(fo)
    spec = np.asarray(spec)

    # ----- Optional time cropping -----
    if t_start is not None or t_end is not None:
        if t_start is None:
            t_start = to[0]
        if t_end is None:
            t_end = to[-1]

        mask = (to >= t_start) & (to <= t_end)

        # keep original time length to match spectrogram axis
        to_full = to
        to = to_full[mask]

        # Detect which axis of spec is time and crop that axis
        if spec.shape[0] == to_full.size:
            # time along axis 0: (T, F)
            spec = spec[mask, :]
        elif spec.shape[1] == to_full.size:
            # time along axis 1: (F, T)
            spec = spec[:, mask]
        else:
            raise ValueError(
                f"Time vector length {to_full.size} does not match "
                f"any axis of spec {spec.shape}"
            )

    # ----- Log power -----
    spec_log = np.log(spec + 1e-12)

    # ----- Smoothing: (time, freq) axes -----
    spec_smooth = gaussian_filter(spec_log, sigma=(time_sigma, freq_sigma))

    # For imshow we want array as (freq, time) → transpose if needed
    # After smoothing, spec_smooth has same orientation as spec.
    # We’ll always do .T for plotting and use extent with to, fo.
    img_data = spec_smooth.T

    # ----- Plotting -----
    if axs is not None:
        im = axs.imshow(
            img_data,
            aspect="auto",
            origin="lower",
            extent=[to[0], to[-1], fo[0], fo[-1]],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        axs.set_ylabel("Freq (Hz)")
        axs.set_xlabel("Time (s)")
        axs.set_ylim(0, 80)
        axs.set_yticks([0, 5, 10, 15, 20])
        if colorbar:
            plt.colorbar(im, ax=axs, label="Power (log)")
        return axs, im

    # No axis provided → make our own
    plt.figure(figsize=(10, 4))
    im = plt.imshow(
        img_data,
        aspect="auto",
        origin="lower",
        extent=[to[0], to[-1], fo[0], fo[-1]],
        vmin=vmin,
        vmax=vmax,
        cmap="jet_r",
    )
    plt.ylabel("Freq (Hz)")
    plt.xlabel("Time (s)")
    plt.ylim(0, 80)
    plt.yticks([0, 5, 10, 15, 20])
    if colorbar:
        plt.colorbar(im, label="Power (log)")
    plt.tight_layout()
    plt.show()
    return axs, im

def plot_delta_lfp(
    lfp,
    fs,
    lowcut=0.5,
    highcut=4.0,
    t_start=None,
    t_end=None,
    downsample=1,
    ax=None,
    color=None,
    title="Delta-band LFP (0.5–4 Hz)"
):

    # --- Ensure 1D ---
    lfp = np.asarray(lfp).reshape(-1).astype(float)

    # --- Clean NaNs and infs ---
    mask = np.isnan(lfp) | np.isinf(lfp)
    if mask.any():
        print(f"Interpolating {mask.sum()} bad samples...")
        t_idx = np.arange(len(lfp))
        lfp[mask] = np.interp(t_idx[mask], t_idx[~mask], lfp[~mask])

    # Build time vector
    n_samples = lfp.size
    t = np.arange(n_samples) / fs

    # Optional cropping
    if t_start is not None or t_end is not None:
        if t_start is None: t_start = t[0]
        if t_end is None:   t_end = t[-1]
        mask = (t >= t_start) & (t <= t_end)
        lfp = lfp[mask]
        t = t[mask]

    # Filter to delta band
    lfp_delta = bandpass_filter_sos(lfp, fs, lowcut=lowcut, highcut=highcut)

    # Downsample for plotting clarity
    t_plot = t[::downsample]
    lfp_plot = lfp_delta[::downsample]

    # Create axis if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    if color is None:
        color = 'black'
    ax.plot(t_plot, lfp_plot, linewidth=0.8,color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)

    return ax, t_plot, lfp_plot


def plot_gamma_lfp(
    lfp,
    fs,
    lowcut=30.0,
    highcut=80.0,
    t_start=None,
    t_end=None,
    downsample=1,
    ax=None,
    color='black',
    title="Gamma-band LFP (30–80 Hz)"
):
    """
    Plot gamma-band filtered LFP using a stable SOS bandpass filter.
    """

    # --- Ensure 1D float array ---
    lfp = np.asarray(lfp).reshape(-1).astype(float)

    # Clean NaNs / infs
    mask = np.isnan(lfp) | np.isinf(lfp)
    if mask.any():
        print(f"Interpolating {mask.sum()} NaN/Inf samples...")
        t_idx = np.arange(len(lfp))
        lfp[mask] = np.interp(t_idx[mask], t_idx[~mask], lfp[~mask])

    # Time vector
    n_samples = lfp.size
    t = np.arange(n_samples) / fs

    # Optional time cropping
    if t_start is not None or t_end is not None:
        if t_start is None:
            t_start = t[0]
        if t_end is None:
            t_end = t[-1]

        mask = (t >= t_start) & (t <= t_end)
        lfp = lfp[mask]
        t = t[mask]

    # Gamma-band filter (stable)
    lfp_gamma = bandpass_filter_sos(lfp, fs, lowcut, highcut)

    # Downsample for plotting clarity
    if downsample > 1:
        t_plot = t[::downsample]
        lfp_plot = lfp_gamma[::downsample]
    else:
        t_plot = t
        lfp_plot = lfp_gamma

    # Create axis if none passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    ax.plot(t_plot, lfp_plot, linewidth=0.8, color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(title)

    return ax, t_plot, lfp_plot