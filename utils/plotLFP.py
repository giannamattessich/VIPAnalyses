from utils.lfpFunctions import *

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