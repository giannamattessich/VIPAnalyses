import numpy as np
import matplotlib.pyplot as plt
from utils.stats import *

def plot_population_mean(traces, t=None, *, label="Population mean", show_sem=True,
                          axs=None, title=None, color='black'):
    """
    traces: (n_cells, n_timepoints)
    t: (n_timepoints,) optional time vector
    """
    traces = np.asarray(traces)

    if t is None:
        t = np.arange(traces.shape[1])

    mean = traces.mean(axis=0)
    if axs is None:
        plt.figure()
        plt.plot(t, mean, label=label, color=color)
        if show_sem:
            sem = traces.std(axis=0, ddof=1) / np.sqrt(traces.shape[0])
            plt.fill_between(t, mean - sem, mean + sem, alpha=0.2)
        plt.legend()
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        axs.plot(t, mean, label=label, color=color)
        if show_sem:
            sem = traces.std(axis=0, ddof=1) / np.sqrt(traces.shape[0])
            axs.fill_between(t, mean - sem, mean + sem, alpha=0.2)
        if title is not None:
            axs.set_title(title)
        return axs 

def plot_fraction_active(traces, fs, t=None, threshold=2.0, title=None, axs=None, color='black'):
    """
    traces: (n_cells, n_timepoints)
    threshold: activity threshold (e.g., z-score or deconv spikes > 0)
    """
    traces = np.asarray(traces)
    if t is None:
        t = np.arange(traces.shape[1])

    frac = (traces > threshold).mean(axis=0)  # fraction of cells active each timepoint
    if axs is None:
        plt.figure()
        plt.plot(t, frac, color=color)
        plt.xlabel("Time")
        plt.ylabel("Fraction active")
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        axs.plot(t, frac, color=color)
        if title is not None:
            axs.set_title(title)
        return axs

def plot_synchrony(traces, t=None, *, z=True, thr=2.0, smooth=None, title=None, axs=None):
    """
    traces: (n_cells, n_timepoints) e.g. dF/F or deconv spikes (continuous)
    z: robust z-score per cell before computing synchrony
    thr: threshold for "active" (in z units if z=True)
    smooth: int window (samples) for simple moving average (optional)
    """
    X = np.asarray(traces)
    if t is None:
        t = np.arange(X.shape[1])

    if z:
        X = zscore_robust(X, axis=1)

    pop_mean = X.mean(axis=0)
    frac_active = (X > thr).mean(axis=0)

    if smooth and smooth > 1:
        k = np.ones(smooth) / smooth
        pop_mean = np.convolve(pop_mean, k, mode="same")
        frac_active = np.convolve(frac_active, k, mode="same")
    if axs is None:
        plt.figure()
        plt.plot(t, pop_mean, label="Population mean")
        plt.plot(t, frac_active, label=f"Fraction active (>{thr})")
        plt.xlabel("Time")
        plt.ylabel("a.u.")
        plt.legend()
        plt.tight_layout()
        if title is not None:
            plt.title(title)
        plt.show()
        return
    else:
        axs.plot(t, pop_mean, label="Population mean")
        axs.plot(t, frac_active, label=f"Fraction active (>{thr})")
        if title is not None:
            axs.set_title(title)
        return axs    
    
def find_events_from_frac(frac, *, thr=0.2, min_gap=5):
    """
    frac: (n_timepoints,) fraction active
    thr: event threshold (0-1)
    min_gap: minimum samples between event peaks
    returns: peak indices
    """
    frac = np.asarray(frac)
    above = frac > thr
    peaks = []
    i = 1
    while i < len(frac) - 1:
        if above[i] and frac[i] >= frac[i-1] and frac[i] >= frac[i+1]:
            peaks.append(i)
            i += min_gap
        else:
            i += 1
    return np.array(peaks, dtype=int)

def plot_event_triggered_frac(traces, fs, *, z=True, thr_active=2.0, thr_event=0.2,
                              pre_s=2.0, post_s=4.0):
    X = np.asarray(traces)
    if z:
        X = zscore_robust(X, axis=1)

    frac = (X > thr_active).mean(axis=0)
    peaks = find_events_from_frac(frac, thr=thr_event, min_gap=int(0.5 * fs))

    pre = int(round(pre_s * fs))
    post = int(round(post_s * fs))
    valid = (peaks - pre >= 0) & (peaks + post < X.shape[1])
    peaks = peaks[valid]

    W = np.stack([frac[p-pre:p+post] for p in peaks], axis=0)  # (n_events, w)
    t = (np.arange(pre + post) - pre) / fs

    mean = W.mean(axis=0)
    sem = W.std(axis=0, ddof=1) / np.sqrt(W.shape[0])

    plt.figure()
    plt.plot(t, mean, label="Event-triggered fraction active")
    plt.fill_between(t, mean - sem, mean + sem, alpha=0.2)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Time from event peak (s)")
    plt.ylabel("Fraction active")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return peaks, frac

def plot_pairwise_corr_hist(traces, *, z=True, max_cells=2000, bins=50):
    X = np.asarray(traces)
    if z:
        X = zscore_robust(X, axis=1)

    # optional downsample for speed
    if X.shape[0] > max_cells:
        idx = np.random.choice(X.shape[0], size=max_cells, replace=False)
        X = X[idx]

    C = np.corrcoef(X)  # (n_cells, n_cells)
    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]

    plt.figure()
    plt.hist(vals, bins=bins)
    plt.xlabel("Pairwise correlation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    return vals

def compute_synchrony_metrics(
    traces,
    fs,
    *,
    z=True,
    thr_active=2.0,
    thr_event=0.2,
    min_gap_s=0.5,
    max_cells_corr=2000,
):
    """
    traces: (n_cells, n_timepoints)
    fs: sampling rate (Hz)
    returns: dict with synchrony metrics
    """
    X = np.asarray(traces)
    n_cells, n_t = X.shape
    duration_min = n_t / fs / 60.0

    if z:
        X = zscore_robust(X, axis=1)

    # --- Fraction active ---
    frac_active = (X > thr_active).mean(axis=0)
    mean_frac_active = frac_active.mean()

    # --- Event rate ---
    min_gap = int(round(min_gap_s * fs))
    peaks = []
    i = 1
    while i < len(frac_active) - 1:
        if (
            frac_active[i] > thr_event
            and frac_active[i] >= frac_active[i - 1]
            and frac_active[i] >= frac_active[i + 1]
        ):
            peaks.append(i)
            i += min_gap
        else:
            i += 1

    event_rate_per_min = len(peaks) / duration_min if duration_min > 0 else np.nan

    # --- Pairwise correlations ---
    Xc = X
    if n_cells > max_cells_corr:
        idx = np.random.choice(n_cells, size=max_cells_corr, replace=False)
        Xc = Xc[idx]

    C = np.corrcoef(Xc)
    iu = np.triu_indices(C.shape[0], k=1)
    median_pairwise_corr = np.median(C[iu])

    return {
        "mean_frac_active": mean_frac_active,
        "event_rate_per_min": event_rate_per_min,
        "median_pairwise_corr": median_pairwise_corr,
    }
