import numpy as np
import matplotlib.pyplot as plt

def extract_windows(traces, event_idx, pre, post):
    """
    traces: (n_cells, n_t)
    event_idx: (n_events,) integer indices
    pre/post: window sizes in samples
    returns: (n_events, n_cells, pre+post)
    """
    traces = np.asarray(traces)
    n_cells, n_t = traces.shape
    w = pre + post

    valid = (event_idx - pre >= 0) & (event_idx + post <= n_t)
    event_idx = np.asarray(event_idx)[valid]

    out = np.empty((len(event_idx), n_cells, w), dtype=traces.dtype)
    for i, ix in enumerate(event_idx):
        out[i] = traces[:, ix - pre : ix + post]
    return out

def plot_trial_aligned_population(traces, event_idx, fs, pre_s=1.0, post_s=2.0):
    pre = int(round(pre_s * fs))
    post = int(round(post_s * fs))
    win = extract_windows(traces, event_idx, pre, post)  # (n_events, n_cells, w)

    # collapse cells -> population trace per trial
    pop_per_trial = win.mean(axis=1)  # (n_events, w)
    mean = pop_per_trial.mean(axis=0)
    sem = pop_per_trial.std(axis=0, ddof=1) / np.sqrt(pop_per_trial.shape[0])

    t = (np.arange(pre + post) - pre) / fs

    plt.figure()
    plt.plot(t, mean, label="Mean across trials (pop mean)")
    plt.fill_between(t, mean - sem, mean + sem, alpha=0.2)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Time from event (s)")
    plt.ylabel("Activity")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fraction_active(traces, t=None, threshold=2.0):
    """
    traces: (n_cells, n_timepoints)
    threshold: activity threshold (e.g., z-score or deconv spikes > 0)
    """
    traces = np.asarray(traces)
    if t is None:
        t = np.arange(traces.shape[1])

    frac = (traces > threshold).mean(axis=0)  # fraction of cells active each timepoint

    plt.figure()
    plt.plot(t, frac)
    plt.xlabel("Time")
    plt.ylabel("Fraction active")
    plt.tight_layout()
    plt.show()
