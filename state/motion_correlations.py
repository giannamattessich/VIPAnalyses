import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, zscore

def corr_motion_dff(dff, motion_at_frames):
    n_cells = dff.shape[0]
    corrs = np.full(n_cells, np.nan)
    valid = ~np.isnan(motion_at_frames)

    m = zscore(motion_at_frames[valid])

    for i in range(n_cells):
        x = zscore(dff[i, valid])
        r, _ = pearsonr(x, m)
        corrs[i] = r

    return corrs

def find_motion_onsets(motion_at_frames,
                       frame_times,
                       high_percentile=80,
                       min_gap_s=2.0):
    """
    Detect motion ONSETS from a frame-aligned motion trace.
    Returns indices (into frame axis).
    """
    motion = np.asarray(motion_at_frames)
    t = np.asarray(frame_times)

    # threshold for "moving"
    thr = np.nanpercentile(motion, high_percentile)
    moving = motion > thr

    # raw onsets: 0 -> 1 transitions
    onset_idx = np.where(np.diff(moving.astype(int)) == 1)[0] + 1

    if onset_idx.size == 0:
        return np.array([], dtype=int)

    # enforce minimum gap between onsets (in seconds)
    dt = np.median(np.diff(t))
    min_gap_frames = int(np.round(min_gap_s / dt))

    kept = [onset_idx[0]]
    for idx in onset_idx[1:]:
        if idx - kept[-1] >= min_gap_frames:
            kept.append(idx)

    return np.array(kept, dtype=int)


def compute_mta(dff, frame_times,
                motion_at_frames,
                pre_s=3.0, post_s=3.0,
                high_percentile=80, min_gap_s=2.0,
                zscore_cells=False):
    """
    Compute motion-triggered averages.

    Returns:
        t_win        : (n_window,) time relative to motion onset
        mta_pop      : (n_window,) population-average MTA
        mta_cells    : (n_cells, n_window) per-cell MTA
        onset_idx_kept: indices of onsets used
    """
    dff = np.asarray(dff)
    t = np.asarray(frame_times)
    motion = np.asarray(motion_at_frames)

    n_cells, n_frames = dff.shape
    dt = np.median(np.diff(t))

    pre_frames = int(np.round(pre_s / dt))
    post_frames = int(np.round(post_s / dt))
    win = pre_frames + post_frames + 1

    # optional per-cell z-scoring before averaging
    if zscore_cells:
        mu = np.nanmean(dff, axis=1, keepdims=True)
        sd = np.nanstd(dff, axis=1, keepdims=True) + 1e-8
        dff_z = (dff - mu) / sd
    else:
        dff_z = dff

    onset_idx = find_motion_onsets(
        motion, t,
        high_percentile=high_percentile,
        min_gap_s=min_gap_s
    )

    trials = []
    kept_onsets = []
    for idx in onset_idx:
        start = idx - pre_frames
        end = idx + post_frames + 1  # slice is [start, end)

        if start < 0 or end > n_frames:
            continue  # skip onsets too close to edges

        trials.append(dff_z[:, start:end])  # (cells, win)
        kept_onsets.append(idx)

    if len(trials) == 0:
        # no valid onsets
        t_win = np.arange(-pre_frames, post_frames+1) * dt
        return t_win, np.full(win, np.nan), np.full((n_cells, win), np.nan), np.array([])

    trials = np.stack(trials, axis=0)          # (trials, cells, win)
    mta_cells = np.nanmean(trials, axis=0)     # (cells, win)
    mta_pop = np.nanmean(mta_cells, axis=0)    # (win,)

    t_win = np.arange(-pre_frames, post_frames+1) * dt
    return t_win, mta_pop, mta_cells, np.array(kept_onsets, dtype=int)

def compute_auroc_motion_per_cell(dff, motion_at_frames, high_percentile=80, zscore_cells=True):
    """
    dff: (n_cells, n_frames)
    motion_at_frames: (n_frames,) continuous motion, aligned to frames
    high_percentile: percentile above which frames are 'motion' (1), others 'still' (0)

    Returns
    -------
    auroc_vals : (n_cells,) AUROC for each cell (NaN if cannot compute)
    labels     : (n_frames,) binary motion labels (0/1)
    """
    dff = np.asarray(dff)
    motion = np.asarray(motion_at_frames)
    n_cells, n_frames = dff.shape

    # mask NaNs
    valid = ~np.isnan(motion)
    if valid.sum() < 10:
        return np.full(n_cells, np.nan), np.full(n_frames, np.nan)

    motion_valid = motion[valid]

    # binary labels: high motion vs rest
    thr = np.percentile(motion_valid, high_percentile)
    labels_valid = (motion_valid > thr).astype(int)

    # need both classes or AUROC is undefined
    if labels_valid.min() == labels_valid.max():
        return np.full(n_cells, np.nan), np.full(n_frames, np.nan)

    # option: z-score each cell
    dff_valid = dff[:, valid]
    if zscore_cells:
        mu = np.nanmean(dff_valid, axis=1, keepdims=True)
        sd = np.nanstd(dff_valid, axis=1, keepdims=True) + 1e-8
        dff_valid = (dff_valid - mu) / sd

    auroc_vals = np.full(n_cells, np.nan)
    for ci in range(n_cells):
        x = dff_valid[ci]
        try:
            auroc_vals[ci] = roc_auc_score(labels_valid, x)
        except Exception:
            auroc_vals[ci] = np.nan

    # put labels back in full-length vector (NaN where invalid)
    labels_full = np.full(n_frames, np.nan)
    labels_full[valid] = labels_valid

    return auroc_vals, labels_full

def spike_triggered_average_motion(
    spikes, frame_times, motion, window_s=(-6.0, 6.0),
    spike_thresh=None, n_shuffles=500, rng=None):
    """
    STA of facial motion around spikes, with null (chance) distribution via circular shuffles.
    Returns:
      tau: time lags (s)
      sta_mean: mean STA across spikes
      sta_sem:  SEM across spikes
      z_sta: z-score of STA vs shuffle null (per lag)
    """
    rng = np.random.default_rng(rng)
    ft = np.asarray(frame_times, float)
    m  = np.asarray(motion, float)
    sp = np.asarray(spikes, float)

    # bin width (assume nearly constant)
    dt = np.median(np.diff(ft))
    pre, post = window_s
    k_pre  = int(np.floor(abs(pre) / dt))
    k_post = int(np.floor(abs(post) / dt))
    tau = np.arange(-k_pre, k_post + 1) * dt

    # pool spikes across cells or pick a subset
    if spike_thresh is None:
        spike_thresh = np.nanpercentile(sp[sp>0], 50) if np.any(sp>0) else 0.0
    spike_mask = sp > spike_thresh
    spike_idx = np.where(spike_mask.sum(axis=0) > 0)[0]  # any cell spiked in that frame
    # (If you want per-cell STAs, loop cells instead of pooling.)

    # collect windows
    segs = []
    for t0 in spike_idx:
        a = t0 - k_pre
        b = t0 + k_post + 1
        if a < 0 or b > m.size:
            continue
        segs.append(m[a:b])
    if len(segs) == 0:
        raise ValueError("No spike-centered windows within bounds.")
    segs = np.vstack(segs)  # (n_spikes, win_len)

    sta_mean = np.nanmean(segs, axis=0)
    sta_sem  = np.nanstd(segs, axis=0, ddof=1) / np.sqrt(segs.shape[0])

    # --- build shuffle null via circularly shifting motion ---
    win_len = segs.shape[1]
    null_means = np.empty((n_shuffles, win_len), float)
    for s in range(n_shuffles):
        shift = rng.integers(0, m.size)
        m_shuf = np.roll(m, shift)
        segs_shuf = []
        for t0 in spike_idx:
            a = t0 - k_pre
            b = t0 + k_post + 1
            if a < 0 or b > m.size:
                continue
            segs_shuf.append(m_shuf[a:b])
        segs_shuf = np.vstack(segs_shuf)
        null_means[s] = np.nanmean(segs_shuf, axis=0)

    null_mu = null_means.mean(axis=0)
    null_sd = null_means.std(axis=0, ddof=1) + 1e-12
    z_sta = (sta_mean - null_mu) / null_sd

    return tau, sta_mean, sta_sem, z_sta

def plot_sta(tau, sta_mean, sta_sem, z_sta=None, title=None):
    # Left panel look: mean ± SEM and dashed line at 0 s
    plt.figure(figsize=(4,3))
    plt.fill_between(tau, sta_mean - sta_sem, sta_mean + sta_sem, alpha=0.3)
    plt.plot(tau, sta_mean)
    plt.axvline(0, linestyle='--')
    plt.xlabel("Delay (s)")
    plt.ylabel("STA (motion)")
    plt.tight_layout()
    plt.show()
    plt.title('Population: spike triggered average by facial motion')

    if z_sta is not None:
        plt.figure(figsize=(4,3))
        plt.plot(tau, z_sta)
        plt.axvline(0, linestyle='--')
        plt.xlabel("Delay (s)")
        plt.ylabel("STA (z-score vs chance)")
        plt.tight_layout()
        plt.show()
    if title is not None:
        plt.title(title)
