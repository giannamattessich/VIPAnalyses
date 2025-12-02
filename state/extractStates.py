import traceback, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from utils.stats import zscore_robust

'''GET STATE FROM MOTION AND LOCOMOTION BOOLEAN VALUES'''
def annotate_state(state_df_row):
    if state_df_row['motion_bool']:
        state_df_row['state'] = 'aroused'  
    else:
        state_df_row['state'] = 'unaroused' 
    return state_df_row

def proportion_motion_correlated_cells(spks, motion, method='spearman', alpha=0.05):
    """
    Compute the proportion of cells significantly correlated with motion.

    Parameters
    ----------
    spks : array (n_cells x n_frames)
        Deconvolved or inferred spike array.
    motion : array (n_frames,)
        Facial motion signal, must match spks.shape[1].
    method : str
        'spearman' (rank) or 'pearson' (linear) correlation.
    alpha : float
        FDR-corrected significance threshold.

    Returns
    -------
    result_df : DataFrame
        Columns: ['cell', 'r', 'p', 'q', 'significant']
    proportion : float
        Fraction of significant cells.
    """

    n_cells, n_frames = spks.shape
    if len(motion) != n_frames:
        raise ValueError("motion must have same length as number of frames in spks")

    # Normalize motion
    motion = np.asarray(motion, float)
    motion -= np.nanmean(motion)
    motion /= (np.nanstd(motion) + 1e-12)

    corrs, pvals = [], []
    for i in range(n_cells):
        s = spks[i, :]
        if np.all(np.isnan(s)):  # skip empty cells
            corrs.append(np.nan)
            pvals.append(np.nan)
            continue
        s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-12)
        if method == 'pearson':
            r, p = pearsonr(s, motion)
        else:
            r, p = spearmanr(s, motion)
        corrs.append(r)
        pvals.append(p)

    # FDR correction
    valid = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    if valid.sum() > 0:
        _, q, _, _ = multipletests(np.array(pvals)[valid], alpha=alpha, method='fdr_bh')
        qvals[valid] = q

    df = pd.DataFrame({
        "cell": np.arange(n_cells),
        "r": corrs,
        "p": pvals,
        "q": qvals,
    })
    df["significant"] = df["q"] < alpha

    proportion = df["significant"].mean(skipna=True)
    return df, proportion

### USE FOR PLOTTING COLORED STATE TIMELINE
### USE AS PROVIDED ARGUMENT FOR X VALS -> 
# in the format [(x_start_1, segment_len_1), (x_start_2, segment_len_2)...]
def state_timeline_ranges(state_indices, restrict_range =False,
                           start_s=None, end_s=None, fps=30):
    idx_ranges = []
    last_start_idx = state_indices[0]
    last_idx_seen = state_indices[0]
    curr_segment_len = 0

    for state_idx in state_indices[1:]:
        curr_segment_len += 1
        if state_idx > last_idx_seen + 2:
            idx_ranges.append((last_start_idx, curr_segment_len))
            curr_segment_len = 0
            last_start_idx = state_idx
        last_idx_seen = state_idx
    idx_ranges = np.array(idx_ranges, dtype=tuple)
    try:
        if restrict_range and start_s is not None and end_s is not None:
            # get samples in range
            idx_ranges = np.array(idx_ranges, dtype=tuple)
            indices_in_range = np.where((idx_ranges[:,0] < end_s*fps) & (idx_ranges[:,0] > start_s*fps))[0]
            idx_ranges = idx_ranges[indices_in_range]
    except:
        traceback.print_exc()
    return [tuple(range) for range in idx_ranges / 30]

def classify_correlation(df_corr, alpha=0.05):
    df = df_corr.copy()

    # Classify
    def classify(row):
        if pd.isna(row.q) or pd.isna(row.r):
            return "no data"
        if row.q < alpha and row.r > 0:
            return "correlated"
        elif row.q < alpha and row.r < 0:
            return "anticorrelated"
        else:
            return "not correlated"

    df["corr_class"] = df.apply(classify, axis=1)

    # Compute percentages
    counts = df["corr_class"].value_counts(normalize=True).mul(100)
    counts = counts.reindex(["correlated", "anticorrelated", "not correlated"], fill_value=0)
    return df, counts    


def plot_correlation(df_corr, plot_title='', bar_colors = ["#6200FF", "#4BEE00", "#939099"]):
    df, counts = classify_correlation(df_corr)
    # Bar plot
    plt.figure(figsize=(5,4))
    bars = plt.bar(counts.index, counts.values, color=bar_colors)
    plt.ylabel("Percentage of cells (%)")
    plt.title(f"{plot_title}: Facial motion correlation across cells")
    # Display percentage labels
    for i, v in enumerate(counts.values):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

    plt.ylim(0, max(counts.values)*1.2)
    plt.tight_layout()
    plt.show()

    return df, counts

def cell_motion_correlations(spikes, motion, zscore=True):
    """
    spikes: (n_cells, T)
    motion: (T,)
    returns: array of correlation coefficients (n_cells,)
    """
    spikes = np.asarray(spikes, float)
    motion = np.asarray(motion, float)
    if zscore:
        spikes = zscore_robust(spikes)
        motion = zscore_robust(motion)
    r_vals = np.array([spearmanr(spikes[i], motion)[0] for i in range(spikes.shape[0])])
    return r_vals

def state_samples_map(state_df):
    aroused_samples = state_df[state_df['state'] == 'aroused']
    unaroused_samples = state_df[state_df['state'] == 'unaroused']
    return {'aroused': aroused_samples, 'unaroused': unaroused_samples}

def plot_state_timeline(state_df, dff_trace, start=None, end=None):
    state_map = state_samples_map(state_df)
    if start is None or end is None:
        start, end = 0, len(state_df)
    aroused_timeranges = state_timeline_ranges(state_map['aroused'].index,
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1]) 
    unaroused_timeranges = state_timeline_ranges(state_map['unaroused'].index,
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1])
    quiet_awake_timeranges = state_timeline_ranges(state_map['quiet awake'].index, 
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1])
    fig, axs = plt.subplots(5, 1, figsize=(50, 30))
    start, end = 0, len(state_df)
    axs[0].plot(np.arange(start, end), state_df['motion'][start:end], color='black')
    axs[1].plot(np.arange(start, end), state_df['treadmill'][start:end], color='black')
    axs[2].plot(np.arange(start, end), state_df['pupil_area'][start:end], color='black')
    #axs[0].axhline(y=18379.056640625, color='r', linestyle='--', label='Reference Line')
    #axs[2].axhline(y=1.2562316279032602, color='r', linestyle='--', label='Reference Line')
    axs[3].broken_barh(aroused_timeranges, (0.2,0.2), facecolors=("black"), label='aroused')
    axs[3].broken_barh(quiet_awake_timeranges, (0.2, 0.2), facecolors=("#4436426E"), label='quiet awake')
    axs[3].broken_barh(unaroused_timeranges, (0.2,  0.2), facecolors=("#E9E9E96D"), label='unaroused')
    axs[3].legend(loc='upper right')
    axs[0].set_title('Facial motion')
    axs[1].set_title('Locomotion')
    axs[2].set_title('Pupil area')
    axs[3].set_title('State timeline')
    axs[4].plot(np.arange(len(dff_trace)), dff_trace)


def plot_state_timeline_vip(state_df, motion_trace, dff_trace, start=None, end=None, title=None):
    state_map = state_samples_map(state_df)
    if start is None or end is None:
        start, end = 0, len(state_df)
    aroused_timeranges = state_timeline_ranges(state_map['aroused'].index,
                                                            restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1]) 
    unaroused_timeranges = state_timeline_ranges(state_map['unaroused'].index,
                                                        restrict_range=True, start_s=state_df['time'].iloc[0], end_s=state_df['time'].iloc[-1])
    fig, axs = plt.subplots(3, 1, figsize=(50, 30))
    start, end = 0, len(motion_trace)
    axs[0].plot(np.arange(start, end), motion_trace[start:end], color='black')
    #axs[0].axhline(y=18379.056640625, color='r', linestyle='--', label='Reference Line')
    #axs[2].axhline(y=1.2562316279032602, color='r', linestyle='--', label='Reference Line')
    axs[2].broken_barh(aroused_timeranges, (0.2,0.2), facecolors=("black"), label='aroused')
    axs[2].broken_barh(unaroused_timeranges, (0.2,  0.2), facecolors=("#5050506C"), label='unaroused')
    axs[2].legend(loc='upper right')
    axs[0].set_title('Facial motion')
    axs[1].set_title('Spikes')
    axs[2].set_title('State timeline')
    axs[1].plot(np.arange(len(dff_trace)), dff_trace, color='black')
    if title is not None:
        plt.suptitle(title)
        
def spike_triggered_average_motion(
    spikes, frame_times, motion, window_s=(-6.0, 6.0),
    spike_thresh=None, n_shuffles=500, rng=None
):
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


# def plot_correlation_hist_population(twop_class, spikes,
#                               plot_title='',
#                                 bar_colors = ["#6200FF", "#4BEE00", "#939099"]):
#     #dataclasses, state_dfs, s2p_outs, recordings = load_data_for_day(day)
#     corr_df_all = pd.DataFrame()
#     print(f'Finding correlations for day {day}')
#     for rec_idx in range(len(s2p_outs)):
#         spike = spikes[rec_idx][:,:-1]
#         aligned_time_df = dataclasses[rec_idx].make_aligned_frame_df(state_dfs[rec_idx])
#         if len(aligned_time_df) < spike.shape[1]:
#             spike = spike[:,:len(aligned_time_df)]
#         motion_z = zscore_robust(aligned_time_df['motion'])
#         spks_z   = zscore_robust(spike)
#         print(spks_z.shape, len(motion_z))
#         # 3) Feed into your correlation/proportion function
#         df_corr, prop = proportion_motion_correlated_cells(spks_z, motion_z, method='spearman', alpha=0.05)
#         corr_df_all = pd.concat([corr_df_all, df_corr], ignore_index=True)
#         print(f"Motion-correlated fraction for {recordings[rec_idx]}: {prop:.1%}")
#     corr_df, counts = classify_correlation(corr_df_all, plot_title=day)
#     plot_correlation(corr_df, plot_title, bar_colors)