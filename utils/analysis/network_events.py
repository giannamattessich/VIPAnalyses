from utils.stats import zscore_robust
import numpy as np, pandas as pd, matplotlib.pyplot as plt

def compute_fraction_active(traces, thr_active=2.0):
    """
    traces: (n_cells, n_time)
    thr_active: threshold for 'active' cell

    Returns:
        frac_active: (n_time,)
    """
    traces = np.asarray(traces)
    frac_active = (traces > thr_active).mean(axis=0)
    return frac_active

def detect_event_peaks(frac_active, fs, thr_event=0.2, min_gap_s=0.5):
    """
    Detect population events from fraction-active trace.

    Returns:
        peaks_idx (samples)
        peaks_time (seconds)
    """
    frac_active = np.asarray(frac_active)
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

    peaks = np.array(peaks, dtype=int)
    return peaks, peaks / fs

def compute_event_rate_and_iei(peaks_idx, fs, n_time):
    """
    peaks_idx: event peak indices
    fs: sampling rate (Hz)
    n_time: total recording length in samples

    Returns:
        event_rate_per_min
        iei_s (array)
    """
    peaks_idx = np.asarray(peaks_idx)

    if peaks_idx.size == 0:
        return np.nan, np.array([])

    duration_min = (n_time / fs) / 60.0
    rate = peaks_idx.size / duration_min

    peaks_t = peaks_idx / fs
    iei = np.diff(peaks_t)

    return rate, iei

def median_pairwise_correlation(traces, max_cells=2000):
    """
    Returns median correlation across cell pairs
    """
    traces = np.asarray(traces)
    n_cells = traces.shape[0]

    if n_cells < 2:
        return np.nan

    if n_cells > max_cells:
        idx = np.random.choice(n_cells, max_cells, replace=False)
        traces = traces[idx]

    C = np.corrcoef(traces)
    iu = np.triu_indices(C.shape[0], k=1)

    return np.median(C[iu])

def compute_event_participation(
    traces,
    peaks_idx,
    fs,
    thr_active=2.0,
    window_s=0.5):
    """
    Returns:
        participation_per_event (array of fractions)
    """
    traces = np.asarray(traces)
    peaks_idx = np.asarray(peaks_idx)

    half_window = int(round(window_s * fs))
    n_cells, n_time = traces.shape
    participation = []

    for p in peaks_idx:
        a = max(0, p - half_window)
        b = min(n_time, p + half_window)

        active_cells = (traces[:, a:b] > thr_active).any(axis=1)
        participation.append(active_cells.mean())
    return np.array(participation)

def analyze_network_synchrony(
    traces,
    fs,
    *,
    zscore=True,
    thr_active=2.0,
    thr_event=None,
    min_gap_s=0.25,
    participation_window_s=0.5
):
    """
    Returns dictionary of all key metrics
    """
    X = zscore_robust(traces) if zscore else traces
    frac = compute_fraction_active(X, thr_active)
    if thr_event is None:
        thr_event = np.quantile(frac, 0.95)
        
    peaks_idx, peaks_time = detect_event_peaks(
        frac, fs,
        thr_event=thr_event,
        min_gap_s=min_gap_s
    )

    rate, iei = compute_event_rate_and_iei(
        peaks_idx, fs, X.shape[1]
    )

    corr = median_pairwise_correlation(X)

    participation = compute_event_participation(
        X,
        peaks_idx,
        fs,
        thr_active=thr_active,
        window_s=participation_window_s
    )

    return {
        "fraction_active_trace": frac,
        "event_indices": peaks_idx,
        "event_times_s": peaks_time,
        "event_rate_per_min": rate,
        "iei_s": iei,
        "median_pairwise_corr": corr,
        "participation_per_event": participation,
        "mean_participation": np.mean(participation) if participation.size else np.nan,
    }

def summarize_network_metrics(results, *, rec_id=None):
    iei = np.asarray(results["iei_s"], dtype=float)
    participation = np.asarray(results["participation_per_event"], dtype=float)

    n_events = int(np.asarray(results["event_indices"]).size)
    rate = float(results["event_rate_per_min"]) if np.isfinite(results["event_rate_per_min"]) else np.nan
    med_corr = float(results["median_pairwise_corr"]) if np.isfinite(results["median_pairwise_corr"]) else np.nan

    row = {
        "rec_id": rec_id,
        "n_events": n_events,
        "event_rate_per_min": rate,
        "iei_mean_s": float(np.mean(iei)) if iei.size else np.nan,
        "iei_median_s": float(np.median(iei)) if iei.size else np.nan,
        "iei_p10_s": float(np.percentile(iei, 10)) if iei.size else np.nan,
        "iei_p90_s": float(np.percentile(iei, 90)) if iei.size else np.nan,
        "median_pairwise_corr": med_corr,
        "mean_participation": float(np.mean(participation)) if participation.size else np.nan,
        "median_participation": float(np.median(participation)) if participation.size else np.nan,
    }

    df = pd.DataFrame([row])

    # Print nicely
    to_print = {k: row[k] for k in row if k != "rec_id"}
    print(f"Recording: {rec_id}" if rec_id is not None else "Recording metrics:")
    for k, v in to_print.items():
        if isinstance(v, float) and np.isfinite(v):
            print(f"  {k:>22s}: {v:.4g}")
        else:
            print(f"  {k:>22s}: {v}")

    return df

def plot_fraction_active_trace(results, fs, title=None, axs=None, color='black', vline_color="#FF00DD"):
    frac = np.asarray(results["fraction_active_trace"], dtype=float)
    event_t = np.asarray(results["event_times_s"], dtype=float)
    t = np.arange(frac.size) / fs
    if axs is None:
        # --- Plot 1: fraction active with detected events ---
        plt.figure()
        plt.plot(t, frac, label="Fraction active", color=color)
        if event_t.size:
            # vertical lines at event times
            ymin, ymax = np.min(frac), np.max(frac)
            plt.vlines(event_t, ymin=ymin, ymax=ymax, alpha=0.25, label="Event peaks", color=vline_color)
        plt.xlabel("Time (s)")
        plt.ylabel("Fraction active")
        plt.title(title or "Network events (fraction active + detected peaks)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        axs.plot(t, frac, label="Fraction active", color=color)
        if event_t.size:
            # vertical lines at event times
            ymin, ymax = np.min(frac), np.max(frac)
            axs.vlines(event_t, ymin=ymin, ymax=ymax, alpha=0.25, label="Event peaks", color=vline_color)
        axs.set_xlabel("Time (s)")
        axs.set_ylabel("Fraction active")
        axs.set_title(title or "Network events (fraction active + detected peaks)")
        return axs     

def plot_iei_histogram(results, max_iei_s=None, title=None, axs=None, color="#3A0A94"):
    iei = np.asarray(results["iei_s"], dtype=float)
    vals = None
    if iei.size:
        vals = iei
        if max_iei_s is not None:
            vals = vals[vals <= max_iei_s]
        if axs is None:
            plt.figure()
            plt.hist(vals, bins=40, color=color)
            plt.xlabel("Inter-event interval (s)")
            plt.ylabel("Count")
            if title is not None:
                plt.title(title)
            else:
                plt.title("IEI distribution")
            plt.tight_layout()
            plt.text(0.5, 0.5, "Not enough events to compute IEIs", ha="center", va="center")
            plt.axis("off")
            plt.show()
        else:
            axs.hist(vals, bins=40, color=color)
            axs.set_xlabel("Inter-event interval (s)")
            axs.set_ylabel("Count")
            if title is not None:
                axs.set_title(title)
            else:
                axs.set_title("IEI distribution")
            return axs
    else:
        if axs is not None: 
            axs.text(0.5, 0.5, "Not enough events to compute IEIs", ha="center", va="center")
            axs.axis("off")
            return axs
        else:
            plt.text(0.5, 0.5, "Not enough events to compute IEIs", ha="center", va="center")
            plt.axis("off")
            plt.show()

def plot_participation_histogram(results, title=None, axs=None, color='black'):
    participation = np.asarray(results["participation_per_event"], dtype=float)
    if participation.size:
        if axs is None:
            plt.hist(participation, bins=30, color=color)
            plt.xlabel("Per-event participation (fraction of cells)")
            plt.ylabel("Count")
            if title is not None:
                plt.title(title)
            else:
                plt.title("Participation distribution")
            plt.tight_layout()
            plt.show()
        else:
            axs.hist(participation, bins=30, color=color)
            axs.set_xlabel("Per-event participation (fraction of cells)")
            axs.set_ylabel("Count")
            if title is not None:
                axs.set_title(title)
            else:
                axs.set_title("Participation distribution")    
            return axs       
    else:
        if axs is None:
            plt.text(0.5, 0.5, "No events -> no participation values", ha="center", va="center")
            plt.axis("off")
            if title is not None:
                plt.title(title)
            plt.show()
        else:
            axs.text(0.5, 0.5, "No events -> no participation values", ha="center", va="center")
            axs.axis("off")
            if title is not None:
                plt.title(title)
            return axs


def plot_network_metrics(results, fs, *, title=None, max_iei_s=None):
    pass
    # frac = np.asarray(results["fraction_active_trace"], dtype=float)
    # event_idx = np.asarray(results["event_indices"], dtype=int)
    # event_t = np.asarray(results["event_times_s"], dtype=float)
    # iei = np.asarray(results["iei_s"], dtype=float)
    # participation = np.asarray(results["participation_per_event"], dtype=float)

    # t = np.arange(frac.size) / fs

    # # --- Plot 1: fraction active with detected events ---
    # plt.figure()
    # plt.plot(t, frac, label="Fraction active")
    # if event_t.size:
    #     # vertical lines at event times
    #     ymin, ymax = np.min(frac), np.max(frac)
    #     plt.vlines(event_t, ymin=ymin, ymax=ymax, alpha=0.25, label="Event peaks")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Fraction active")
    # plt.title(title or "Network events (fraction active + detected peaks)")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # --- Plot 2: IEI histogram ---
    # plt.figure()
    # if iei.size:
    #     vals = iei
    #     if max_iei_s is not None:
    #         vals = vals[vals <= max_iei_s]
    #     plt.hist(vals, bins=40)
    #     plt.xlabel("Inter-event interval (s)")
    #     plt.ylabel("Count")
    #     plt.title("IEI distribution")
    #     plt.tight_layout()
    # else:
    #     plt.text(0.5, 0.5, "Not enough events to compute IEIs", ha="center", va="center")
    #     plt.axis("off")
    # plt.show()

    # # --- Plot 3: Participation histogram ---
    # plt.figure()
    # if participation.size:
    #     plt.hist(participation, bins=30)
    #     plt.xlabel("Per-event participation (fraction of cells)")
    #     plt.ylabel("Count")
    #     plt.title("Participation distribution")
    #     plt.tight_layout()
    # else:
    #     plt.text(0.5, 0.5, "No events -> no participation values", ha="center", va="center")
    #     plt.axis("off")
    # plt.show()


# from __future__ import annotations
# from dataclasses import dataclass
# from typing import Dict, Optional, Tuple, List
# import numpy as np

# # =========================
# # Helpers
# # =========================

# def mad(x: np.ndarray, axis=None, keepdims=False, eps: float = 1e-12) -> np.ndarray:
#     """Median absolute deviation (scaled like std if you want: *1.4826)."""
#     x = np.asarray(x)
#     med = np.nanmedian(x, axis=axis, keepdims=True)
#     m = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
#     if not keepdims and axis is not None:
#         med = np.squeeze(med, axis=axis)
#         m = np.squeeze(m, axis=axis)
#     return np.maximum(m, eps)


# def gaussian_smooth_1d(x: np.ndarray, sigma: float, truncate: float = 3.0) -> np.ndarray:
#     """
#     Simple Gaussian smoothing (pure numpy). sigma in samples.
#     If sigma<=0, returns x.
#     """
#     x = np.asarray(x, dtype=float)
#     if sigma is None or sigma <= 0:
#         return x

#     radius = int(np.ceil(truncate * sigma))
#     k = np.arange(-radius, radius + 1)
#     w = np.exp(-0.5 * (k / sigma) ** 2)
#     w /= np.sum(w)

#     # pad reflect to reduce edge artifacts
#     xp = np.pad(x, (radius, radius), mode="reflect")
#     y = np.convolve(xp, w, mode="valid")
#     return y


# def binary_dilate_1d(x: np.ndarray, half_width: int) -> np.ndarray:
#     """
#     Dilate a 1D boolean array by half_width samples on each side.
#     Pure numpy via convolution.
#     """
#     x = np.asarray(x, dtype=bool)
#     if half_width <= 0:
#         return x
#     k = np.ones(2 * half_width + 1, dtype=int)
#     y = np.convolve(x.astype(int), k, mode="same") > 0
#     return y


# def find_contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
#     """
#     Return [(start, end_exclusive), ...] for contiguous True runs in a boolean mask.
#     """
#     mask = np.asarray(mask, dtype=bool)
#     if mask.size == 0:
#         return []

#     # transitions
#     d = np.diff(mask.astype(int))
#     starts = np.where(d == 1)[0] + 1
#     ends = np.where(d == -1)[0] + 1

#     if mask[0]:
#         starts = np.r_[0, starts]
#     if mask[-1]:
#         ends = np.r_[ends, mask.size]

#     return list(zip(starts.tolist(), ends.tolist()))


# # =========================
# # 1) Cell-event raster
# # =========================

# def events_from_spikes(spks: np.ndarray,
#                        threshold: float = 0.0,
#                        min_gap_frames: int = 0) -> np.ndarray:
#     """
#     Convert deconvolved spikes to boolean event raster.
#     spks: (n_cells, n_time)
#     threshold: consider spks > threshold as events
#     min_gap_frames: optional "refractory" merging; if >0, events separated by <= gap are merged.
#     """
#     spks = np.asarray(spks)
#     E = spks > threshold

#     if min_gap_frames > 0:
#         # merge close events per cell by dilating then re-thresholding
#         out = np.zeros_like(E, dtype=bool)
#         for i in range(E.shape[0]):
#             out[i] = binary_dilate_1d(E[i], half_width=min_gap_frames)
#         E = out

#     return E


# def events_from_dff(dff: np.ndarray,
#                     z_thresh: float = 3.5,
#                     min_event_frames: int = 1,
#                     smooth_sigma_frames: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Threshold ΔF/F into events using per-cell MAD z-score.
#     dff: (n_cells, n_time)
#     Returns:
#         E: bool (n_cells, n_time)
#         z: float (n_cells, n_time) MAD z-score
#     """
#     dff = np.asarray(dff, dtype=float)
#     if smooth_sigma_frames and smooth_sigma_frames > 0:
#         dff_s = np.zeros_like(dff)
#         for i in range(dff.shape[0]):
#             dff_s[i] = gaussian_smooth_1d(dff[i], sigma=smooth_sigma_frames)
#     else:
#         dff_s = dff

#     med = np.nanmedian(dff_s, axis=1, keepdims=True)
#     m = mad(dff_s, axis=1, keepdims=True)
#     z = (dff_s - med) / m

#     E = z > z_thresh

#     # enforce minimum event duration per cell
#     if min_event_frames > 1:
#         out = np.zeros_like(E, dtype=bool)
#         for i in range(E.shape[0]):
#             runs = find_contiguous_true_runs(E[i])
#             for s, e in runs:
#                 if (e - s) >= min_event_frames:
#                     out[i, s:e] = True
#         E = out

#     return E, z


# # =========================
# # 2) Network event detection
# # =========================

# @dataclass
# class NetworkEvents:
#     pop_activity: np.ndarray              # (T,)
#     pop_activity_smooth: np.ndarray       # (T,)
#     event_mask: np.ndarray                # (T,) bool: inside an event
#     windows: List[Tuple[int, int]]        # list of (start, end_exclusive) in frames


# def detect_network_events(E: np.ndarray,
#                           *,
#                           smooth_sigma_frames: float = 1.0,
#                           thr_percentile: float = 95.0,
#                           min_event_frames: int = 2,
#                           merge_gap_frames: int = 1) -> NetworkEvents:
#     """
#     Detect network events from cell-event raster E by thresholding smoothed population activity.

#     E: (n_cells, T) bool
#     smooth_sigma_frames: gaussian smoothing on pop activity
#     thr_percentile: threshold at this percentile of smoothed pop activity
#     min_event_frames: discard events shorter than this
#     merge_gap_frames: merge events separated by <= gap frames
#     """
#     E = np.asarray(E, dtype=bool)
#     n_cells, T = E.shape

#     pop = E.mean(axis=0)  # fraction active per frame
#     pop_s = gaussian_smooth_1d(pop, sigma=smooth_sigma_frames) if smooth_sigma_frames > 0 else pop

#     thr = np.percentile(pop_s, thr_percentile)
#     mask = pop_s > thr

#     # merge gaps
#     if merge_gap_frames > 0 and mask.any():
#         # If two True runs are separated by <= gap, fill the gap:
#         runs = find_contiguous_true_runs(mask)
#         merged = []
#         cur_s, cur_e = runs[0]
#         for s, e in runs[1:]:
#             if s - cur_e <= merge_gap_frames:
#                 cur_e = e
#             else:
#                 merged.append((cur_s, cur_e))
#                 cur_s, cur_e = s, e
#         merged.append((cur_s, cur_e))
#         # rebuild mask
#         mask2 = np.zeros(T, dtype=bool)
#         for s, e in merged:
#             mask2[s:e] = True
#         mask = mask2

#     # enforce minimum duration
#     windows = []
#     for s, e in find_contiguous_true_runs(mask):
#         if (e - s) >= min_event_frames:
#             windows.append((s, e))

#     # rebuild final mask from filtered windows
#     mask_final = np.zeros(T, dtype=bool)
#     for s, e in windows:
#         mask_final[s:e] = True

#     return NetworkEvents(
#         pop_activity=pop,
#         pop_activity_smooth=pop_s,
#         event_mask=mask_final,
#         windows=windows )

# # =========================
# # 3) Per-event metrics
# # =========================

# def per_event_metrics(E: np.ndarray,
#                       net: NetworkEvents,
#                       *,
#                       dff: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
#     """
#     Compute per-network-event metrics from E and detected windows.

#     Metrics returned as arrays of length n_events:
#       - start, end (frames)
#       - duration_frames
#       - participation_peak: max fraction active within window
#       - participation_mean: mean fraction active within window
#       - participation_union: fraction of cells active at least once in window
#       - pop_auc: sum(pop_activity) within window (fraction-active * frames)
#       - optional dff_mean_auc: mean across cells of sum(dff) in window (if dff provided)
#     """
#     E = np.asarray(E, dtype=bool)
#     n_cells, T = E.shape

#     if dff is not None:
#         dff = np.asarray(dff, dtype=float)
#         if dff.shape != (n_cells, T):
#             raise ValueError(f"dff must have shape {(n_cells, T)}, got {dff.shape}")

#     starts = []
#     ends = []
#     dur = []
#     p_peak = []
#     p_mean = []
#     p_union = []
#     pop_auc = []
#     dff_mean_auc = []

#     pop = net.pop_activity  # fraction active per frame

#     for (s, e) in net.windows:
#         starts.append(s)
#         ends.append(e)
#         dur.append(e - s)

#         pop_w = pop[s:e]
#         p_peak.append(np.max(pop_w) if pop_w.size else np.nan)
#         p_mean.append(np.mean(pop_w) if pop_w.size else np.nan)
#         pop_auc.append(np.sum(pop_w) if pop_w.size else np.nan)

#         Ew = E[:, s:e]
#         # union participation: a cell counts if it fired at least once in window
#         active_any = Ew.any(axis=1)
#         p_union.append(active_any.mean())

#         if dff is not None:
#             # session-agnostic: average (across cells) of per-cell AUC within window
#             dff_w = dff[:, s:e]
#             dff_mean_auc.append(np.mean(np.sum(dff_w, axis=1)))

#     out = {
#         "start_frame": np.asarray(starts, dtype=int),
#         "end_frame": np.asarray(ends, dtype=int),
#         "duration_frames": np.asarray(dur, dtype=int),
#         "participation_peak": np.asarray(p_peak, dtype=float),
#         "participation_mean": np.asarray(p_mean, dtype=float),
#         "participation_union": np.asarray(p_union, dtype=float),
#         "pop_auc": np.asarray(pop_auc, dtype=float),
#     }
#     if dff is not None:
#         out["dff_mean_auc"] = np.asarray(dff_mean_auc, dtype=float)

#     return out


# def session_summary_from_events(event_metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
#     """
#     Summarize per-event metrics into a session-level dict.
#     """
#     n = len(event_metrics["duration_frames"])
#     if n == 0:
#         return {
#             "n_events": 0,
#             "mean_participation_union": np.nan,
#             "median_participation_union": np.nan,
#             "mean_participation_peak": np.nan,
#             "mean_duration_frames": np.nan,
#         }

#     pu = event_metrics["participation_union"]
#     pp = event_metrics["participation_peak"]
#     dur = event_metrics["duration_frames"]

#     return {
#         "n_events": int(n),
#         "mean_participation_union": float(np.nanmean(pu)),
#         "median_participation_union": float(np.nanmedian(pu)),
#         "mean_participation_peak": float(np.nanmean(pp)),
#         "mean_duration_frames": float(np.nanmean(dur)),
#     }


# def event_rate_and_iei(net: NetworkEvents, fs_hz: float) -> Dict[str, np.ndarray | float]:
#     """
#     Compute event rate (events/min) and IEI (seconds) from detected windows.
#     fs_hz: sampling rate of the time axis for E (e.g., imaging frame rate).
#     """
#     windows = net.windows
#     n_events = len(windows)
#     T = net.event_mask.size

#     total_sec = T / fs_hz
#     rate_per_min = (n_events / total_sec) * 60.0 if total_sec > 0 else np.nan

#     if n_events >= 2:
#         # use event onsets
#         onsets = np.array([s for s, _ in windows], dtype=float) / fs_hz
#         iei = np.diff(onsets)
#     else:
#         iei = np.array([], dtype=float)

#     return {"event_rate_per_min": float(rate_per_min), "iei_s": iei}


# # =========================
# # Example usage
# # =========================
# if __name__ == "__main__":
#     # fake data: 200 cells, 3000 frames
#     rng = np.random.default_rng(0)
#     n_cells, T = 200, 3000
#     dff = rng.normal(0, 1, size=(n_cells, T))
#     # inject a few population events
#     for t0 in [400, 1200, 2000, 2600]:
#         cells = rng.choice(n_cells, size=80, replace=False)
#         dff[cells, t0:t0+10] += 5.0

#     E, z = events_from_dff(dff, z_thresh=3.5, min_event_frames=1, smooth_sigma_frames=0.5)
#     net = detect_network_events(E, smooth_sigma_frames=1.0, thr_percentile=95.0,
#                                 min_event_frames=3, merge_gap_frames=2)

#     em = per_event_metrics(E, net, dff=dff)
#     summ = session_summary_from_events(em)
#     er = event_rate_and_iei(net, fs_hz=1.366)  # example imaging fps

#     print("Session summary:", summ)
#     print("Event rate/min:", er["event_rate_per_min"], "IEI count:", len(er["iei_s"]))
