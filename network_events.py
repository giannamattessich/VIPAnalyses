from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

# =========================
# Helpers
# =========================

def mad(x: np.ndarray, axis=None, keepdims=False, eps: float = 1e-12) -> np.ndarray:
    """Median absolute deviation (scaled like std if you want: *1.4826)."""
    x = np.asarray(x)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    m = np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    if not keepdims and axis is not None:
        med = np.squeeze(med, axis=axis)
        m = np.squeeze(m, axis=axis)
    return np.maximum(m, eps)


def gaussian_smooth_1d(x: np.ndarray, sigma: float, truncate: float = 3.0) -> np.ndarray:
    """
    Simple Gaussian smoothing (pure numpy). sigma in samples.
    If sigma<=0, returns x.
    """
    x = np.asarray(x, dtype=float)
    if sigma is None or sigma <= 0:
        return x

    radius = int(np.ceil(truncate * sigma))
    k = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= np.sum(w)

    # pad reflect to reduce edge artifacts
    xp = np.pad(x, (radius, radius), mode="reflect")
    y = np.convolve(xp, w, mode="valid")
    return y


def binary_dilate_1d(x: np.ndarray, half_width: int) -> np.ndarray:
    """
    Dilate a 1D boolean array by half_width samples on each side.
    Pure numpy via convolution.
    """
    x = np.asarray(x, dtype=bool)
    if half_width <= 0:
        return x
    k = np.ones(2 * half_width + 1, dtype=int)
    y = np.convolve(x.astype(int), k, mode="same") > 0
    return y


def find_contiguous_true_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return [(start, end_exclusive), ...] for contiguous True runs in a boolean mask.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    # transitions
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    return list(zip(starts.tolist(), ends.tolist()))


# =========================
# 1) Cell-event raster
# =========================

def events_from_spikes(spks: np.ndarray,
                       threshold: float = 0.0,
                       min_gap_frames: int = 0) -> np.ndarray:
    """
    Convert deconvolved spikes to boolean event raster.
    spks: (n_cells, n_time)
    threshold: consider spks > threshold as events
    min_gap_frames: optional "refractory" merging; if >0, events separated by <= gap are merged.
    """
    spks = np.asarray(spks)
    E = spks > threshold

    if min_gap_frames > 0:
        # merge close events per cell by dilating then re-thresholding
        out = np.zeros_like(E, dtype=bool)
        for i in range(E.shape[0]):
            out[i] = binary_dilate_1d(E[i], half_width=min_gap_frames)
        E = out

    return E


def events_from_dff(dff: np.ndarray,
                    z_thresh: float = 3.5,
                    min_event_frames: int = 1,
                    smooth_sigma_frames: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Threshold ΔF/F into events using per-cell MAD z-score.
    dff: (n_cells, n_time)
    Returns:
        E: bool (n_cells, n_time)
        z: float (n_cells, n_time) MAD z-score
    """
    dff = np.asarray(dff, dtype=float)
    if smooth_sigma_frames and smooth_sigma_frames > 0:
        dff_s = np.zeros_like(dff)
        for i in range(dff.shape[0]):
            dff_s[i] = gaussian_smooth_1d(dff[i], sigma=smooth_sigma_frames)
    else:
        dff_s = dff

    med = np.nanmedian(dff_s, axis=1, keepdims=True)
    m = mad(dff_s, axis=1, keepdims=True)
    z = (dff_s - med) / m

    E = z > z_thresh

    # enforce minimum event duration per cell
    if min_event_frames > 1:
        out = np.zeros_like(E, dtype=bool)
        for i in range(E.shape[0]):
            runs = find_contiguous_true_runs(E[i])
            for s, e in runs:
                if (e - s) >= min_event_frames:
                    out[i, s:e] = True
        E = out

    return E, z


# =========================
# 2) Network event detection
# =========================

@dataclass
class NetworkEvents:
    pop_activity: np.ndarray              # (T,)
    pop_activity_smooth: np.ndarray       # (T,)
    event_mask: np.ndarray                # (T,) bool: inside an event
    windows: List[Tuple[int, int]]        # list of (start, end_exclusive) in frames


def detect_network_events(E: np.ndarray,
                          *,
                          smooth_sigma_frames: float = 1.0,
                          thr_percentile: float = 95.0,
                          min_event_frames: int = 2,
                          merge_gap_frames: int = 1) -> NetworkEvents:
    """
    Detect network events from cell-event raster E by thresholding smoothed population activity.

    E: (n_cells, T) bool
    smooth_sigma_frames: gaussian smoothing on pop activity
    thr_percentile: threshold at this percentile of smoothed pop activity
    min_event_frames: discard events shorter than this
    merge_gap_frames: merge events separated by <= gap frames
    """
    E = np.asarray(E, dtype=bool)
    n_cells, T = E.shape

    pop = E.mean(axis=0)  # fraction active per frame
    pop_s = gaussian_smooth_1d(pop, sigma=smooth_sigma_frames) if smooth_sigma_frames > 0 else pop

    thr = np.percentile(pop_s, thr_percentile)
    mask = pop_s > thr

    # merge gaps
    if merge_gap_frames > 0 and mask.any():
        # If two True runs are separated by <= gap, fill the gap:
        runs = find_contiguous_true_runs(mask)
        merged = []
        cur_s, cur_e = runs[0]
        for s, e in runs[1:]:
            if s - cur_e <= merge_gap_frames:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
        # rebuild mask
        mask2 = np.zeros(T, dtype=bool)
        for s, e in merged:
            mask2[s:e] = True
        mask = mask2

    # enforce minimum duration
    windows = []
    for s, e in find_contiguous_true_runs(mask):
        if (e - s) >= min_event_frames:
            windows.append((s, e))

    # rebuild final mask from filtered windows
    mask_final = np.zeros(T, dtype=bool)
    for s, e in windows:
        mask_final[s:e] = True

    return NetworkEvents(
        pop_activity=pop,
        pop_activity_smooth=pop_s,
        event_mask=mask_final,
        windows=windows )

# =========================
# 3) Per-event metrics
# =========================

def per_event_metrics(E: np.ndarray,
                      net: NetworkEvents,
                      *,
                      dff: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Compute per-network-event metrics from E and detected windows.

    Metrics returned as arrays of length n_events:
      - start, end (frames)
      - duration_frames
      - participation_peak: max fraction active within window
      - participation_mean: mean fraction active within window
      - participation_union: fraction of cells active at least once in window
      - pop_auc: sum(pop_activity) within window (fraction-active * frames)
      - optional dff_mean_auc: mean across cells of sum(dff) in window (if dff provided)
    """
    E = np.asarray(E, dtype=bool)
    n_cells, T = E.shape

    if dff is not None:
        dff = np.asarray(dff, dtype=float)
        if dff.shape != (n_cells, T):
            raise ValueError(f"dff must have shape {(n_cells, T)}, got {dff.shape}")

    starts = []
    ends = []
    dur = []
    p_peak = []
    p_mean = []
    p_union = []
    pop_auc = []
    dff_mean_auc = []

    pop = net.pop_activity  # fraction active per frame

    for (s, e) in net.windows:
        starts.append(s)
        ends.append(e)
        dur.append(e - s)

        pop_w = pop[s:e]
        p_peak.append(np.max(pop_w) if pop_w.size else np.nan)
        p_mean.append(np.mean(pop_w) if pop_w.size else np.nan)
        pop_auc.append(np.sum(pop_w) if pop_w.size else np.nan)

        Ew = E[:, s:e]
        # union participation: a cell counts if it fired at least once in window
        active_any = Ew.any(axis=1)
        p_union.append(active_any.mean())

        if dff is not None:
            # session-agnostic: average (across cells) of per-cell AUC within window
            dff_w = dff[:, s:e]
            dff_mean_auc.append(np.mean(np.sum(dff_w, axis=1)))

    out = {
        "start_frame": np.asarray(starts, dtype=int),
        "end_frame": np.asarray(ends, dtype=int),
        "duration_frames": np.asarray(dur, dtype=int),
        "participation_peak": np.asarray(p_peak, dtype=float),
        "participation_mean": np.asarray(p_mean, dtype=float),
        "participation_union": np.asarray(p_union, dtype=float),
        "pop_auc": np.asarray(pop_auc, dtype=float),
    }
    if dff is not None:
        out["dff_mean_auc"] = np.asarray(dff_mean_auc, dtype=float)

    return out


def session_summary_from_events(event_metrics: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Summarize per-event metrics into a session-level dict.
    """
    n = len(event_metrics["duration_frames"])
    if n == 0:
        return {
            "n_events": 0,
            "mean_participation_union": np.nan,
            "median_participation_union": np.nan,
            "mean_participation_peak": np.nan,
            "mean_duration_frames": np.nan,
        }

    pu = event_metrics["participation_union"]
    pp = event_metrics["participation_peak"]
    dur = event_metrics["duration_frames"]

    return {
        "n_events": int(n),
        "mean_participation_union": float(np.nanmean(pu)),
        "median_participation_union": float(np.nanmedian(pu)),
        "mean_participation_peak": float(np.nanmean(pp)),
        "mean_duration_frames": float(np.nanmean(dur)),
    }


def event_rate_and_iei(net: NetworkEvents, fs_hz: float) -> Dict[str, np.ndarray | float]:
    """
    Compute event rate (events/min) and IEI (seconds) from detected windows.
    fs_hz: sampling rate of the time axis for E (e.g., imaging frame rate).
    """
    windows = net.windows
    n_events = len(windows)
    T = net.event_mask.size

    total_sec = T / fs_hz
    rate_per_min = (n_events / total_sec) * 60.0 if total_sec > 0 else np.nan

    if n_events >= 2:
        # use event onsets
        onsets = np.array([s for s, _ in windows], dtype=float) / fs_hz
        iei = np.diff(onsets)
    else:
        iei = np.array([], dtype=float)

    return {"event_rate_per_min": float(rate_per_min), "iei_s": iei}


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # fake data: 200 cells, 3000 frames
    rng = np.random.default_rng(0)
    n_cells, T = 200, 3000
    dff = rng.normal(0, 1, size=(n_cells, T))
    # inject a few population events
    for t0 in [400, 1200, 2000, 2600]:
        cells = rng.choice(n_cells, size=80, replace=False)
        dff[cells, t0:t0+10] += 5.0

    E, z = events_from_dff(dff, z_thresh=3.5, min_event_frames=1, smooth_sigma_frames=0.5)
    net = detect_network_events(E, smooth_sigma_frames=1.0, thr_percentile=95.0,
                                min_event_frames=3, merge_gap_frames=2)

    em = per_event_metrics(E, net, dff=dff)
    summ = session_summary_from_events(em)
    er = event_rate_and_iei(net, fs_hz=1.366)  # example imaging fps

    print("Session summary:", summ)
    print("Event rate/min:", er["event_rate_per_min"], "IEI count:", len(er["iei_s"]))
