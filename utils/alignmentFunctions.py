import numpy as np, matplotlib.pyplot as plt, os, pandas as pd
from scipy.signal import butter, filtfilt
import intan.importrhdutilities as rhd_utils
from utils.filtering import *

### SOURCE: EHSAN + GABRIEL
def detectTransitions(analogSignal, earliestSample=0, histBinNumber=4, upTransition=False,\
                        latestSample=None, outputPlot=None, lowpassFilter=True,\
                             fs=20e3, lowPassfilterBand=500):
    """
    Input an analog signal and find the largest transitions 
    in the signal using histogram digitization

    Args:
        analogSignal (array-like): Signal to detect transitions

        earliestSample (int): First sample (in 20e3 hz rate) of signal to detect

        histBinNumber (int): How many histogram bins used to digitize signal

        upTransition (bool): whether signal transitions are up or down
        
        latestSample (int or None; default: None (entire signal duration)): last sample to detect transition (in 20e3 hz) 

        outputPlot (bool): whether to show binned signal histogram plot

        lowpassFilter (bool): whether to filter signal before transition detection

        latestSample (int or None; default: None (entire signal duration)): last sample to detect transition (in 20e3 hz) 

        fs (int; default is intan rate = 20000 hz): signal sampling rate 

        lowPassfilterBand (int): lowpass filter cutoff for signal when lowpassFilter == True

    Returns:
        triggerStart (np.ndarray)
            Indices of transition starts

        triggerStop (np.ndarray):
            Indices of end of transitions   

    """
    if not(latestSample):
        latestSample = len(analogSignal)

    if lowpassFilter:
        analogSignal = butter_lowpass_filter(analogSignal, lowPassfilterBand, fs, order=5)
    # plotting the histogram of values of analog signal
    histOutput = np.histogram(analogSignal,bins=histBinNumber)

    # identifing the two peaks on the distribution of analog values
    # we use these two values to set a decision boundry to detect the transition between two levels
    firstHistPeakEdge = np.argsort(histOutput[0])[-1]  # the position of the first peak on the histogram
    secondHistPeakEdge = np.argsort(histOutput[0])[-2] # the position of the second peak on the histogram

    # difining the cut level as the distance between the edges of the two peaks on the histogram 
    cutLevel = (histOutput[1][firstHistPeakEdge] + histOutput[1][firstHistPeakEdge + 1] \
                + histOutput[1][secondHistPeakEdge] + histOutput[1][secondHistPeakEdge + 1]) / 4 

    # defining a degitized version for the analog signal
    digitizedSignal = np.zeros(analogSignal.shape)
    # set the digitized strobe to 1 wherever the analog signal is more than the threshold value
    digitizedSignal[analogSignal>cutLevel] = 1

    # detecting the up and down transitions in the digitized signal
    upTransitionSignal = np.where(np.diff(digitizedSignal)==1)[0]
    downTransitionSignal = np.where((np.diff(digitizedSignal)==-1))[0]

    if upTransition:
        triggerStart = upTransitionSignal
        triggerStop = downTransitionSignal
    else:
        triggerStart = downTransitionSignal
        triggerStop = upTransitionSignal
        
    # just keeping those are that happen later than the earliest valid moment for the signal
    triggerStart = triggerStart[triggerStart>earliestSample]
    triggerStop = triggerStop[triggerStop>earliestSample]

    # and those that are happening before the latest desired time
    triggerStart = triggerStart[triggerStart<latestSample]
    triggerStop = triggerStop[triggerStop<latestSample]
    triggeredTransitionPlot = None

    if len(triggerStart) != len(triggerStop) or len(triggerStart) == 0:
        print(f"Warning: transition counts mismatched or empty. \
              Found {len(triggerStart)} starts but {len(triggerStop)} stops.")
        meanDur = np.nan
        triggeredTransitionPlot = 0
    else:
        meanDur = np.mean(triggerStop - triggerStart) / fs
        triggeredTransitionPlot = 1

    if outputPlot:
        plt.figure()
        plt.hist(analogSignal,bins=histBinNumber)
        plt.title('histogram of the analog values')
        # 
        if triggeredTransitionPlot:
            plt.figure()
            plt.title('all transitions triggered by detected transition time')
            transWindowToLook = int(1.25*meanDur*1e3)
            plt.xlabel('ms')
            for transitionTime in triggerStart[:]: 
                plt.plot(np.arange(-transWindowToLook,transWindowToLook,1e3/fs),\
                        analogSignal[int(transitionTime-transWindowToLook*fs/1e3):\
                                int(transitionTime+transWindowToLook*fs/1e3)],'gray')       
            plt.figure()
            plt.title('5 sample transitions zoomed-in')
            transWindowToLook = 25
            plt.xlabel('ms')
            for transitionTime in triggerStart[:5]:
                plt.plot(np.arange(-transWindowToLook,transWindowToLook,1e3/fs),\
                        analogSignal[int(transitionTime-transWindowToLook*fs/1e3):\
                                int(transitionTime+transWindowToLook*fs/1e3)],'gray')
            plt.axvline(0)
            plt.show()
    return triggerStart, triggerStop

def get_analog_times(analog_signal, fs=20000, lowpassFilter=True, lowPassfilterBand=500.0, histBins=4,
                 start_sample=0, last_sample=None, upTransition=False, outputPlot=False):
    """
    Wrapper for detectTransitions function to return start and end transition TIMES instead of signal indices

    Args:
        analogSignal (array-like): Signal to detect transitions

        fs (int; default is intan rate = 20000 hz): signal sampling rate 

        lowpassFilter (bool): whether to filter signal before transition detection

        lowPassfilterBand (int): lowpass filter cutoff for signal when lowpassFilter == True
        
        start_sample (int): First sample index (in 20e3 hz rate) of signal to detect

        last_sample (int or None; default: None (entire signal duration)): last sample to detect transition (in 20e3 hz) 

        histBins (int): How many histogram bins used to digitize signa
        
        upTransition (bool): whether signal transitions are up or down

        outputPlot (bool): whether to show binned signal histogram plot

    Returns:
        start_times (np.ndarray): signal transition start times
        end_times (np.ndarray): signal transition end times
    """
    # We want the rising edges of the scope TTL
    if analog_signal is not None:
        analog_signal = np.asarray(analog_signal)
        if last_sample is None:
            last_sample = len(analog_signal) - 1
        print(f'Getting transitions from {0}:{last_sample / fs} s')
        starts, ends = detectTransitions(analog_signal, earliestSample=start_sample, latestSample=last_sample, histBinNumber=histBins,
            upTransition=upTransition, lowpassFilter=lowpassFilter, fs=fs, lowPassfilterBand=lowPassfilterBand, outputPlot=outputPlot)
        start_times = starts / fs
        end_times = ends / fs
        print(f'Found {len(start_times)} raw triggers')   
    else:
        return None, None
    return start_times, end_times

def align_scope_triggers_to_frames(s2p_output, scope_times):
    """
    ERROR CHECKING FUNCTION: ENSURE NUMBER 2P FRAMES AND NUM TRIGGERS ARE SAME LENGTH AND CORRECT

    Args:
        s2p_output (Suite2POutput object from getSuite2POutput.py): s2p object output 

        scope_times (array-like): current detected times of scope pulses 

    Returns:
        scope_times, scope_end_times (array-like, array-like): adjusted start and end times of 2P scope

    """
    print(f'Aligning triggers... {len(scope_times)} scope times and {s2p_output.nframes} s2p frames')
    if len(scope_times) != s2p_output.nframes:
        ### FIX ERROR WHERE MORE FRAMES THAN TRIGGERS (INTAN TURNED OFF EARLY): 
        # add -1 to end of scope times where no frame found
        if len(scope_times) < s2p_output.nframes:
            frame_num_diff = s2p_output.nframes - len(scope_times)
            print(f'WARNING!!! Found {len(scope_times)} triggers but {s2p_output.nframes} frames. \
                    Adding {frame_num_diff} from experiment to match trigger counts...')
                        # add fake triggers to end when scope still on and experiment end 
            extra_frame_times = [-1] *  frame_num_diff
            print(f'Adding {frame_num_diff} extra scope trigger times for frames')
            scope_times = np.append(scope_times, extra_frame_times)
            scope_times_end = np.append(scope_times_end, extra_frame_times)

        ## FIX ERROR WHERE MORE TRIGGERS THAN FRAMES (SCOPE TURNED OFF EARLY)
        # truncate trigger times to num frames 
        elif len(scope_times) > s2p_output.nframes:
            scope_times = scope_times[:s2p_output.nframes]
            scope_times_end = scope_times[:s2p_output.nframes]

    # CHECK TRIGGER START AND STOP LENS
    # make sure scope start and end triggers are same length, if not add fake transitions to 
    # to shorter transition times list 
    scope_triggers_diff = len(scope_times_end) - len(scope_times)
    if scope_triggers_diff != 0:
        print(f'WARNING!! Found a difference of {scope_triggers_diff} for stop - start scope triggers.')
        transitions_to_append = [-1] * abs(scope_triggers_diff)
        if scope_triggers_diff > 0:
            print(f'Appending {len(transitions_to_append)} empty transitions to start scope times...')
            scope_times = np.append(scope_times, transitions_to_append)
        elif scope_triggers_diff < 0:
            print(f'Appending {len(transitions_to_append)} empty transitions to end scope times...')
            scope_times_end = np.append(scope_times_end, transitions_to_append)

    return scope_times, scope_times_end

def stack_traces(list_of_arrays, fill_value=np.nan):
    """
    Stacks a list of 1D or 2D NumPy arrays with inconsistent column dimensions
    by padding the shorter arrays.

    Parameters
    ----------
    list_of_arrays : list of np.ndarray
        A list of 1D (T,) or 2D (N×T) arrays.
    fill_value : scalar
        Value used to pad shorter arrays (default = np.nan).

    Returns
    -------
    stacked : np.ndarray
        2D array with shape (sum(rows), max(columns)).
    """
    # Filter out None/empty entries
    arrays = [np.asarray(a) for a in list_of_arrays if a is not None and a.size > 0]
    if not arrays:
        return np.array([])

    # Coerce all to 2D
    arrays_2d = []
    for a in arrays:
        if a.ndim == 1:
            a = a[np.newaxis, :]   # shape → (1, T)
        elif a.ndim > 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {a.shape}")
        arrays_2d.append(a)

    # Find max # of columns
    max_cols = max(a.shape[1] for a in arrays_2d)

    # Pad all arrays to that length
    padded = []
    for a in arrays_2d:
        n_rows, n_cols = a.shape
        if n_cols < max_cols:
            pad = np.full((n_rows, max_cols), fill_value, dtype=float)
            pad[:, :n_cols] = a
            padded.append(pad)
        else:
            padded.append(a)

    # Stack vertically
    stacked_output = np.vstack(padded)
    stacked_output = stacked_output.dropna(axis=1, how='all')
    return stacked_output

def restrict_traces(arrays, *, align="left", pad=False, fill=np.nan):
    """
    Make a stack with a common time length from a list of arrays.
    - Accepts 1D (T,) or 2D (N x T) arrays.
    - If pad=False (default): truncate each to the minimum T.
      align: 'left' | 'right' | 'center'
    - If pad=True: pad each to the maximum T with 'fill' instead of truncating.
    """
    # Clean & coerce to 2D: 1D -> (1, T)
    norm = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        elif arr.ndim > 2:
            raise ValueError(f"Expected 1D/2D arrays, found shape {arr.shape}")
        norm.append(arr)

    if not norm:
        return None

    lens = [a.shape[1] for a in norm]
    min_T, max_T = int(min(lens)), int(max(lens))

    if not pad:
        # Truncate to min length
        out = []
        for a in norm:
            if align == "left":
                out.append(a[:, :min_T])
            elif align == "right":
                out.append(a[:, -min_T:])
            elif align == "center":
                start = (a.shape[1] - min_T) // 2
                out.append(a[:, start:start + min_T])
            else:
                raise ValueError("align must be 'left', 'right', or 'center'")
        return np.vstack(out)
    else:
        # Pad to max length
        out = []
        for a in norm:
            pad_left = pad_right = 0
            need = max_T - a.shape[1]
            if need <= 0:
                out.append(a); continue
            if align == "left":
                pad_right = need
            elif align == "right":
                pad_left = need
            elif align == "center":
                pad_left = need // 2
                pad_right = need - pad_left
            else:
                raise ValueError("align must be 'left', 'right', or 'center'")
            out.append(np.pad(a, ((0,0),(pad_left,pad_right)),
                              mode="constant", constant_values=fill))
        return np.vstack(out)
    
def resample_traces(traces, frame_times, len_cam_times):
    """
    Resample 2P traces to match sampling rate of other data/signals 

    Args:
        traces (2D np.ndarray of shape (num_cells, num_frames)): Calcium traces to resample
        frame_times (array-like): frame times of scope
        len_cam_times (int): number of samples to stretch traces to

    Returns:
        traces_resampled (np.ndarray): resampled calcium traces matching fps of time samples 
    
    """
    from scipy.signal import resample
    # if 1d array reshape to create 2d for resampling
    if traces.ndim == 1:
        traces = traces.reshape(1, -1)
    resampled_traces = resample(traces, len_cam_times, t=frame_times, axis=1)
    traces_resampled, resampled_timestamps = resampled_traces
    return traces_resampled

def debug_triggers(signal, earliestSample=0, signal_fs=20e3):
  """
  Debugging function to check trigger durations and determine if they align with expected timings of signal
  Output digitized histogram plot to see transition amplitude bins

  Args:
    signal (array-like): signal to detect transitions bug 

    earliestSample (int): first index of sample to detect transitions

    signal_fs (int; default intan fs = 20e3): sampling rate of signal

  Returns:
    triggers_df (pd.DataFrame): dataframe containing transition starts, ends, durations, and times between triggers
    use to examine output of transitions function
  
  """
  starts, stops = detectTransitions(signal, earliestSample=earliestSample, outputPlot=True)

  start_times, stop_times = starts/signal_fs,stops/signal_fs

  triggers_df = pd.DataFrame({
  'start':start_times, 'end':stop_times, 'trigger dur': stop_times - start_times,
  'time_since_last_trigger': np.diff(start_times, append=True)})
  
  return triggers_df

def motion_to_2p_bins(frame_times, motion, motion_times, max_gap_s=0.5, fill_value='interpolate'):
    ft = np.asarray(frame_times, float)
    mt = np.asarray(motion_times, float)
    m  = np.asarray(motion, float)

    if ft.size == 0:
        return np.array([], dtype=float)

    # Build 2P bin edges (robust to very short recordings)
    edges = np.empty(ft.size + 1, float)
    if ft.size >= 2:
        edges[1:-1] = 0.5 * (ft[:-1] + ft[1:])
        edges[0]    = ft[0] - (ft[1] - ft[0]) / 2.0
        edges[-1]   = ft[-1] + (ft[-1] - ft[-2]) / 2.0
    else:
        # single frame: make a tiny bin around it
        half = 0.5
        edges[0] = ft[0] - half
        edges[1] = ft[0] + half

    # Bin-mean motion into frames
    bin_idx = np.digitize(mt, edges) - 1  # [-1..T2P]
    motion_2p = np.full(ft.size, np.nan, float)
    for k in range(ft.size):
        in_bin = (bin_idx == k)
        if np.any(in_bin):
            motion_2p[k] = np.nanmean(m[in_bin])

    # Fill NaNs without dropping frames
    ok_mask = np.isfinite(motion_2p)
    if ok_mask.sum() == 0:
        # nothing valid at all → just return NaNs (or choose a constant if you prefer)
        return motion_2p

    if fill_value == 'interpolate':
        motion_2p[~ok_mask] = np.interp(ft[~ok_mask], ft[ok_mask], motion_2p[ok_mask])
    elif fill_value == 'nearest':
        # ---- FIXED: use count of valid points, not len(ok_mask) ----
        x = ft[ok_mask]
        y = motion_2p[ok_mask]
        if x.size == 1:
            motion_2p[~ok_mask] = y[0]
        else:
            pos = np.searchsorted(x, ft, side='left')
            pos = np.clip(pos, 1, x.size - 1)  # clip using x.size, not len(ok_mask)
            left_x, right_x = x[pos - 1], x[pos]
            use_left = np.abs(ft - left_x) <= np.abs(ft - right_x)
            fill_idx = ~ok_mask & use_left
            motion_2p[fill_idx] = y[pos - 1][fill_idx]
            fill_idx = ~ok_mask & ~use_left
            motion_2p[fill_idx] = y[pos][fill_idx]
    else:
        # constant fill
        motion_2p[~ok_mask] = float(fill_value)

    # Handle large source gaps without dropping frames
    if mt.size >= 2:
        dt_src = np.diff(mt)
        big_gap = (dt_src > max_gap_s)
        if big_gap.any():
            for i in np.where(big_gap)[0]:
                bad = (ft > mt[i]) & (ft < mt[i + 1])
                # keep current filled values; if you want to mark them:
                # motion_2p[bad] = np.nan  # or a sentinel
                # pass for now
                pass

    return motion_2p

def categorical_to_2p(frame_times, state_values, state_times):
    """
    Align categorical (string) states to 2P frame times by nearest time sample.
    Returns: array of states, one per frame.
    """
    ft = np.asarray(frame_times)
    st = np.asarray(state_times)
    vals = np.asarray(state_values)

    # find nearest state_time index for each 2P frame
    idx = np.searchsorted(st, ft, side='left')
    idx = np.clip(idx, 0, len(st)-1)

    # choose whichever state_time is actually closer
    prev = np.maximum(idx-1, 0)
    next_ = np.minimum(idx, len(st)-1)
    nearer = np.where(
        np.abs(ft - st[prev]) < np.abs(ft - st[next_]),
        prev,
        next_
    )

    return vals[nearer]


def make_aligned_frame_df(frame_times, state_df):
    motion_2p = motion_to_2p_bins(frame_times, state_df['motion'], state_df['time'])
    state_2p = categorical_to_2p(frame_times, state_df['state'], state_df['time'])
    aligned_df = pd.DataFrame({
        "frame_time": frame_times,
        "motion_energy": motion_2p,
        "state": state_2p
    })
    return aligned_df

def peri_event_average(signal, t, events, win=(-2, 2), fs=None):
    # returns time-relative grid and average trace
    pre, post = win
    assert fs is not None or np.allclose(np.diff(t), 1/np.mean(np.diff(t)))
    dt = np.median(np.diff(t)); fs = fs or (1/dt)
    npre, npost = int(abs(pre)*fs), int(abs(post)*fs)
    stacks = []
    for ev in events:
        i = np.searchsorted(t, ev)
        if i-npre < 0 or i+npost >= len(signal): continue
        stacks.append(signal[i-npre:i+npost])
    if not stacks:
        return None, None
    S = np.vstack(stacks)
    tr = np.linspace(pre, post, S.shape[1], endpoint=False)
    return tr, S.mean(0), S

def bin_spikes(spike_times, t_lfp):
    idx = np.clip(np.searchsorted(t_lfp, spike_times), 0, len(t_lfp)-1)
    train = np.zeros_like(t_lfp, dtype=float)
    np.add.at(train, idx, 1.0)
    return train

def resample_to_grid(x_times, x_values, new_times):
    # assumes x_times are sorted and cover new_times range
    return np.interp(new_times, x_times, x_values)
# def match_column_by_nearest_time(
#     camera_times,
#     frame_df,
#     time_col: str,
#     value_col: str,
#     method: str = "nearest",      # "nearest" | "backward" | "forward"
#     tolerance=None,               # None, float seconds, or pd.Timedelta
#     dedup: str = "last",          # how to resolve duplicate times in frame_df: "last"|"first"|None
#     return_indices: bool = False  # also return the chosen row indices into frame_df (after sort/dedup)
# ):
#     """
#     Map each camera time to a value from frame_df[value_col] chosen by temporal proximity.

#     Parameters
#     ----------
#     camera_times : array-like of timestamps (float seconds or datetime-like)
#     frame_df     : DataFrame with at least [time_col, value_col]
#     time_col     : name of the time column in frame_df
#     value_col    : name of the value column to sample from frame_df
#     method       : "nearest" (default), "backward" (last <= t), or "forward" (first >= t)
#     tolerance    : optional max time gap allowed (float seconds or pd.Timedelta).
#                    If provided and the nearest/selected sample is farther than tolerance,
#                    the output will be set to None for that camera time.
#     dedup        : drop duplicate frame times keeping the "last" or "first" before matching.
#     return_indices : if True, also return the integer indices (into the sorted/deduped view).

#     Returns
#     -------
#     values : np.ndarray of length len(camera_times)
#     (indices) : optional np.ndarray of chosen indices into the sorted/deduped frame_df view
#     """
#     if time_col not in frame_df or value_col not in frame_df:
#         raise KeyError("time_col or value_col not found in frame_df")

#     # Prepare source times/values (sorted, optional de-dup)
#     src = frame_df[[time_col, value_col]].copy()
#     src = src.sort_values(time_col, kind="mergesort")  # stable sort
#     if dedup in ("first", "last"):
#         src = src.drop_duplicates(subset=time_col, keep=dedup)

#     # Convert times to a common numeric axis (seconds) while supporting datetimes
#     def _to_seconds(a):
#         s = pd.Series(a)
#         if pd.api.types.is_datetime64_any_dtype(s):
#             return pd.to_datetime(s).view("int64") / 1e9  # ns -> s
#         return s.astype(float).to_numpy()

#     t_src = _to_seconds(src[time_col].to_numpy())
#     v_src = src[value_col].to_numpy()
#     t_cam = _to_seconds(camera_times)

#     if len(t_src) == 0:
#         raise ValueError("frame_df has no rows after optional dedup.")
#     n = len(t_src)

#     # Optional tolerance (seconds)
#     if tolerance is None:
#         tol_sec = None
#     elif isinstance(tolerance, pd.Timedelta):
#         tol_sec = tolerance.total_seconds()
#     else:
#         tol_sec = float(tolerance)

#     # Vectorized index selection
#     idx = np.searchsorted(t_src, t_cam, side="left")

#     if method == "backward":
#         pick = np.clip(idx - 1, 0, n - 1)
#     elif method == "forward":
#         pick = np.clip(idx, 0, n - 1)
#     elif method == "nearest":
#         left = np.clip(idx - 1, 0, n - 1)
#         right = np.clip(idx, 0, n - 1)
#         # choose nearer; on ties, prefer left (earlier)
#         choose_right = np.abs(t_src[right] - t_cam) < np.abs(t_src[left] - t_cam)
#         pick = left.copy()
#         pick[choose_right] = right[choose_right]
#     else:
#         raise ValueError("method must be 'nearest', 'backward', or 'forward'")

#     # Apply tolerance if requested
#     if tol_sec is not None:
#         dt = np.abs(t_src[pick] - t_cam)
#         invalid = dt > tol_sec
#     else:
#         invalid = np.zeros_like(pick, dtype=bool)

#     # Build output (preserve dtype if possible; fall back to object when mixing None)
#     out = v_src[pick].copy()
#     if invalid.any():
#         # ensure we can place None without error
#         if out.dtype.kind in "fiu":  # numeric -> promote to object to hold None
#             out = out.astype(object)
#         out[invalid] = None

#     if return_indices:
#         return out, pick
#     return out