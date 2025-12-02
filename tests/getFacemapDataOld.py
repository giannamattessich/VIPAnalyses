import os, numpy as np
from scipy.ndimage import gaussian_filter1d
import pandas as pd
from utils.alignmentFunctions import butter_lowpass_filter

'''  ***** MAIN METHOD TO SAVE STATE VARIABLE DATAFRAME FOR EASY USE LATER*****  '''
def get_state_df(facemap_data, camera_times, treadmill_signal=None, treadmill_data=True,
                  cam_fps=30, min_dur_s=1.0, smoothing_kernel=5, movement_percentile=70):
    """
    Function to get an easy to use dataframe that aligns timing information with state:
    Saves raw face motion, pupil area, and locomotion (treadmill) information
    Boolean columns (containing 0 or 1; 0 == False, 1 == True) indicate whether animal was moving at time

    Args:
        facemap_data (dict): Loaded facemap data, can be extracted from get_facemap_data function

        camera_times (numpy arr): Camera times from triggers 

        treadmill_signal (numpy arr; default:None): Raw treadmill analog signal from intan

        treadmill_data (bool; default:True): whether you have treadmill data/have it stored

        cam_fps (float; default: 30): sampling rate of cam, usually 30 hz

        treadmill_fps (float; default: 20e3): sampling rate of intan (20000 hz)

        smoothing_kernel (int; default: 5): sigma factor to smooth motion signal

        min_dur_s (float; default: 1): how many seconds of movement to require as considered moving
        
        movement_percentile (int; default: 70): percentile threshold for movement detection

    Returns:
        state_dataframe (pandas DataFrame): Loaded facemap data 
    """
    motion_1 = facemap_data['motion'][1]  # (1D numpy array)
    # Optional pupil data
    pupil = None
    if 'pupil' in facemap_data:
        pupil = facemap_data['pupil']
        pupil_area = pupil[0]['area_smooth']
      # Set as appropriate
    motion_smoothed = gaussian_filter1d(motion_1, sigma=smoothing_kernel)

    motion_thresh = np.percentile(motion_smoothed, movement_percentile)
    #locomotion_thresh = np.percentile()
    motion_smoothed_bool = motion_smoothed > motion_thresh

    motion_diffs = np.diff(motion_smoothed_bool.astype(np.int8), prepend=0)
    rising_indices = np.where(motion_diffs == 1)[0] 
    falling_indices = np.where(motion_diffs == -1)[0] 
    
    # start/end inside an ON segment
    # find indices of rising edges of signal
    if motion_smoothed_bool[0] and (
        len(rising_indices) == 0 or (len(falling_indices) and rising_indices[0] > falling_indices[0])):
        rising_indices = np.r_[0, rising_indices]     
    # find indices of falling edges in signal
    if motion_smoothed_bool[-1] and (len(falling_indices) == 0 or (len(rising_indices) and falling_indices[-1] < rising_indices[-1])):
        falling_indices = np.r_[falling_indices, len(motion_smoothed_bool)]  


    min_duration_frames = int(cam_fps * min_dur_s)  # min duration of movement, ex. 1 second of motion required
    facial_motion_chunk = np.zeros_like(motion_smoothed_bool, dtype=bool)   # preallocte motion chunk arr

    ### Time  alignment to camera triggers
    # Use trigger times and cam_fps to pick nearest motion-frame index (handle irregular trigger spacing)
    # Assumes camera_times are in seconds and aligned to the same timebase as the motion signal (frame 0 at camera_times[0]).
    # If motion is sampled at the camera frame rate, frame index ≈ round((t - t0) * cam_fps).
    t0 = camera_times[0]
    total_time = camera_times[-1] - camera_times[0]
    motion_fs = (len(motion_smoothed) - 1) / total_time

    # Fill facial motion segments with min-duration enforcement
    for start, end in zip(rising_indices, falling_indices):
        if (end - start) >= min_duration_frames:
            facial_motion_chunk[start:end] = True

    # rescale indices 
    rescaled_indices = np.round((camera_times - t0) * motion_fs).astype(int)  
    rescaled_indices = np.clip(rescaled_indices, 0, len(facial_motion_chunk) - 1) 

    # Stretch/interpolate (nearest by time)
    facial_motion_stretched = facial_motion_chunk[rescaled_indices].astype(int)   # keep your int outputs
    fMot_smooth_stretched = motion_smoothed[rescaled_indices]
    fMot_raw_stretched = motion_1[rescaled_indices]

    # find percentage frames above threshold
    p70 = (motion_smoothed > motion_thresh).mean()
    print("Motion threshold value is:", p70)
    # after stretching to cam times, proportion should be ~unchanged values
    print("Proportion above threshold:", facial_motion_stretched.mean())
    
    if pupil is not None:
        pupil_stretched = pupil_area[rescaled_indices]  # align pupil via time indices too
        
    facial_motion_df = pd.DataFrame({
        'time': camera_times,
        'motion': facial_motion_stretched,
        'motion_raw': fMot_raw_stretched,
        'motion_smooth': fMot_smooth_stretched,
    })

    if pupil is not None:
        facial_motion_df['pupil_area'] = pupil_stretched

    if treadmill_data and treadmill_signal is not None:
        treadmill_signal_resampled, treadmill_moving_bool = get_locomotion(treadmill_signal,
                                                                           camera_times=camera_times,
                                                                           movement_percentile=movement_percentile)
        facial_motion_df['treadmill_raw'] = treadmill_signal_resampled
        facial_motion_df['locomotion']= treadmill_moving_bool
    return facial_motion_df

def get_locomotion(treadmill_signal, camera_times, min_gap=0.3, min_bout=1.0,
                    treadmill_fps=20e3, cam_fps=30, movement_percentile=70, edge_detect=True):
    
    '''
    Use treadmill signal to detect relative locomotion values. 
        Also return boolean at indices where animal is moving

    Args:
        treadmill_signal (np.array): raw tredmill signal recorded from intan

        camera_times (np.array): times of camera triggers

        min_gap (float): minimum gap between motion frames (seconds)

        min_bout (float): minimum time to consider movement (seconds)

        treadmill_fps (float; default is intan sampling rate (20e3)): sample rate of treadmill signal

        cam_fps (float; default: 30): sample rate of camera trigger

        movement_percentile (int; default 70): percentile to consider motion values as moving 

        edge_detect (bool; default:True): whether to include frames where movement occurs at edge         
    
    Returns:
        treadmill_resampled (np.array of int): list of locomotion values (scaled to motion fs)

        treadmill_motion_bool (np.array of bool): list of locomotion frames (1 == moving, 0 == stationary)
    
    '''
    # apply butter and filtfilt to smooth noisy treadmill signal
    treadmill_signal = butter_lowpass_filter(treadmill_signal, cutoff=30)
    # 1) time vector for treadmill samples, then interpolate to camera timestamps
    treadmill_time = np.arange(len(treadmill_signal), dtype=float) / float(treadmill_fps)
    # check if movement at edges, then interpolate
    if edge_detect:
        first_val, last_val = treadmill_signal[0], treadmill_signal[-1]
        treadmill_resampled = np.interp(camera_times, treadmill_time, treadmill_signal,
                                        left=first_val, right=last_val).astype(float)
    else:
        treadmill_resampled = np.interp(camera_times, treadmill_time, treadmill_signal,
                                        left=np.nan, right=np.nan).astype(float)

    # threshold values from treadmill signal
    if np.isfinite(treadmill_resampled).any():
        treadmill_threshold = np.nanpercentile(treadmill_resampled, movement_percentile)
    else:
        treadmill_threshold = np.nan  # degenerate case

    above = (treadmill_resampled > treadmill_threshold) & np.isfinite(treadmill_resampled)

    # 3) rising/falling edges (MATLAB: diff([0; mask])==1 and diff([mask; 0])==-1)
    movement_onsets  = np.where(np.diff(above.astype(np.int8),  prepend=0) ==  1)[0]
    movement_offsets = np.where(np.diff(above.astype(np.int8),  append=0) == -1)[0]

    # lengths can differ by at most 1 if recording starts/ends inside a bout.
    movement_min = min(len(movement_onsets), len(movement_offsets))
    movement_onsets, movement_offsets = movement_onsets[:movement_min], movement_offsets[:movement_min]

    # 4) N×2 intervals (inclusive indices)
    movement_intervals = np.column_stack([movement_onsets, movement_offsets]).astype(int)

    # 5) merge adjacent bouts whose OFF gap < minGap (in frames); enforce optional minDur
    minGap_frames = int(round(min_gap * cam_fps))   # e.g., merge gaps shorter than 0.3 s
    minDur_frames = int(round(min_bout * cam_fps))   # e.g., require ≥ 1.0 s ON

    movement_change = True
    while movement_change and len(movement_intervals) > 1:
        movement_change = False
        for idx in range(len(movement_intervals) - 1, 0, -1):  # walk backward when deleting rows
            gap = movement_intervals[idx, 0] - movement_intervals[idx - 1, 1]
            if gap < minGap_frames:
                movement_intervals[idx - 1, 1] = movement_intervals[idx, 1]  # extend previous
                movement_intervals = np.delete(movement_intervals, idx, axis=0)
                movement_change = True

    if movement_intervals.size:
        durations = movement_intervals[:, 1] - movement_intervals[:, 0] + 1
        movement_intervals = movement_intervals[durations >= minDur_frames]

    # 6) build a 0/1 mask on camera frames from intervals
    treadmill_motion_bool = np.zeros(len(camera_times), dtype=int)
    for start_idx, end_idx in movement_intervals:
        treadmill_motion_bool[start_idx:end_idx+1] = 1

    return treadmill_resampled, treadmill_motion_bool