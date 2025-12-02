from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import traceback
from utils.filtering import *
from state import extractStates
from utils.getFPS import *
from utils import stats

'''  ***** MAIN METHOD TO SAVE STATE VARIABLE DATAFRAME FOR EASY USE LATER*****  '''
def get_state_df(facemap_data, camera_times, 
                 treadmill_signal=None, 
                  cam_fps=None,
                    filter_treadmill=True, smoothing_kernel=5,
                    movement_percentile=70, min_duration_s=1.0,
                    pupil_data=False,
                    paw_data=False,
                    # what index is motion saved in facemap,
                    use_motsvd = False,
                    motion_indices = {'motion':1},
                    annotate_state=False,
                    min_max_norm=True):
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
        movement_percentile (int; default: 70): percentile threshold for movement detection

    Returns:
        state_dataframe (pandas DataFrame): Loaded facemap data 
    """

    def clip_percentiles(x, pmin=0.1, pmax=99.9):
        lo, hi = np.percentile(x, [pmin, pmax])
        return np.clip(x, lo, hi)

    
    if 'motion' not in motion_indices:
        raise ValueError(f'Could not read facial motion data from facemap because a motion index was not provided.')

    if use_motsvd:
        motion_raw = np.nanmean(facemap_data['motSVD'][motion_indices['motion']], axis=1)
    else:
        motion_raw = facemap_data['motion'][motion_indices['motion']]  # (1D np array)

    t0 = camera_times[0]
    total_duration = camera_times[-1] - camera_times[0]
    if cam_fps is None:
        cam_fps = get_camera_fps(camera_times)
        print(f"Estimated camera FPS ≈ {cam_fps:.3f}")

    min_duration_frames = int(cam_fps * min_duration_s)

    # Smooth signals with gaussian filter 
    # --- EDIT (safeguard): remove cliffs before smoothing/thresholding ---
    motion_raw = clip_percentiles(despike_cliffs(motion_raw, k=9, z=6.0, only_drops=False))
    motion_smoothed = gaussian_filter1d(motion_raw, sigma=smoothing_kernel)
    print(f'Facemap motion signal contains {len(motion_smoothed)} frames, camera captured {len(camera_times)} frames.')

    # get boolean array of motion values 
    def get_rescaled_signal(filtered_signal, movement_percentile):
        threshold = np.percentile(filtered_signal, movement_percentile)
        print(f'Threshold for signal {threshold}')
        # Create boolean array of whether motion value is in top 70th percentile 
        in_percentile_bool = filtered_signal > threshold
        # Get differences between motion boolean values
        # Rising difference (Not moving -> moving) = 1 
        # Falling difference (Moving -> not moving) = -1 
        motion_diffs = np.diff(in_percentile_bool.astype(np.int8), prepend= in_percentile_bool[0])        
        rising_indices = np.where(motion_diffs == 1)[0] 
        falling_indices = np.where(motion_diffs == -1)[0] 
        # Handle start/end inside an ON segment
        if in_percentile_bool[0] and (
            len(rising_indices) == 0 or (len(rising_indices) and rising_indices[0] > falling_indices[0])):
            rising_indices = np.r_[0, rising_indices] 

        if in_percentile_bool[-1] and (len(falling_indices) == 0 or (
            len(rising_indices) and falling_indices[-1] < rising_indices[-1])):
            falling_indices = np.r_[falling_indices, len(in_percentile_bool)]  
        # set bools to false where num frames dont match min duration
        #bool_chunk = np.zeros_like(in_percentile_bool, dtype=bool)  
        bool_chunk = in_percentile_bool.copy()
        for start, end in zip(rising_indices, falling_indices):
            if (end - start) < min_duration_frames:
                bool_chunk[start:end] = 0

#     # ---- Time-based alignment to camera triggers ----
#     # Use trigger times and cam_fps to pick nearest motion-frame index, handle irregular trigger spacing
#     # Assumes camera_times are in seconds and aligned to the same timebase as the motion signal (frame 0 at camera_times[0]).
#     # If motion is sampled at the camera frame rate, frame index ≈ round((t - t0) * cam_fps).
        signal_fs = (len(filtered_signal) - 1) / total_duration
        rescaled_indices = np.round((camera_times - t0) * signal_fs).astype(int)  
        rescaled_indices = np.clip(rescaled_indices, 0, len(bool_chunk) - 1) 
        stretched_signal_smooth = filtered_signal[rescaled_indices]
        return rescaled_indices, stretched_signal_smooth, bool_chunk[rescaled_indices]
    
    rescaled_indices_motion, facial_motion_stretched, motion_bool = get_rescaled_signal(motion_smoothed, movement_percentile)
    print("motion_bool True ratio:", motion_bool.mean())

    treadmill_smoothed = None
    if treadmill_signal is not None:
        treadmill_smoothed = treadmill_signal
        if filter_treadmill:
            treadmill_smoothed = butter_lowpass_filter(treadmill_signal)
        # --- EDIT (safeguard): remove cliffs on treadmill as well ---
        treadmill_smoothed = clip_percentiles(despike_cliffs(treadmill_smoothed, k=25, z=8.0, only_drops=False))
        rescaled_indices_treadmill, treadmill_stretched, locomotion_bool = get_rescaled_signal(treadmill_smoothed, movement_percentile)

    pupil = None
    pupil_stretched = None
    if 'pupil' in facemap_data and pupil_data:
        pupil = facemap_data['pupil']
        pupil_area = pupil[0]['area_smooth']
        pupil_smoothed = gaussian_filter1d(pupil_area, sigma=smoothing_kernel)
        rescaled_indices_pupil, pupil_stretched, pupil_bool = get_rescaled_signal(pupil_smoothed, movement_percentile)

    left_paw_raw, right_paw_raw = None, None
    left_paw_smoothed, right_paw_smoothed = None, None
    if paw_data:
        print('Getting paw data....')
        if 'left_paw' in motion_indices.keys():
            #left_paw_raw = facemap_data['motion'][motion_indices['left_paw']]
            left_paw_raw = np.nanmean(facemap_data['motSVD'][motion_indices['left_paw']], axis=1)
            left_paw_raw = clip_percentiles(despike_cliffs(left_paw_raw, k=9, z=6.0, only_drops=False))
            left_paw_smoothed = gaussian_filter1d(left_paw_raw, sigma=smoothing_kernel)
            rescaled_indices_left_paw, left_paw_stretched, left_paw_bool = get_rescaled_signal(left_paw_smoothed, movement_percentile)
            print('Got data for left paw!')
        else:
            print('Facemap motion index was not provided for left paw!!')
        if 'right_paw' in motion_indices.keys():
            #right_paw_raw = facemap_data['motion'][motion_indices['right_paw']]
            right_paw_raw = np.nanmean(facemap_data['motSVD'][motion_indices['right_paw']], axis=1)
            right_paw_raw = clip_percentiles(despike_cliffs(right_paw_raw, k=9, z=6.0, only_drops=False))
            right_paw_smoothed = gaussian_filter1d(right_paw_raw, sigma=smoothing_kernel)
            rescaled_indices_right_paw, right_paw_stretched, right_paw_bool  = get_rescaled_signal(right_paw_smoothed, movement_percentile)
            print('Got data for right paw!')
        else:
            print('Facemap motion index was not provided for right paw!!')
    
    if min_max_norm:
        facial_motion_stretched = stats.min_max_norm(facial_motion_stretched)
    facial_motion_df = pd.DataFrame({
        'time': camera_times,
        'motion_bool': motion_bool,
        'motion': facial_motion_stretched,})

    if treadmill_signal is not None:
        treadmill_df = pd.DataFrame({
            'treadmill': treadmill_stretched,
        }) 
        facial_motion_df.insert(loc=2, column='locomotion_bool', value=locomotion_bool)  
        facial_motion_df = pd.concat([facial_motion_df, treadmill_df], axis=1)

    if pupil is not None:
        facial_motion_df['pupil_area'] = pupil_stretched
        facial_motion_df.insert(loc=3, column='pupil_bool', value=pupil_bool)  
    
    if left_paw_smoothed is not None:
        print('Adding left paw data')
        if min_max_norm: 
            facial_motion_df['left_paw'] = stats.min_max_norm(left_paw_stretched)
        facial_motion_df['left_paw'] = left_paw_stretched
        facial_motion_df['left_paw_bool'] = left_paw_bool

    if right_paw_smoothed is not None:
        if min_max_norm: 
            facial_motion_df['right_paw'] = stats.min_max_norm(left_paw_stretched)
        facial_motion_df['right_paw'] = right_paw_stretched
        facial_motion_df['right_paw_bool'] = right_paw_bool

    facial_motion_df = pd.DataFrame(facial_motion_df)  # no-op; keeps structure
    if annotate_state:
        facial_motion_df['state'] = [None] * len(facial_motion_df)
        facial_motion_df = facial_motion_df.apply(lambda x: extractStates.annotate_state(x), axis=1) 
    # fill spontaneous stim types as spontaneous blocks
    return facial_motion_df




# def get_state_df(
#     facemap_data, 
#     camera_times, 
#     treadmill_signal=None, 
#     treadmill_data=True,
#     cam_fps=30, 
#     smoothing_kernel=5, 
#     movement_percentile=70, 
#     min_dur_s=3,
#     to_parquet=False,
#     parquet_output_name = None,
#     to_csv = False,
#     csv_output_name = None
# ):
#     """
#     Function to get an easy to use dataframe that aligns timing information with state:
#     Saves raw face motion, pupil area, and locomotion (treadmill) information
#     Boolean columns (containing 0 or 1; 0 == False, 1 == True) indicate whether animal was moving at time

#     Args:
#         facemap_data (dict): Loaded facemap data, can be extracted from get_facemap_data function
#         camera_times (numpy arr): Camera times from triggers 
#         treadmill_signal (numpy arr; default:None): Raw treadmill analog signal from intan
#         treadmill_data (bool; default:True): whether you have treadmill data/have it stored
#         cam_fps (float; default: 30): sampling rate of cam, usually 30 hz
#         treadmill_fps (float; default: 20e3): sampling rate of intan (20000 hz)
#         smoothing_kernel (int; default: 5): sigma factor to smooth motion signal
#         movement_percentile (int; default: 70): percentile threshold for movement detection

#     Returns:
#         state_dataframe (pandas DataFrame): Loaded facemap data 
#     """

#     # ---------- Helper to avoid redefining logic for motion vs treadmill ----------
#     def get_motion_signal(
#         raw_signal: np.ndarray,
#         camera_times: np.ndarray,
#         smoothing_kernel: int,
#         movement_percentile: float,
#         cam_fps: float,
#         min_dur_s: float
#     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
#         """
#         Shared pipeline:
#         - Smooth with gaussian
#         - Percentile threshold → boolean
#         - Rising/Falling edge detection & start/end handling
#         - Enforce minimum duration (in frames at cam_fps)
#         - Compute per-signal sampling rate via total_time and index by nearest time
#         - Stretch raw / smooth / boolean to camera_times

#         Returns:
#             smoothed, bool_chunk (post-min-dur), rescaled_indices, stretched_bool, stretched_smooth, stretched_raw, signal_fs
#         """
#         # Smooth signals with gaussian filter
#         smoothed = gaussian_filter1d(raw_signal, sigma=smoothing_kernel)

#         # Set threshold for movement. Default: 70
#         thresh = np.percentile(smoothed, movement_percentile)

#         # Create boolean array of whether value is in top percentile
#         bool0 = smoothed > thresh

#         # Get differences between boolean values
#         # Rising difference (Not moving -> moving) = 1 
#         # Falling difference (Moving -> not moving) = -1 
#         diffs = np.diff(bool0.astype(np.int8), prepend=bool0[0])
#         rising = np.where(diffs == 1)[0]
#         falling = np.where(diffs == -1)[0]

#         # Handle start/end inside an ON segment
#         if bool0[0] and (len(rising) == 0 or (len(rising) and rising[0] > falling[0])):
#             rising = np.r_[0, rising]
#         if bool0[-1] and (len(falling) == 0 or (len(rising) and falling[-1] < rising[-1])):
#             falling = np.r_[falling, len(bool0)]

#         # set minimum duration of movement -> default min_dur_s seconds
#         min_duration_frames = int(cam_fps * min_dur_s)
#         bool_chunk = np.zeros_like(bool0, dtype=bool)

#         # Fill segments with min-duration enforcement (pair rises with falls)
#         for start, end in zip(rising, falling):
#             if (end - start) >= min_duration_frames:
#                 bool_chunk[start:end] = True

#         # ---- Time-based alignment to camera triggers ----
#         # Use trigger times and signal_fs to pick nearest sample index
#         t0 = camera_times[0]
#         total_time = camera_times[-1] - camera_times[0]
#         signal_fs = (len(smoothed) - 1) / total_time if total_time > 0 else 0.0

#         rescaled_indices = np.round((camera_times - t0) * signal_fs).astype(int)
#         rescaled_indices = np.clip(rescaled_indices, 0, len(bool_chunk) - 1)

#         # Stretch/interpolate (nearest by time)
#         stretched_bool = bool_chunk[rescaled_indices].astype(int)
#         stretched_smooth = smoothed[rescaled_indices]
#         stretched_raw = raw_signal[rescaled_indices]

#         return smoothed, bool_chunk, rescaled_indices, stretched_bool, stretched_smooth, stretched_raw, signal_fs

#     # -------------------- Motion branch (kept names/comments) --------------------
#     motion_1 = facemap_data['motion'][1]  # (1D np array)

#     # Pupil data: optional, if key in facemap data then use it
#     pupil = None
#     if 'pupil' in facemap_data:
#         pupil = facemap_data['pupil']
#         pupil_area = pupil[0]['area_smooth']

#     # Use shared helper for motion
#     motion_smoothed, facial_motion_chunk, rescaled_indices_motion, facial_motion_stretched, fMot_smooth_stretched, fMot_raw_stretched, motion_fs = \
#         get_motion_signal(motion_1, camera_times, smoothing_kernel, movement_percentile, cam_fps, min_dur_s)

#     print(f'Facemap motion signal contains {len(motion_smoothed)} frames, camera captured {len(camera_times)} frames.')

#     # #Percent above threshold on the SAME array used to threshold
#     motion_threshold = np.percentile(motion_smoothed, movement_percentile)
#     p70 = (motion_smoothed > motion_threshold).mean()
#     print("Motion threshold value is:", motion_threshold)

#     # pupil aligned to motion timebase (same indexing choice as before)
#     if pupil is not None:
#         pupil_stretched = pupil_area[rescaled_indices_motion]

#     # ------------------ Treadmill branch (no duplicate logic) --------------------
#     treadmill_smoothed = treadmill_raw_stretched = treadmill_smooth_stretched = None
#     treadmill_indices_rescaled = None
#     treadmill_fs = 0.0
#     treadmill_stretched = None
#     treadmill_chunk = None

#     if treadmill_data and treadmill_signal is not None:
#         (treadmill_smoothed,
#          treadmill_chunk,
#          treadmill_indices_rescaled,
#          treadmill_stretched,
#          treadmill_smooth_stretched,
#          treadmill_raw_stretched,
#          treadmill_fs) = get_motion_signal(
#             treadmill_signal, camera_times, smoothing_kernel, movement_percentile, cam_fps, min_dur_s)


#     # -------------------- Build DataFrame (kept names/structure) -----------------
#     facial_motion_df = pd.DataFrame({
#         'time': camera_times,
#         #'motion': facial_motion_stretched,
#         "motion_bool": facial_motion_stretched,
#         'motion_raw': fMot_raw_stretched,
#         'motion': fMot_smooth_stretched,
#     })

#     if pupil is not None:
#         facial_motion_df['pupil_area'] = pupil_stretched

#     if treadmill_data and treadmill_signal is not None:
#         treadmill_df = pd.DataFrame({
#             #'locomotion': treadmill_stretched,
#             "locomotion_bool": treadmill_stretched,
#             'treadmill_raw': treadmill_raw_stretched,          # uses treadmill indices (fixed)
#             'treadmill': treadmill_smooth_stretched,
#         }) 
#         facial_motion_df = pd.concat([facial_motion_df, treadmill_df], axis=1)

#     # Keep annotate_state usage and column name
#     facial_motion_df['state'] = [None] * len(facial_motion_df)
#     facial_motion_df = facial_motion_df.apply(lambda x: annotate_state(x), axis=1)

#     if to_parquet:
#         if parquet_output_name is not None:
#             if not parquet_output_name.endswith('.parquet'):
#                 parquet_output_name += '.parquet'
#             facial_motion_df.to_parquet(parquet_output_name)
#         else:
#             print(f'Did not save dataframe to parquet file. No output file name was provided.')

#     if to_csv:
#         if csv_output_name is not None:
#             if not csv_output_name.endswith('.csv'):
#                 csv_output_name += '.csv'
#             facial_motion_df.to_csv(csv_output_name)
#         else:
#             print(f'Did not save dataframe to CSV file. No output file name was provided.')

#     return facial_motion_df

