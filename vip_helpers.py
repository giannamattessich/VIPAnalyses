import re, os, joblib
import pandas as pd, numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from utils.stats import *
from state.extractStates import *
from utils.alignmentFunctions import *
from state.motion_correlations import *
# get name of day for animal from a string

def get_day_from_path(recording_string):
    pattern = r'p\d{1,2}'
    match = re.search(pattern, recording_string)
    if match is None:
        return None
    return match.group()

def load_vip_data():
    if not os.path.exists('paths_df.parquet'):
        dataclasses_path = '/home/gianna/Desktop/PythonProjects/VIPxTigerAnalyses/dataclasses'
        stat_dfs_path = '/home/gianna/Desktop/PythonProjects/VIPxTigerAnalyses/state_dfs'

        back_left_data = [os.path.join(
            dataclasses_path, 'Back_left', recording_name) for recording_name in os.listdir(os.path.join(dataclasses_path, 'Back_left'))]
        back_left_data = sorted(back_left_data)
        back_left_data = [back_left_data[-1]] + back_left_data[:-1]

        back_left_sdf = [os.path.join(
            stat_dfs_path, 'Back_left', recording_name) for recording_name in os.listdir(os.path.join(stat_dfs_path, 'Back_left'))]
        back_left_sdf = sorted(back_left_sdf)
        back_left_sdf = [back_left_sdf[-1]] + back_left_sdf[:-1]
        #-------

        back_right_data = [os.path.join(
            dataclasses_path, 'Back_right', recording_name) for recording_name in os.listdir(os.path.join(dataclasses_path, 'Back_right'))]
        back_right_data = sorted(back_right_data)
        back_right_data = [back_right_data[-1]] + back_right_data[:-1]

        back_right_sdf = [os.path.join(
            stat_dfs_path, 'Back_right', recording_name) for recording_name in os.listdir(os.path.join(stat_dfs_path, 'Back_right'))]
        back_right_sdf = sorted(back_right_sdf)
        back_right_sdf = [back_right_sdf[-1]] + back_right_sdf[:-1]
        #---

        both_front_data = [os.path.join(
            dataclasses_path, 'Both_front', recording_name) for recording_name in os.listdir(os.path.join(dataclasses_path, 'Both_front'))]
        both_front_data = sorted(both_front_data)
        both_front_data = [both_front_data[-1]] + both_front_data[:-1]

        both_front_sdf = [os.path.join(
            stat_dfs_path, 'Both_front', recording_name) for recording_name in os.listdir(os.path.join(stat_dfs_path, 'Both_front'))]
        both_front_sdf = sorted(both_front_sdf)
        both_front_sdf = [both_front_sdf[-1]] + both_front_sdf[:-1]
        #---

        front_left_data = [os.path.join(
            dataclasses_path, 'Front_left', recording_name) for recording_name in os.listdir(os.path.join(dataclasses_path, 'Front_left'))]
        front_left_data = sorted(front_left_data)
        front_left_data = [front_left_data[-1]] + front_left_data[:-1]

        front_left_sdf = [os.path.join(
            stat_dfs_path, 'Front_left', recording_name) for recording_name in os.listdir(os.path.join(stat_dfs_path, 'Front_left'))]
        front_left_sdf = sorted(front_left_sdf)
        front_left_sdf = [front_left_sdf[-1]] + front_left_sdf[:-1]

        def fill_col(arr_to_fill):
            day_min, day_max = 9, 20
            curr_day = day_min
            new_arr = []

            for recording in arr_to_fill:
                try:
                    rec_day = int(get_day_from_path(recording)[1:])
                except:
                    new_arr.append(None)
                    continue

                # Fill missing days before this recording
                while rec_day > curr_day or rec_day is None:
                    new_arr.append(None)
                    curr_day += 1

                new_arr.append(recording)
                curr_day += 1

            # Fill remaining days up to day_max
            while curr_day <= day_max:
                new_arr.append(None)
                curr_day += 1
            return new_arr

        paths_dataframe = pd.DataFrame({'day': pd.Series([f'p{day}' for day in range(9, 20)]),
                                'Back_left_dataclass': pd.Series(fill_col(back_left_data)),
                                'Back_right_dataclass': pd.Series(fill_col(back_right_data)),
                                'Both_front_dataclass' : pd.Series(fill_col(both_front_data)),
                                'Front_left_dataclass': pd.Series(fill_col(front_left_data)),
                                'Back_left_state_df': pd.Series(fill_col(back_left_sdf)),
                                'Back_right_state_df': pd.Series(fill_col(back_right_sdf)),
                                'Both_front_state_df' : pd.Series(fill_col(both_front_sdf)),
                                'Front_left_state_df': pd.Series(fill_col(front_left_sdf))})
        paths_dataframe = paths_dataframe[~paths_dataframe['day'].isin(['p11', 'p18'])]
        paths_dataframe = paths_dataframe.dropna(how='all')
        paths_dataframe.to_parquet('paths_df.parquet', engine='pyarrow')
    else:
        paths_dataframe = pd.read_parquet('paths_df.parquet', engine='pyarrow')
    return paths_dataframe

def get_animal_names():
    return ['Back_left', 'Back_right', 'Both_front', 'Front_left']

def load_animal_data(animal_name):
    if not os.path.exists('paths_df.parquet'):
        raise ValueError('Paths df doesnt exist. run load_vip_data first to get')
    else:
        paths_df = pd.read_parquet('paths_df.parquet')
        dataclasses = paths_df[f'{animal_name}_dataclass']
        recording_days = np.where(~dataclasses.isnull())[0]
        recording_days = paths_df.iloc[recording_days]['day'].to_numpy()
        dataclasses = dataclasses.dropna().to_numpy()
        state_dfs = paths_df[f'{animal_name}_state_df'].dropna().to_numpy()
        dataclasses = [joblib.load(vip_obj) for vip_obj in dataclasses if vip_obj is not None]
        state_dfs = [pd.read_parquet(state_df) for state_df in state_dfs]
        s2p_outs = [data.s2p_out for data in dataclasses]
    return dataclasses, state_dfs, s2p_outs, recording_days

def load_animal_day_dict(animal_name):
    recordings = {}
    if not os.path.exists('paths_df.parquet'):
        raise ValueError('Paths df doesnt exist. run load_vip_data first to get')
    else:
        paths_df = pd.read_parquet('paths_df.parquet')
        dataclasses = paths_df[f'{animal_name}_dataclass']
        recording_days = np.where(~dataclasses.isnull())[0]
        recording_days = paths_df.iloc[recording_days]['day'].to_numpy()
        dataclasses = dataclasses.dropna().to_numpy()
        state_dfs = paths_df[f'{animal_name}_state_df'].dropna().to_numpy()
        dataclasses = [joblib.load(vip_obj) for vip_obj in dataclasses if vip_obj is not None]
        state_dfs = [pd.read_parquet(state_df) for state_df in state_dfs]
        s2p_outs = [data.s2p_out for data in dataclasses]
    return dataclasses, state_dfs, s2p_outs, recording_days    

def load_data_for_day(day):
    if not os.path.exists('paths_df.parquet'):
        raise ValueError('Paths df doesnt exist. run load_vip_data first to get')
    else:
        paths_df = pd.read_parquet('paths_df.parquet')
        print(f'Getting data for {day}')
        if not day.startswith('p'):
            raise ValueError(f'Provide day starting with p..')
        paths_df_cols = paths_df.columns
        dataclass_cols = [col for col in paths_df_cols if col.endswith('dataclass')]
        statedf_cols = [col for col in paths_df_cols if col.endswith('state_df')]
        day_row = paths_df[paths_df['day'] == day]
        dataclasses = day_row[dataclass_cols].values[0]
        recordings = dataclasses
        state_dfs = day_row[statedf_cols].values[0]
        dataclasses_day = [joblib.load(vip_obj) for vip_obj in dataclasses if vip_obj is not None]
        state_dfs_day = [pd.read_parquet(state_df) for state_df in state_dfs if state_df is not None]
        s2p_outs_day = [data.s2p_out for data in dataclasses_day]
        return dataclasses_day, state_dfs_day, s2p_outs_day, recordings

def sta_population_multirec(spikes, frame_times, motion, window_s=(-6,6), n_shuffles=500):
    """
    Average STA across multiple recordings with potentially different frame_times dt.
    - spikes: (n_cells, T)   # same T across recs (your pooled/stacked spikes)
    - frame_times: (R, T) or (T,)
    - motion: (R, T) or (T,)
    Returns: tau_ref, sta_mean_ref, sta_sem_ref, z_sta_ref on a common tau grid.
    """
    ft = np.asarray(frame_times)
    mo = np.asarray(motion)

    # If single 1D time/motion, just run once.
    if ft.ndim == 1 and mo.ndim == 1:
        return spike_triggered_average_motion(spikes, ft, mo, window_s, n_shuffles)

    if ft.ndim != 2 or mo.ndim != 2 or ft.shape != mo.shape:
        raise ValueError("frame_times and motion must both be (R, T) with same shape.")

    outs = []
    min_tau_lo = -np.inf
    max_tau_hi = +np.inf
    min_len = np.inf

    # Run STA per recording
    for r in range(ft.shape[0]):
        tau_r, m_r, s_r, z_r = spike_triggered_average_motion(
            spikes, ft[r], mo[r], window_s, n_shuffles
        )
        outs.append((tau_r, m_r, s_r, z_r))
        # track common overlap + shortest length
        min_tau_lo = max(min_tau_lo, tau_r.min())
        max_tau_hi = min(max_tau_hi, tau_r.max())
        min_len = min(min_len, tau_r.size)

    # Build a common tau grid within overlap, with the shortest length
    tau_ref = np.linspace(min_tau_lo, max_tau_hi, int(min_len))

    def _interp_on_ref(tau_r, arr_r):
        # arr_r is 1D length len(tau_r)
        return np.interp(tau_ref, tau_r, arr_r)

    means_ref = []
    z_ref = []
    for (tau_r, m_r, s_r, z_r) in outs:
        means_ref.append(_interp_on_ref(tau_r, m_r))
        z_ref.append(_interp_on_ref(tau_r, z_r))

    means_ref = np.vstack(means_ref)  # (R, L)
    z_ref = np.vstack(z_ref)          # (R, L)

    sta_mean_ref = np.nanmean(means_ref, axis=0)
    sta_sem_ref  = np.nanstd(means_ref, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(means_ref), axis=0))
    z_sta_ref    = np.nanmean(z_ref, axis=0)

    return tau_ref, sta_mean_ref, sta_sem_ref, z_sta_ref

def plot_correlation_hist_vip(day, plot_title='', bar_colors = ["#6200FF", "#4BEE00", "#939099"]):
    dataclasses, state_dfs, s2p_outs, recordings = load_data_for_day(day)
    spikes = [s2p_out.get_cell_spikes() for s2p_out in s2p_outs]
    corr_df_all = pd.DataFrame()
    for rec_idx in range(len(s2p_outs)):
        spike = spikes[rec_idx][:,:-1]
        aligned_time_df = dataclasses[rec_idx].make_aligned_frame_df(state_dfs[rec_idx])
        if len(aligned_time_df) < spike.shape[1]:
            spike = spike[:,:len(aligned_time_df)]
        print(aligned_time_df['motion'].shape)
        motion_z = zscore_robust(aligned_time_df['motion'], axis=0)
        spks_z   = zscore_robust(spike)

        # 3) Feed into your correlation/proportion function
        df_corr, prop = proportion_motion_correlated_cells(spks_z, motion_z, method='spearman', alpha=0.05)
        corr_df_all = pd.concat([corr_df_all, df_corr], ignore_index=True)
        print(f"Motion-correlated fraction for {recordings[rec_idx]}: {prop:.1%}")
    corr_df, counts = classify_correlation(corr_df_all)
    if plot_title == '':
        plot_title = day
    plot_correlation(corr_df, plot_title, bar_colors)

def plot_spike_ta_vip(day, title=''):
    dataclasses, state_dfs, s2p_outs, recordings = load_data_for_day(day)
    pop_spikes = [s2p_out.get_cell_spikes() for s2p_out in s2p_outs]
    pop_spikes = restrict_traces(pop_spikes)
    for rec_idx in range(len(recordings)):
        aligned_dfs = [dataclasses[rec_idx].make_aligned_frame_df(state_dfs[rec_idx])]
    frame_times = restrict_traces([aligned_df['frame_time'] for aligned_df in aligned_dfs])
    motion_2p = restrict_traces([aligned_df['motion'] for aligned_df in aligned_dfs])
        #tau, sta_mean, sta_sem, z_sta = spike_triggered_average_motion(
        #pop_spikes, frame_times, motion_2p, window_s=(-6, 6), n_shuffles=500)
    tau, sta_mean, sta_sem, z_sta = sta_population_multirec(pop_spikes, frame_times, motion_2p)
    if title == '':
        title = day
    plot_sta(tau, sta_mean, sta_sem, z_sta, title=title)

def pop_spikes_z_restricted(day, pop_spikes_df):
    if not (type(pop_spikes_df) == pd.DataFrame):
        pop_spikes_df = pd.read_parquet('outputs/pop_spikes_z_restricted.parquet')
    pop_spikes_day = pop_spikes_df[pop_spikes_df['day'] == day].iloc[:, 2:].to_numpy()
    print(pop_spikes_day)
    pop_spikes_restriced = restrict_traces(pop_spikes_day)
    pop_spikes_dayz = zscore_robust(pop_spikes_restriced)
    return pop_spikes_dayz

def pop_motion_z_restricted(day, pop_spikes_df):
    if not (type(pop_spikes_df) == pd.DataFrame):
        pop_spikes_df = pd.read_parquet('outputs/pop_spikes_z_restricted.parquet')
    pop_spikes_day = pop_spikes_df[pop_spikes_df['day'] == day].iloc[:, 2:].to_numpy()
    print(pop_spikes_day)
    pop_spikes_restriced = restrict_traces(pop_spikes_day)
    pop_spikes_dayz = zscore_robust(pop_spikes_restriced)
    return pop_spikes_dayz

# def get_restricted_spikes():

# def get_restricted_spikes_zscored():

# def get_spikes():

# def get_restricted_spikes_zscored():
