import numpy as np, matplotlib.pyplot as plt
import os, traceback
from utils.alignmentFunctions import restrict_traces
from utils.stats import zscore_robust
from twop.plots import *
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize

class Track2POutput:
    def __init__(self, track2p_path, plane='plane0'):
        self.track2p_path = track2p_path
        if not track2p_path.endswith('track2p'):
            if os.path.exists(os.path.join(track2p_path, 'track2p')):
                self.track2p_path = os.path.join(track2p_path, 'track2p')
            else:
                raise ValueError(f'Provided track2p path {track2p_path} does not exist !')
        try:
            self.match_mat = np.load(os.path.join(self.track2p_path, f'{plane}_match_mat.npy'), allow_pickle=True)
            print(f'Shape of match matrix for cells present: {self.match_mat.shape} (cells, days)')
            self.s2p_indices = np.load(os.path.join(self.track2p_path, f'{plane}_suite2p_indices.npy'), allow_pickle=True)
            self.track_ops = np.load(os.path.join(self.track2p_path, 'track_ops.npy'), allow_pickle=True).item()
            self.data_paths = self.track_ops['all_ds_path']
            self.days = [os.path.basename(path) for path in self.data_paths]
        except Exception:
            traceback.print_exc()

    def s2p_outs_from_path(self):
        s2p_basepath = os.path.join(os.path.dirname(self.track2p_path), 'suite2p')
        print(s2p_basepath)

    def get_longitudinal_cells_dict(self, s2p_outs, num_days_required=3):
        ### num_days_required -> NUMBER OF DAYS THAT REQUIRE CONTAINING CELL
        # match_matrix = self.match_mat[np.sum(self.match_mat != None, axis=1) >= num_days_required]
        # curr_cell = 0
        # tracked_cells_dict = {}
        # for row_idx in range(len(match_matrix)):
        #     tracked_cell = match_matrix[row_idx, :]
        #     valid_days = np.where(tracked_cell != None)[0]
        #     for day_idx in valid_days:
        #         cell_on_day = tracked_cell[day_idx]
        #         p_day = self.days[day_idx]
        #         spike_train_day = s2p_outs[p_day].spks[cell_on_day]
        #         if f'C{curr_cell}' not in tracked_cells_dict.keys():
        #             tracked_cells_dict[f'C{curr_cell}'] = {p_day:spike_train_day}
        #         else:
        #             tracked_cells_dict[f'C{curr_cell}'][p_day] = spike_train_day
        #     curr_cell += 1
        # return tracked_cells_dict
            row = np.sum(self.match_mat != None, axis=1) >= num_days_required
            match_matrix = self.match_mat[row]

            tracked_cells_dict = {}

            for track_id, row in enumerate(match_matrix):
                cell_key = f"C{track_id}"
                tracked_cells_dict[cell_key] = {}

                valid_days = np.where(row != None)[0]

                for day_idx in valid_days:
                    cell_on_day = row[day_idx]
                    p_day = self.days[day_idx]

                    # extract Suite2p spikes for that day's ROI
                    spikes = s2p_outs[p_day].spks[cell_on_day]

                    tracked_cells_dict[cell_key][p_day] = spikes
            print('fioxes')
            return tracked_cells_dict
    
    def plot_longitudinal_traces(self, tracked_cells_dict, colors = ["#00CCFF", "#2B70D6", "#090DEB", "#3102DA", "#480FE2"]):
        cell_idx = 0
        fig, axs = plt.subplots(len(list(tracked_cells_dict.values())[0]), len(tracked_cells_dict), figsize=(50,10))
        for cell, traces in tracked_cells_dict.items():
            #fig, axs = plt.subplots(len(traces), len(tracked_cells_dict), figsize=(60,10))
            trace_idx = 0
            for trace_day, trace in traces.items():
                axs[trace_idx, cell_idx].plot(trace, color=colors[trace_idx])
                axs[trace_idx, cell_idx].set_title(trace_day, fontsize=12, fontweight='bold')
                trace_idx += 1
            cell_idx += 1
        plt.show()

    def plot_per_day_raster(self, long_cells_dict, savefig=False, outpath=None):
        try:
            rec_days = list(long_cells_dict['C0'].keys())
        except:
            traceback.print_exc()
            print('No cells found in provided longitudinal dictionary !!')
        fig, axs = plt.subplots(1, len(rec_days), figsize = (12, 2))
        day_idx = 0
        for rec_day in rec_days:
            pop_traces = []
            for cell in long_cells_dict.keys():
                cell_trace_day = long_cells_dict[cell][rec_day]
                pop_traces.append(cell_trace_day)
            pop_spikes = restrict_traces(pop_traces)
            spks_z    = zscore_robust(pop_spikes, axis=1)
            vmin = np.nanpercentile(spks_z, 1)
            vmax = np.nanpercentile(spks_z, 99)
            # # robust display range from the *stacked* array
            # vmin = np.nanpercentile(spks_z, 1)
            # vmax = np.nanpercentile(spks_z, 99)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
                # fallback to safe defaults if array is constant/NaN
                vmin, vmax = -1.0, 1.0

            norm = Normalize(vmin=vmin, vmax=vmax)

            axs[day_idx].imshow(spks_z, aspect='auto', cmap='binary',
                                origin='lower', norm=norm)
            n_samples = spks_z.shape[1]
            duration_sec = n_samples / 30

            tick_step = 300   # whole-number steps
            ticks_sec = np.arange(0, duration_sec + tick_step, tick_step)
            ticks_idx = ticks_sec * 30

            axs[day_idx].set_xticks(ticks_idx)
            axs[day_idx].set_xticklabels([f"{int(t)}" for t in ticks_sec])
            axs[day_idx].set_xlabel("Time (s)")
            if spks_z.shape[0] == 1:
                axs[day_idx].yaxis.set_major_locator(mticker.NullLocator())
                axs[day_idx].yaxis.set_major_formatter(mticker.NullFormatter())
            #axs[day_idx].invert_yaxis()
            axs[day_idx].set_title(rec_day)
            day_idx +=1
        if savefig:
            if outpath is not None:
                if not outpath.endswith('.png'):
                    outpath += '.png'
                plt.savefig(outpath)
            else:
                print(f'savfig = True but outpath for figure is none. Set outpath to proper destination file for fixing.')
        plt.show()

    def get_data_fordays_dict(self):
        import joblib
        dataclasses_dict = {}
        state_dfs_dict = {}
        s2p_outs = {}
        for path in (self.data_paths):
                basename = os.path.basename(path)
                dataclass_path = os.path.join(path, f'{basename}_dataclass.joblib')
                dataclass = joblib.load(dataclass_path)
                dataclasses_dict[basename] = dataclass
                state_df = dataclass.make_state_df()
                state_dfs_dict[basename] = state_df
                s2p_outs[basename] = dataclass.s2p_out
        return dataclasses_dict, state_dfs_dict, s2p_outs
    
    def get_day_spikes(self, day_key, long_cells_dict):
        all_traces = []
        for cell in long_cells_dict.keys():
            trace = long_cells_dict[cell][day_key]
            all_traces.append(trace)
        pop_spikes = restrict_traces(all_traces)
        spks_z    = zscore_robust(pop_spikes, axis=1)
        return spks_z
    
    def get_per_cell_raster(self, long_cells_dict):
        for cell in list(long_cells_dict.keys()):
            rec_days = list(long_cells_dict[cell].keys())
            c0_traces = [trace for trace in long_cells_dict[cell].values()]
            c0_trace = restrict_traces(c0_traces)
            fig, axs = plt.subplots(len(rec_days), 1, figsize=(12, 4), sharex=True)
            plt.subplots_adjust(hspace=0.65)
            plt.suptitle(f'Cell {cell}: Longitudinal Traces')
            plt.xlabel('Frame #')
            for trace_idx, trace in enumerate(c0_trace):
                day_trace_2d = c0_trace[trace_idx].reshape(1, -1)
                axs[trace_idx].imshow(day_trace_2d, cmap='binary', aspect='auto') # For a single row
                axs[trace_idx].set_title(rec_days[trace_idx])
                axs[trace_idx].set_yticks([])
        plt.show()