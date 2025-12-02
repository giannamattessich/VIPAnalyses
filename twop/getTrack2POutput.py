import numpy as np, matplotlib.pyplot as plt
import os, traceback

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
        match_matrix = self.match_mat[np.sum(self.match_mat != None, axis=1) >= num_days_required]
        curr_cell = 0
        tracked_cells_dict = {}
        for row_idx in range(len(match_matrix)):
            tracked_cell = match_matrix[row_idx, :]
            valid_days = np.where(tracked_cell != None)[0]
            for day_idx in valid_days:
                cell_on_day = tracked_cell[day_idx]
                spike_train_day = s2p_outs[day_idx].spks[cell_on_day]
                p_day = self.days[day_idx]
                if f'C{curr_cell}' not in tracked_cells_dict.keys():
                    tracked_cells_dict[f'C{curr_cell}'] = {p_day:spike_train_day}
                else:
                    tracked_cells_dict[f'C{curr_cell}'][p_day] = spike_train_day
            curr_cell += 1
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