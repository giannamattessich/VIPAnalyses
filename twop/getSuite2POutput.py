import os, traceback, numpy as np, pandas as pd, matplotlib.pyplot as plt
import suite2p
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter, median_filter
from utils.stats import movquant
import matplotlib.ticker as ticker
from utils.getDataFiles import get2p_foldername_field

class Suite2POutput:
    """
    Class to initialize Suite2P output data for a given plane and perform further analyses after 2p ran

    Parameters:
        suite2p_path (str; Path-like): output folder after running suite2p

        plane (str, optional): plane in the format 'plane{plane #}', default is plane0

        scope_fs (float; default:1.366): capture rate of 2P scope
    """
    def __init__(self, suite2p_path, plane='plane0', scope_fs=1.366): 
        self.suite2p_path = suite2p_path
        self.tif_folder = os.path.dirname(suite2p_path)
        # if basepath accidentally provided, try checking if suite2p folder exists within
        # provided suite2p_path
        if not self.suite2p_path.endswith('suite2p'):
            poss_paths = [os.path.join(self.suite2p_path, 'suite2p'),
                          get2p_foldername_field(self.suite2p_path)]
            for poss_path in poss_paths:
                if poss_path is not None:
                    if os.path.exists(poss_path):
                        self.suite2p_path = poss_path
        # check if there is data in folder
        plane_path = os.path.normpath(os.path.join(self.suite2p_path, plane))
        print(f'Found Suite2P Path {self.suite2p_path}!')

        if not os.path.exists(plane_path):
            raise ValueError(f'{plane_path} is not a valid suite2p path.')
        try:
        # check if can find scope fps from experiment meta data file (must be within same 2p folder in 'field_xx'), if not default to provided arg to function
            experiment_metadata_file = os.path.join(os.path.dirname(self.suite2p_path), 'Experiment.xml')
            if not os.path.exists(experiment_metadata_file):
                print(f"WARNING!!!! Experiment.xml file ({experiment_metadata_file}) not found and defaulting to scope_fs=30. This will cause incorrect downstream analyses.")
                self.scope_fs = scope_fs
            else:
                from utils.getFPS import get_fps_from_xml
                scope_fps = get_fps_from_xml(experiment_metadata_file)
                print(f'Found scope fps from Experiment.xml file: {scope_fps}')
                self.scope_fs = scope_fs
        except:
            traceback.print_exc()
        suite2p_path = plane_path
        try:
            # load s2p output files into class 
            # flourescence, neuropil flourescence, iscell, options, spikes, and stats files, etc
            self.F = np.load(os.path.join(suite2p_path,'F.npy'), allow_pickle= True)
            self.Fneu = np.load(os.path.join(suite2p_path,'Fneu.npy'), allow_pickle= True)
            self.iscell = np.load(os.path.join(suite2p_path, 'iscell.npy'), allow_pickle= True)[:, 0].astype(bool)
            self.ops = np.load(os.path.join(suite2p_path, 'ops.npy'), allow_pickle= True)
            self.ops_dict = self.ops[()]
            self.spks = np.load(os.path.join(suite2p_path, 'spks.npy'), allow_pickle= True)
            self.stat = np.load(os.path.join(suite2p_path, 'stat.npy'), allow_pickle= True)
            self.rois, self.nframes = self.F.shape[0], self.F.shape[1] - 1
            self.cell_indices = np.where(self.iscell == True)[0]
            self.num_cells = len(self.cell_indices)
        except Exception:
            traceback.print_exc()

    def calc_deltaF(self, method='baseline', q=0.1, window_len=600, detrend=True, F_neuropil=False, 
                     save_csv=False, output_filepath=None, output_df=False):
        """
        Function to calculate ΔF/F of cells' calcium traces

        Args:
            method (str; default='baseline', options= 'baseline', 'median', 'mean'): method used to calculate DFF

            q (float between 0 and 1; default: 0.1): quantile to use for moving baseline

            window_len (int; default=600): window length in seconds to filter/convolve signal

            detrend (bool; default=True): whether to detrend calcium signal with a median filter

            **F_neuropil (bool; default=False): set to True if calculating ΔF of neuropils

            save_csv (bool; default=False): whether to save ΔF/F output as a csv

            output_filepath (str or Path-like; default=None): output file name to save dff to csv

            output_df (bool; default=False): whether to output DeltaF calculation result as a pandas df (pd.DataFrame)

        Returns:
            deltaF (np.ndarray, pd.DataFrame): array of shape (num_cells, num_frames+1) with delta f of calcium signal OR
            pd.DataFrame of length num_frames if output_df == True
            
            timeEst (1D np.ndarray): estimated times of scope frames from scope_fps

        """
        # get raw fluourescence from cells 
        rawF = self.F[self.cell_indices, :].astype(np.float64)
        if F_neuropil:
            neuropil_F = self.Fneu[self.cell_indices, :].astype(np.float64)
            neuropil_coef = float(self.ops_dict.get("neucoeff", 0.7))
            F_neuropil_corrected = self.F[self.cell_indices, :] - neuropil_coef * neuropil_F
            rawF = F_neuropil_corrected
        if method == 'median':
            medianF = np.median(rawF, axis=1, keepdims=True)
            deltaF = (rawF - medianF) / medianF
        elif method == 'mean':
            meanF = np.mean(rawF, axis=1, keepdims=True)
            deltaF = (rawF - meanF) / meanF
        elif method == "baseline":
            # using sliding baseline (10 minutes window)
            winLen = min(round(window_len * self.scope_fs), rawF.shape[1])  # window size in frames
            baseline = np.zeros_like(rawF)
            for i in range(rawF.shape[0]):
                # Use a 10th percentile moving baseline (approximation using quantile filter)
                baseline[i, :] = movquant(rawF[i, :], q=q, window=winLen)
            deltaF = (rawF - baseline) / baseline
        else:
            raise ValueError(f"{method} method of computing df/F not recognized")
        # remove cells with all NaNs (ex. edge artifacts)
        valid_cells = ~np.isnan(deltaF).all(axis=1)
        deltaF = deltaF[valid_cells, :]
        # remove low-frequency trends using median filter
        if detrend:
            trend = median_filter(deltaF, size=(1, window_len))  # apply median filter across time axis
            deltaF = deltaF - trend
        # add timing info from num frames and scope fps
        num_frames = rawF.shape[1]
        timeEst = np.arange(num_frames) / self.scope_fs
        ### save to csv if true and output filepath provided
        if save_csv and output_filepath is not None:
            if not output_filepath.endswith('.csv'):
                output_filepath += '.csv'
            np.savetxt(output_filepath, deltaF, delimiter=',')
        ## output dataframe if output_df = True 
        if output_df:
            if (deltaF.shape[1]) != len(timeEst):
                raise ValueError('Timestamps are not the same length as deltaF frames!!')
                # Create DataFrame with rows: first is time, then deltaF rows
            all_data = np.vstack([timeEst[np.newaxis, :], deltaF])  # shape: (n_cells+1, n_timepoints)
            # Create DataFrame
            dff_df = pd.DataFrame(all_data)
            return dff_df, timeEst
        return deltaF, timeEst
    
    def get_cell_spikes(self, save_to_csv= False, output_filepath= None, output_df= False, df_filepath= None):
        """
        Args:
            save_to_csv (bool; default=False): whether to save spikes of cells as a CSV file

            output_filepath (str or Path-like; default=None): output file path to save spikes to  

            output_df (bool; default=False): whether to output cell spikes as a pandas df (pd.DataFrame)
            
            df_filepath (str or Path-like; default=None): output file name to save spikes dataframe to csv if output_df == True

        Returns:
            spikes (np.ndarray OR pd.DataFrame): spikes of cells in shape (num_cells, num_frames) OR pandas DF is output_df = True
        """
        cell_spikes = self.spks[self.cell_indices, :].astype(np.float64)
        if save_to_csv and output_filepath is not None:
            if not output_filepath.endswith('.csv'):
                output_filepath += '.csv'
            np.savetxt(output_filepath, cell_spikes, delimiter=',')
        ## output dataframe if save_df = True and output_df_path  
        if output_df:
            spikes_df = pd.DataFrame(cell_spikes)
            if df_filepath is not None:
                spikes_df.to_csv(df_filepath)
            return spikes_df
        return cell_spikes
    
    def get_SNR(self):
        """
        Function to calculate the per-cell signal-to-noise ratio 

        Args:
            None

        Returns:
            SNR (array-like): array of length = cells, with SNR value per cell
        """
        neuropil_coef = float(self.ops_dict.get("neucoeff", 0.7))
        iscell = self.iscell
        neuropil_f = self.Fneu[iscell, :]
        cells_f = self.F[iscell, :]
        # #Fs = F - r*Fneu              
        F_neuropil_corrected = cells_f - neuropil_coef * neuropil_f            # neuropil-corrected
        neuropil_dff = self.calc_deltaF(F_neuropil=True)[0]
        # Noise estimate: scaled MAD of baseline points (low ΔF/F)
        # pick frames below the 20th percentile per cell as "baseline"
        # initialize to empty arrs of length frames
        SNR = np.zeros(F_neuropil_corrected.shape[0])
        noise_sigma = np.zeros(F_neuropil_corrected.shape[0])
        #resp_amp = np.zeros(Fs.shape[0])
        signal_peaks = np.zeros(F_neuropil_corrected.shape[0])

        for cell_idx in range(F_neuropil_corrected.shape[0]):
            dff_threshold = np.percentile(neuropil_dff[cell_idx], 20)
            dff_above_thresh = neuropil_dff[cell_idx][neuropil_dff[cell_idx] <= dff_threshold]
            mad = np.median(np.abs(dff_above_thresh - np.median(dff_above_thresh)))
            sigma = 1.4826 * mad + 1e-12
            noise_sigma[cell_idx] = sigma
            # signal amplitude: robust peak (95th percentile of ΔF/F)
            signal_peaks[cell_idx] = np.percentile(neuropil_dff[cell_idx], 95)
            SNR[cell_idx] = signal_peaks[cell_idx] / sigma
        return SNR

    def get_spike_df(self, scope_times = None, remove_zeros=False):
        spikes = self.get_cell_spikes()
        num_frames = self.nframes
        cells, times, spike_vals = [] , [] , []
        spike_df = pd.DataFrame({}, columns=['Cell','Time','Value'])
        for cell in range(spikes.shape[0]):
            cell_spikes = spikes[cell]
            if spikes.shape[1] != self.nframes:
                cell_spikes = cell_spikes[:-1]
            cells.extend([cell] * num_frames)
            spike_vals.extend(cell_spikes)
            if scope_times is not None:
                times.extend(scope_times)
            else:
                times.extend(np.arange(0, len(cell_spikes)) * self.scope_fs)
        spike_df = pd.DataFrame({'Cell': cells, 'Time': times, 'Value': spike_vals})
        # if destination_path != None:
        #     spike_df.to_csv(destination_path)
        return spike_df            


            # for spike in cell_spikes:
            #     cells.append([cell] * len())

    # def get_spike_df(self, scope_times = None, remove_zeros=False):
    #         spikes = self.spks[self.cell_indices, :]
    #         cells, times, spike_vals = [], [], []
    #         spike_df = pd.DataFrame({}, columns=['Cell','Time','Value'])
    #         # iter through cell rows
    #         for row in range(len(spikes)):
    #             spikes_cell = spikes[row]
    #             cells.extend([row] * len(scope_times))
    #             times.extend(scope_times)
    #             spike_vals.extend(spikes_cell)
    #         #     if remove_zeros:
    #         #         nonzero_times = np.where(spikes[row] != 0)[0]
    #         #     else:
    #         #         nonzero_times = np.arange(0, len(spikes))
    #         #         #nonzero_times = np.asarray(spikes[row])
    #         #     # spike val
    #         #     nonzero_spks = spikes[row][nonzero_times]
    #         #     if scope_times is not None:
    #         #         nonzero_times = scope_times
    #         #     else:
    #         #         nonzero_times = nonzero_times * self.scope_fs

    #         #     cells.extend([f'C{row}'] * len(nonzero_times))
    #         #     times.extend(nonzero_times)
    #         #     spike_vals.extend(nonzero_spks)
    #         #     #new_rows = pd.DataFrame({'Cell': cell, 'Time': nonzero_times, 'Value': nonzero_spks})
    #         print(len(cells), len(times), len(spike_vals))
    #         spike_df = pd.DataFrame({'Cell': cells, 'Time': times, 'Value': spike_vals})
    #         # if destination_path != None:
    #         #     spike_df.to_csv(destination_path)
    #         return spike_df

    # def get_spike_df(self):
    #         spikes = self.spks[self.cell_indices, :]
    #         cells, times, spike_vals = [], [], []
    #         #spike_df = pd.DataFrame({}, columns=['Cell','Time','Value'])
    #         for row in range(len(spikes)):
    #             nonzero_times = np.where(spikes[row] != 0)[0]
    #             # spike val
    #             nonzero_spks = spikes[row][nonzero_times]
    #             nonzero_times = nonzero_times * self.scope_fs
    #             cells.extend([f'C{row}'] * len(nonzero_times))
    #             times.extend(nonzero_times)
    #             spike_vals.extend(nonzero_spks)
    #             #new_rows = pd.DataFrame({'Cell': cell, 'Time': nonzero_times, 'Value': nonzero_spks})
    #         spike_df = pd.DataFrame({'Cell': cells, 'Time': times, 'Value': spike_vals})
    #         # if destination_path != None:
    #         #     spike_df.to_csv(destination_path)
    #         return spike_df