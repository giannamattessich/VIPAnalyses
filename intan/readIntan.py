import os, numpy as np, traceback
import intan.importrhdutilities as rhd_utils
import matplotlib.pyplot as plt

# return name of intan analog recording file and intan header file
def get_intan_files(intan_basepath):
    """
    Given a data basepath, find .dat intan files containing the recording data. 

    If amp_analog_aux file (concatenated data file of all input signals) exists, return this.
    If not found, return analogin.dat file (nonconcatenated analog signals data).

    Args:
        intan_basepath (str, Path-like): data filepath of directory containing intan files 
                                         directory must contain:
                                            - either 'amplifier_analogin_auxiliary_int16.dat' OR 'analogin.dat'
                                            - 'info.rhd'
    Returns:
        amp_analog_aux_in (str): filepath name of file containing signals data
        intan_header (dict): dictionary containing header info from found info.rhd file
        time_file (str, None): filepath of time.dat file, None if not found
    """
    # === Locate Intan analog input file ===
    amp_analog_aux_in = os.path.join(intan_basepath, 'amplifier_analogin_auxiliary_int16.dat')
    basename = os.path.basename(intan_basepath)
    if not os.path.exists(amp_analog_aux_in):
        for poss_file in [f"{basename}.dat", "analogin.dat"]:
            path_candidate = os.path.join(intan_basepath, poss_file)
            if (os.path.exists(path_candidate)):
                if os.path.getsize(path_candidate) != 0:
                    amp_analog_aux_in = path_candidate
                    break
                print(f'WARNING!! Found candidate path file {path_candidate} in basepath, but it is empty. \
                      Please check file names and their sizes: the data file should be called\
                      amp_analogin_auxiliary_int16.dat file, (basepath).dat or analogin.dat')
        else:
            raise ValueError('No valid analog input file found in intan_basepath.')
    # attempt to find time.dat file. if not found it is ok as this does not contain signal amplitude info
    time_file = os.path.join(intan_basepath, 'time.dat')
    if not os.path.exists(time_file):
        time_file = None
        print('Warning: could not infer channel #s from missing time.dat file.')
    # Load Intan info header
    info_file = os.path.join(intan_basepath, 'info.rhd')
    if not os.path.exists(info_file):
        raise ValueError('Info file (info.rhd) not found in intan_basepath.')
    ## read intan header for info 
    with open(info_file, 'rb') as f:
        intan_header = rhd_utils.read_header(f)
    return amp_analog_aux_in, intan_header, time_file


def get_intan_data(intan_basepath, plot_intan=False, intan_plot_start_s=0, intan_plot_end_s=None, intan_fs = 20e3):
    """
    Read from intan header to get number timepoints and channel information, all signals data, volt conversion units for file type

    Args:
        intan_basepath (str, Path-like): data filepath of directory containing intan files 
                                         directory must contain:
                                            - either 'amplifier_analogin_auxiliary_int16.dat' OR 'analogin.dat'
                                            - 'info.rhd'
        plot_intan (bool; default False): whether to plot with all signals plotted
    Returns:
        data (np.ndarray / numpy memorymap): numpy mmap of shape (num_channels, num_samples)
        fs_analog (sample rate of intan; default=20e3):"
        convertUnitsVolt (float)
        intan_header (dict)
    """
    # === Locate Intan analog input file ===
    amp_analog_aux_in, intan_header, time_file = get_intan_files(intan_basepath)
    fs_analog = intan_header['sample_rate']
    # Determine data type and microvolt conversion
    if 'analogin.dat' in amp_analog_aux_in:
        dtype = 'uint16'
        convertUnitsVolt = 0.000050354
    else:
        dtype = 'int16'
        convertUnitsVolt = 0.00010071
    # calc num channels from header, if available use time.dat file instead
    num_channels = (len(intan_header['amplifier_channels']) +
                    len(intan_header['aux_input_channels']) +
                    len(intan_header['board_adc_channels']))
    analog_filesize = os.path.getsize(amp_analog_aux_in)
    num_samples_total = analog_filesize // np.dtype(dtype).itemsize
    # uncomment if you have ONLY a time.dat file. WARNING: analog data is not located here 
  #  if os.path.exists(time_file):
    #     # get file size, each sample is int32 (4 bytes)
    #     inferred_channels = num_channels
    #     #num_time_samples = os.path.getsize(time_file) // 4
    #     #num_channels = num_samples_total // num_time_samples
    #     if inferred_channels != num_channels:
    #         print('Warning! Number channels from time samples differ from header!')
    num_timepoints = num_samples_total // num_channels
    # Memory-map analog data file
    mmap = np.memmap(amp_analog_aux_in, dtype=dtype, mode='r')
    data = mmap.reshape((num_timepoints, num_channels)).T
    if plot_intan:
        if intan_plot_start_s is not None:
            if intan_plot_end_s is None:
                intan_plot_end_s = num_timepoints // intan_fs
            plot_raw_intan(data, start_s=intan_plot_start_s, end_s=intan_plot_end_s, intan_fs=fs_analog)
        else:
            traceback.print_exc()
            print(f'Attempted to create raw intan plot: intan time not provided')
    return data, fs_analog, convertUnitsVolt, intan_header

def plot_raw_intan(intan_data, start_s, end_s, intan_fs=20e3):
    start, end = int(start_s*intan_fs), int(end_s * intan_fs)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(intan_data.shape[0], 1, figsize=(8, 12))
    for chan in range(intan_data.shape[0]):
        axs[chan].plot(np.arange(start, end), intan_data[chan][start:end])
    plt.show()