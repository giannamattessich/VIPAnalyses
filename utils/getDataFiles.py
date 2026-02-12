import os, traceback, shutil
import numpy as np
import pickle, joblib
from intan.readIntan import get_intan_data


def get_all_intan_data(intan_basepath, twop_chan=2, pd_chan=5, camera_chan=3, treadmill_chan=6,
                        plot_intan=False, intan_plot_start_s= None, intan_plot_end_s= None):
## PROVIDE ANALOG CHANNELS AS NUM ADC CHANNEL (0-8), NOT ANALOG/AUX TOTAL

    """
    Extract and load data from whole basepath: including 2P and intan data
    
    Args:

        intan_basepath (str, Path-like): path of directory where intan data is located 

        twop_chan (int; default 2): recording channel for scope

        pd_chan (int; default 5): photodiode channel number

        camera_chan (int; default 3): camera channel number

        treadmill_chan (int; default 6): treadmill channel number 

        plot_intan (bool; default False): 

        intan_plot_start_s (int; default None): 

        intan_plot_end_s (int; default None)

    Returns:

        fs_intan (default is 20e3): intan sampling rate

        phodiode_raw (np.array): raw photodiode signal, None if no channel provided

        twop_raw (np.array): raw 2P scope signal, None if no channel provided

        camera_raw (np.array): raw camera TTL signal, None is no channel provided

        treadmill_raw (np.array): raw treadmill signal, None if no channel

    """
    data, fs_intan, convertUnitsVolt, header = get_intan_data(intan_basepath, plot_intan=plot_intan,
                                                              intan_plot_start_s= intan_plot_start_s,
                                                                intan_plot_end_s= intan_plot_end_s)
    # if amp channels were recorded, add number amp channels to analog channels, as the 
    # amplifier_analogin_auxiliary_int16.dat file is concatenated with ALL channels
    # if not keep the same analog channel numbers
    num_amp_channels = header['num_amplifier_channels']
    # if num_amp_channels > 0:
    #     if twop_chan is not None:
    #         twop_chan += num_amp_channels
    #     if pd_chan is not None:
    #         pd_chan += num_amp_channels
    #     if camera_chan is not None:
    #         camera_chan += num_amp_channels
    #     if treadmill_chan is not None:
    #         treadmill_chan += num_amp_channels
    photodiode_raw, twop_raw, camera_raw, treadmill_raw = None, None, None, None
    if pd_chan is not None:
        try:
            photodiode_raw = data[pd_chan] * convertUnitsVolt
        except:
            print(f'Could not find photodiode data, or channel was not provided.')
    if twop_chan is not None:
        try:
            twop_raw = data[twop_chan] * convertUnitsVolt
        except:
            traceback.print_exc()
            print(f'Could not find 2P trigger data, or channel was not provided.')
    if camera_chan is not None:
        try:
            camera_raw = data[camera_chan] * convertUnitsVolt
        except:
            print(f'Could not get camera data, or channel was not provided.')
    if treadmill_chan is not None:
        try:
            treadmill_raw = data[treadmill_chan] * convertUnitsVolt
        except:
            print(f'Could not find treadmill data, or channel was not provided.')
    return fs_intan, photodiode_raw, twop_raw, camera_raw, treadmill_raw

def get_facemap_data(data_basepath):
    """
    HELPER FUNCTION TO INPUT DATA BASEPATH OF WHERE FACEMAP OUTPUT IS STORED,
    OUTPUT IS DICT WITH FACEMAP DATA-> if numpy file is saved it will load data 
    from numpy file. if not found, try to load .mat 

    Args:
        data_basepath (str): Path to directory where facemap data is stored

    Returns:
        facemap_data (dict): Loaded facemap data 
    """
    file_type = None
    for file in os.listdir(data_basepath):
        # if numpy file found, break loop
        if file.endswith('_proc.npy'):
            facemap_file_path = os.path.join(data_basepath, file)
            file_type = 'npy'
            break
        # only use mat file if numpy file doesnt exist
        elif file.endswith('_proc.mat'):
            facemap_file_path = os.path.join(data_basepath, file)
            file_type = 'mat'
    if file_type is None:
        print(f'WARNING!! The facemap file was not found in the path {data_basepath}')
    if file_type == 'npy':
        facemap_data = np.load(facemap_file_path, allow_pickle=True).item()
    elif file_type == 'mat':
        import scipy.io as sio
        facemap_data = sio.loadmat(facemap_file_path)
    print(f'Found facemap file path {facemap_file_path}')
    return facemap_data

'''
Generic function to load .pkl file into memory

Args:
    file_path: .pkl file to read

Returns:
    data(np.ndarray): data from file
'''
def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# create an array of len (ncells_TOTAL, nframes_TOTAL) by stacking values of all recording and filling with nan for recordings that 
# have less frames than the max num of frames in recordings
def pad_and_stack_vals(values_dict, s2p_outs_dict):
    max_frames_group = max({recording:s2p_out.nframes for recording,s2p_out in s2p_outs_dict.items()}.values())
    padded_arrs = {recording:np.pad(value_arr, pad_width=((0, 0), (0, max_frames_group - value_arr.shape[1])), constant_values=np.nan) 
                            for recording, value_arr in values_dict.items()}
    flat_valuelist = list(padded_arrs.values())
    stacked_arr = np.vstack(flat_valuelist)
    return stacked_arr

def rename_dat(rec_path):
    rec = os.path.basename(rec_path)
    new_dat_path = os.path.join(rec_path, f'{rec}.dat')
    dat_file_to_rename = None
    if not os.path.exists(new_dat_path):
        for file in os.listdir(rec_path):
            if os.path.isfile(os.path.join(rec_path, file)):
                if file == 'amplifier_analogin_auxiliary_int16.dat':
                    dat_file_to_rename = os.path.join(rec_path, file)
                elif file == 'analogin.dat':
                    if os.path.getsize(os.path.join(rec_path, 'analogin.dat')) != 0:
                        dat_file_to_rename = os.path.join(rec_path, file)
    else:
        return
    if dat_file_to_rename is not None:
        print(f'Renaming {dat_file_to_rename} for {new_dat_path}')   
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            print("Continuing...")
        elif confirm == 'n':
            print("Terminating.")
            exit()
        os.rename(dat_file_to_rename, new_dat_path)
    else:
        print(f'Found no intan dat to rename. Skipping rec {rec_path}')

def rename_all_dats(rec_dirnames):
#def rename_dat_to_base(rec_dirnames):
    for day_path in rec_dirnames:
        rename_dat(day_path)

def copy_neuroscope_xml(source_xml, rec_dirnames):
    if not source_xml.endswith('.xml'):
        # try to see if directory was provided by accident
        source_xml = os.path.join(source_xml, f'{os.path.basename(source_xml)}.xml')
    if not os.path.exists(source_xml):
        raise ValueError(f'Provided source xml does not exist.')
    for day_path in rec_dirnames:
        xml_path = os.path.join(day_path, f'{os.path.basename(day_path)}.xml')
        if not os.path.exists(xml_path):
            shutil.copy(source_xml, xml_path)
            print(f'Copied {source_xml} to {day_path}!')

# def delete_xmls(source_day, datapaths):
#     del_files = [os.path.join(path, f'{os.path.basename(path)}.xml') for path in datapaths if not path.endswith('p9')]
#     for file in del_files:
#         os.remove(file)

def get2p_foldername_field(data_basepath):
    """ 
    Get 2P folder name, and append suite2p path to get the datapath of suite2p files.
    ***NOTE: this only works if you have data saved under folder starting with the word 'field'

    Args:
        data_basepath (str): parent base folder of tif files
    Returns:
        suite2p folder path (str): path to s2p folder
    """
    files = os.listdir(data_basepath)
    foldername = ''
    for folder in files:
        if folder.lower().startswith('field'):
            foldername = os.path.join(data_basepath, folder)
    if foldername == '':
        return None
    return os.path.join(data_basepath, foldername, 'suite2p')

def get_recording_paths_sorted(basepath):
    rec_paths = [
        os.path.join(basepath, rec_path) for rec_path in os.listdir(basepath) if not rec_path.endswith('stim') and rec_path.startswith('p')]
    days = []
    for path in rec_paths:
        day = os.path.basename(path)
        days.append(int(day[1:]))
    days = sorted(days)
    days = [os.path.join(basepath, f'p{str(day)}') for day in days]    
    return days