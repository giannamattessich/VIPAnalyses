import os,traceback, numpy as np
from intan.importrhdutilities import read_header
from utils.filtering import *
import pandas as pd
from intan.readIntan import get_intan_files 
from scipy.signal import coherence
from scipy.signal import spectrogram

#import data, fs_intan, convertUnitsVolt, header = get_intan_data(intan_basepath)
'''Python translated matlab code inspired by buz_code function LFPfromDat: https://github.com/buzsakilab/buzcode'''

'''Check if cuda available and cupy installed. '''
try:
    import cupy as cp
    from cupyx.scipy.signal import firwin as firwin_gpu, filtfilt as filtfilt_gpu
    os.environ["CUPY_NVRTC_EXTRA_FLAGS"] = "--std=c++17"
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    print("CuPy not available — using CPU fallback.")

def extract_lfp(
    basepath: str,
    channels: list,                        # <- indices (0-based) of channels to extract as LFP
    out_fs: int = 1250,
    lopass: float = 450.0,
    overwrite = True,
    output_lfppath: str = None,
    csv_savepath: str = None,
    return_data: bool = True,
    chunk_size: int = 2_000_000,           # input samples per chunk (at in_fs)
    raw_memmap: np.ndarray | np.memmap = None,  # <- OPTIONAL: preallocated raw matrix (samples, total_channels)
    dtype_in = np.int16
):
    """
    Stream-extract LFP from Intan analog/amplifier data.
    - If `raw_memmap` is provided (shape: [total_samples, total_channels]), we stream from it.
      Otherwise we read the interleaved Intan file directly.
    - Only processes the channels listed in `channels`.
    - Writes an .lfp file with int16, decimated to `out_fs`.
    - Optionally returns float32 downsampled data in RAM.

    Parameters
    ----------
    basepath : str
        Recording folder.
    channels : list[int]
        0-based channel indices to extract as LFP (w.r.t. Intan channel ordering you used to build raw_memmap).
    out_fs : int
        Target sampling rate after decimation.
    lopass : float
        Low-pass cutoff before decimation (Hz). Should be < 0.45 * out_fs.
    output_lfppath : str or None
        Destination .lfp file path. Defaults to `<basepath>/<basename>.lfp`.
    csv_savepath : str or None
        Optional CSV path to also dump float data (big).
    return_data : bool
        If True, return float32 array (channels, samples_out).
    chunk_size : int
        Number of **input** samples to process per chunk.
    raw_memmap : np.ndarray | np.memmap or None
        If provided, must be shape (total_samples, total_channels) and dtype matches `dtype_in`.
    dtype_in : numpy dtype
        Input dtype (usually int16 for Intan).

    Returns
    -------
    final_data : np.ndarray (float32, channels x out_frames) if return_data else None
    """
    # --- helpers you already have ---
    amp_analog_aux_in, intan_header, time_file = get_intan_files(basepath)
    basename = os.path.basename(basepath)

    # Output file path handling
    if output_lfppath is None:
        lfp_out = os.path.join(basepath, f"{basename}.lfp")
    else:
        lfp_out = output_lfppath if output_lfppath.endswith(".lfp") else output_lfppath + ".lfp"

    if csv_savepath is not None and not csv_savepath.endswith(".csv"):
        csv_savepath += ".csv"

    # Intan header info
    in_fs = int(intan_header["sample_rate"])
    total_channels = (intan_header["num_amplifier_channels"]
                      + intan_header["num_board_adc_channels"]
                      + intan_header["num_aux_input_channels"])

    # Validate channels
    channels = np.asarray(channels, dtype=int)
    if channels.ndim != 1 or channels.size == 0:
        raise ValueError("`channels` must be a non-empty 1D list/array of channel indices.")
    if (channels < 0).any() or (channels >= total_channels).any():
        raise ValueError(f"`channels` indices must be in [0, {total_channels-1}]")

    # Decimation ratio
    if in_fs % out_fs != 0:
        raise ValueError(f"in_fs ({in_fs}) must be an integer multiple of out_fs ({out_fs}).")
    sample_ratio = in_fs // out_fs

    # Low-pass sanity check
    nyq_out = 0.5 * out_fs
    if lopass >= 0.9 * nyq_out:
        # keep a bit of guard to avoid aliasing / filter rolloff issues
        lopass = 0.9 * nyq_out
        print(f"[Note] Adjusted lopass to {lopass:.1f} Hz to stay below out_fs Nyquist.")

    # Figure total frames (samples) at input
    if raw_memmap is not None:
        # Expecting shape (total_samples, total_channels)
        if raw_memmap.ndim != 2 or raw_memmap.shape[1] != total_channels:
            raise ValueError("`raw_memmap` must have shape (total_samples, total_channels) matching Intan header.")
        total_frames = raw_memmap.shape[0]
    else:
        # reading from .dat
        bytes_per_sample = np.dtype(dtype_in).itemsize
        fsize = os.path.getsize(amp_analog_aux_in)
        total_frames = fsize // (bytes_per_sample * total_channels)

    out_frames = (total_frames + sample_ratio - 1) // sample_ratio   # number of timepoints after decimation
    n_sel = len(channels)                                            # number of channels you are extracting

    # Create output memmap (.lfp file)
    if overwrite:
        lfp_out_map = np.memmap(lfp_out, dtype=np.int16, mode="w+", shape=(out_frames, n_sel))
    else:
        lfp_out_map = np.memmap(lfp_out, dtype=np.int16, mode="r+", shape=(out_frames, n_sel))
    # Output time length
    #out_frames = (total_frames + sample_ratio - 1) // sample_ratio  # ceil

    # Make chunk_size divisible by sample_ratio for clean decimation edges
    if chunk_size % sample_ratio != 0:
        chunk_size += sample_ratio - (chunk_size % sample_ratio)

    # Prepare output memmap: (out_frames, n_selected_channels) as int16
    # n_sel = channels.size
    # lfp_out_map = np.memmap(lfp_out, dtype=np.int16, mode="w+", shape=(out_frames, n_sel))
    # Streaming
    out_pointer = 0
    downsampled_float_list = [] if return_data else None

    def _process_block(block_samps_nc):
        """
        block_samps_nc: np.ndarray of shape (samples, total_channels), dtype int16
        returns: None (writes to lfp_out_map and maybe collects float32)
        """
        nonlocal out_pointer, downsampled_float_list

        if block_samps_nc.size == 0:
            return

        # Select requested channels and switch to (channels, samples)
        sel = block_samps_nc[:, channels]                   # (samples, n_sel)
        sel_cs = sel.T.astype(np.float32, copy=False)       # (n_sel, samples)

        # Filter and decimate
        filtered_cs = sinc_lowpass_filter(sel_cs, cutoff=lopass, fs=in_fs)  # (n_sel, samples)
        down_cs = filtered_cs[:, ::sample_ratio]                               # (n_sel, out_samps_chunk)

        # Prepare int16 for disk
        if GPU_AVAILABLE:
            down_int = cp.asnumpy(cp.around(cp.asarray(down_cs)).astype(np.int16))
            down_f32 = cp.asnumpy(cp.asarray(down_cs, dtype=cp.float32)) if return_data else None
        else:
            down_int = np.around(down_cs).astype(np.int16)
            down_f32 = down_cs.astype(np.float32, copy=False) if return_data else None

        # Write (out_frames_chunk, n_sel) — our map is (out_frames, n_sel)
        n_out = down_int.shape[1]
        lfp_out_map[out_pointer:out_pointer + n_out, :] = down_int.T
        out_pointer += n_out

        if return_data:
            downsampled_float_list.append(down_f32)

    # --- Iterate over source ---
    if raw_memmap is not None:
        # Stream from existing matrix (samples, channels)
        n_batches = (total_frames + chunk_size - 1) // chunk_size
        print(f"Processing from raw_memmap in {n_batches} batches...")
        start = 0
        while start < total_frames:
            stop = min(start + chunk_size, total_frames)
            _process_block(raw_memmap[start:stop, :])
            start = stop
    else:
        # Read interleaved Intan .dat directly
        n_batches = (total_frames + chunk_size - 1) // chunk_size
        print(f"Processing from disk in {n_batches} batches...")
        bytes_per_frame = np.dtype(dtype_in).itemsize * total_channels
        with open(amp_analog_aux_in, "rb") as fid_in:
            frames_read = 0
            while frames_read < total_frames:
                # Read chunk of complete frames (samples)
                to_read = min(chunk_size, total_frames - frames_read)
                raw = np.fromfile(fid_in, dtype=dtype_in, count=to_read * total_channels)
                if raw.size == 0:
                    break
                # Guard against partial frame at EOF
                if raw.size % total_channels != 0:
                    drop = raw.size % total_channels
                    print(f"Warning: Dropping {drop} stray values at EOF.")
                    raw = raw[:-drop]
                block = raw.reshape(-1, total_channels)      # (samples, total_channels)
                _process_block(block)
                frames_read += block.shape[0]

    # Flush output
    lfp_out_map.flush()
    print(f"LFP extraction complete. Saved to: {lfp_out}")

    # Optional return and CSV dump
    if return_data:
        final_data = np.hstack(downsampled_float_list)  # (n_sel, out_frames)
        if csv_savepath is not None:
            # CSV rows = timepoints => transpose to (out_frames, n_sel)
            np.savetxt(csv_savepath, final_data.T, delimiter=",", fmt="%.4f")
        return final_data
    else:
        if csv_savepath is not None:
            print("[Note] csv_savepath is ignored because return_data=False (no float data in RAM).")
        return None
    
def load_lfp_channel(lfp_path, n_channels, chan_idx):
    raw = np.memmap(lfp_path, dtype=np.int16, mode="r")
    n_samples = raw.size // n_channels
    data = np.reshape(raw[:n_samples * n_channels], (n_samples, n_channels))
    lfp = data[:, chan_idx].astype(np.float32)
    return lfp

def load_lfp(lfp_source, num_channels=2):
    if isinstance(lfp_source, str):
        if os.path.exists(lfp_source):
            try:
                lfp = np.memmap(lfp_source,
                        dtype=np.int16, mode='r')
                #if num_channels != 1:
                lfp = lfp.reshape(num_channels, len(lfp) // num_channels)
                # else:
                #     lfp = lfp.reshape(-1, 1)
                #else:
                return np.array(lfp)
            except:
                print(f'Failed to load .lfp file')
                traceback.print_exc()
        else:
            raise ValueError(f'Provided lfp path does not exist.')
    else:
        try:
            lfp = np.array(lfp_source) 
            return lfp  
        except:
            print('Could not cast LFP source to numpy array.')
            traceback.print_exc()

def extract_lfp_preallocated(basepath, out_fs=1250, lopass=450, output_lfppath=None, csv_savepath=None, return_data=True):
        """
        FASTER, Same as function above, but speed up by preallocating numpy memory map before write
        """
        amp_analog_aux_in, intan_header, time_file = get_intan_files(basepath)
        basename = os.path.basename(basepath)

        if output_lfppath is None:
            lfp_out = os.path.join(basepath, f"{basename}.lfp")
        else:
            if not output_lfppath.endswith('.lfp'):
                output_lfppath += '.lfp'
            lfp_out = output_lfppath
        if csv_savepath is not None:
            if not csv_savepath.endswith('.csv'):
                csv_savepath += '.csv'
        #get sampling rate and num amp channels from header
        in_fs = intan_header['sample_rate']
        num_channels = intan_header['num_amplifier_channels'] + intan_header['num_board_adc_channels'] + intan_header['num_aux_input_channels']
    
        sample_ratio = int(in_fs / out_fs)
        # samples are int16 
        bytes_per_sample = np.dtype(np.int16).itemsize
        fsize = os.path.getsize(amp_analog_aux_in)
        total_frames = fsize // (bytes_per_sample * num_channels)  
        # Output length (time dimension after decimation)
        out_frames = (total_frames + sample_ratio - 1) // sample_ratio
        #check if chunk size is divisible by downsampling factor 
        if chunk_size % sample_ratio != 0:
            # if not divisible, add difference of ratio by the remainder of chunk size and sample ratio
            chunk_size += sample_ratio - (chunk_size % sample_ratio)
        # get process batches -> divide file size by num_chunk_bytes * num channels * int16 per sample = bytes per sample * bytes per chunk
        n_batches = fsize // (chunk_size * num_channels * bytes_per_sample)
        print(f'Processing in {n_batches} batches.')

        lfp_out_map = np.memmap(lfp_out, dtype=np.int16, mode="w+", shape=(out_frames, num_channels))

        batch_num = 0
        out_pointer = 0

        with open(amp_analog_aux_in, "rb") as fid_in:
            downsampled_float_list = []

            while True:
                raw = np.fromfile(fid_in, dtype=np.int16, count=chunk_size * num_channels)
                if raw.size == 0:
                    break  # End of file

                if raw.size % num_channels != 0:
                    print(f"Warning: Incomplete sample at end of file. Dropping {raw.size % num_channels} values.")
                    raw = raw[:raw.size - (raw.size % num_channels)]

                # reshape into (channels, samples)
                data = raw.reshape(-1, num_channels).T

                # filter and downsample
                filtered = sinc_lowpass_filter(data, cutoff=lopass, fs=in_fs)
                downsampled_batch = filtered[:, ::sample_ratio]

                # float copy if needed later
                downsampled_float = downsampled_batch.astype(np.float32)

                if GPU_AVAILABLE:
                    downsampled_int = cp.asnumpy(cp.around(downsampled_batch).astype(np.int16))
                    downsampled_float = cp.asnumpy(downsampled_float)
                else:
                    downsampled_int = np.around(downsampled_batch).astype(np.int16)

                # write into memmap
                n_out = downsampled_int.shape[1]
                lfp_out_map[out_pointer:out_pointer + n_out, :] = downsampled_int.T
                out_pos += n_out

                downsampled_float_list.append(downsampled_float)
                batch_num += 1

        # flush memmap to disk
        lfp_out_map.flush()

        # also return float data in RAM
        final_data = np.hstack(downsampled_float_list)

        print(f"LFP extraction complete. Saved to: {lfp_out}")

        if csv_savepath is not None:
            np.savetxt(csv_savepath, final_data.T, delimiter=",", fmt="%.4f")

        return final_data

def extract_lfp_allrecs(rec_paths_list: list,
    channels: list,                        # <- indices (0-based) of channels to extract as LFP
    out_fs= 1250,
    lopass= 450.0,
    overwrite = True,
    output_lfppath = None,
    csv_savepath = None,
    return_data = True,
    chunk_size= 2_000_000,           # input samples per chunk (at in_fs)
    raw_memmap= None,  # <- OPTIONAL: preallocated raw matrix (samples, total_channels)
    dtype_in = np.int16):
    for rec_path in rec_paths_list:
        extract_lfp(rec_path, channels, out_fs=out_fs, lopass=lopass,
                    overwrite=overwrite, output_lfppath=output_lfppath, csv_savepath=csv_savepath,
                    return_data=return_data, chunk_size=chunk_size, raw_memmap=raw_memmap, dtype_in=dtype_in)