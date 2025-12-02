import numpy as np
import tifffile
import os

'''
Method to convert a 2P .RAW file and save to provided output_dir 
Will output and save 2 subfolders in output_dir -> TIFS for each channel

'''
def save_raw_frames_to_tifs(file_path, width, height, output_dir, chunk_size=100):
    """
    Save each frame from a 2-channel .raw file as individual .tif files for both channels.

    Parameters:
        file_path: Path to .raw file
        width: Image width
        height: Image height
        output_dir: Directory to save .tif files
        chunk_size: Number of frames per read chunk

    Returns:
        None -> saves tifs and splits into 2 folders: one for and one for other 
    """
    os.makedirs(output_dir, exist_ok=True)
    channel1_folder, channel2_folder = os.path.join(output_dir, 'channel1'), os.path.join(output_dir, 'channel2')
    print(f'Created directories {channel1_folder} and {channel2_folder}')
    os.makedirs(channel1_folder, exist_ok=True)
    os.makedirs(channel2_folder, exist_ok=True)
    frame_size = width * height * 2  # 2 channels
    bytes_per_frame = frame_size * 2  # uint16
    frame_counter = 0

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size * bytes_per_frame)
            if not data:
                break
            chunk = np.frombuffer(data, dtype=np.uint16)
            num_pixels = chunk.size
            num_frames = num_pixels // frame_size
            if num_frames == 0:
                break
            chunk = chunk[:num_frames * frame_size]
            chunk = chunk.reshape((num_frames, 2, height, width))

            for i in range(num_frames):
                chan1_chunk = chunk[i, 0, :, :]
                chan2_chunk = chunk[i, 1, :, :]
                tifffile.imwrite(os.path.join(output_dir, 'channel1', f'chan1_frame_{frame_counter}.tif'), chan1_chunk)
                tifffile.imwrite(os.path.join(output_dir, 'channel2', f'chan2_frame_{frame_counter}.tif'), chan2_chunk)
                frame_counter += 1

#####MAIN FUNCTION TO RUN EXAMPLE
# save_raw_frames_to_tifs(
#     file_path='/home/gianna/Desktop/2P_example_data/p10_acs11_blp_rg/2p/Image_001_001.raw',
#     width=1024,
#     height=1024,
#     output_dir='/home/gianna/Desktop/2P_example_data/p10_acs11_blp_rg/2p/tif_frames',
#     chunk_size=100
# )