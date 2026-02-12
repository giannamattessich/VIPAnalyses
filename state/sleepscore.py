from utils.getDataFiles import *
from state.sleepScoring import *
from vip_helpers import *

import os
import subprocess
from pathlib import Path

MATLAB_BIN = "/usr/local/MATLAB/R2025b/bin/matlab"  # adjust if needed
BUZCODE_PATH = "/home/gianna/Documents/MATLAB/buzcode"
recording = "/mnt/Batista_Lab/Joslyn/Experiments/VIPxTiger/dob_8.3.25/preprocessed/Back_left/p17"

def _matlab_str(s: str) -> str:
    """Escape Python string to a MATLAB single-quoted string."""
    return "'" + s.replace("'", "''") + "'"

def run_sleep_score_for_rec(rec_path: str,
                            matlab_bin: str = MATLAB_BIN,
                            buzcode_path: str = BUZCODE_PATH,
                            bad_channels:  list = None) -> None:
    """
    rec_path: full path to the recording folder (the one that contains the xml, basepath, etc.).

    This replicates, for each recording:

        matlab -batch "addpath(genpath(buzcode_path));
                       SleepScoreMaster('basepath','/full/path/to/rec','noPrompts',true);"
    """
    rec_path = os.path.abspath(rec_path)
    buzcode_path = os.path.abspath(buzcode_path)

    # Build the MATLAB command string
    mat_cmd = (
        f"addpath(genpath({_matlab_str(buzcode_path)}));"
        f"SleepScoreMaster({_matlab_str(rec_path)}, 'noPrompts', true);"
    )

    cmd = [matlab_bin, "-batch", mat_cmd]

    # We *can* set cwd to the rec parent or just leave it; since we pass full basepath, cwd is less critical.
    parent_dir = str(Path(rec_path).parent)

    print(f"[MATLAB] Running SleepScoreMaster('basepath','{rec_path}','noPrompts',true)")
    result = subprocess.run(
        cmd,
        cwd=parent_dir,          # safe; can also be rec_path itself
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"[MATLAB] ERROR for {rec_path}")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        raise RuntimeError(f"SleepScoreMaster failed for {rec_path}")
    else:
        print(f"[MATLAB] Done for {rec_path}")
        # Optional: inspect result.stdout if you want logs


def sleep_score_allrecs(all_rec_paths,
                        buzcode_path: str = BUZCODE_PATH,
                        matlab_bin: str = MATLAB_BIN):
    for path in all_rec_paths:
        if not os.path.exists(os.path.join(path, f'{os.path.basename(path)}.SleepState.states.mat')):
            try:
                run_sleep_score_for_rec(path, matlab_bin=matlab_bin, buzcode_path=buzcode_path)
                sleep_state_ints, _ = get_state_intervals_df(path)
                flattened_ints_df = flatten_state_ints_df(sleep_state_ints)
                flattened_ints_df.to_parquet(os.path.join(path, 'sleep_state_df.parquet'))
            except:
                traceback.print_exc()
                print(f'Sleep scoring failed for {path}')
                continue
        # do whatever you want with flattened_ints_df

        # do whatever with flattened_ints_df

# def sleep_score_allrecs(all_rec_paths,
#                         buzcode_path: str = BUZCODE_PATH,
#                         matlab_bin: str = MATLAB_BIN):
#     for path in all_rec_paths:
#         basepath = Path(path)
#         ss_file = basepath / f"{basepath.name}.SleepState.states.mat"

#         if ss_file.exists():
#             print(f"[PY] SleepState file already exists for {basepath}, skipping MATLAB.")
#         else:
#             run_sleep_score_for_rec(str(basepath), matlab_bin=matlab_bin, buzcode_path=buzcode_path)

#         # Now do your usual Python side
#         sleep_state_ints, _ = get_state_intervals_df(str(basepath))
#         flattened_ints_df = flatten_state_ints_df(sleep_state_ints)
#         # do whatever you need with flattened_ints_df

# # 1. get rec paths
# sleep_score_allrecs(datapaths, buzcode_path='/home/gianna/Documents/MATLAB/buzcode')
# import matlab.engine
# #matlab_root = '/usr/local/MATLAB/R2025b'
# buzcode_path = '/home/gianna/Documents/MATLAB/buzcode'
# eng = matlab.engine.start_matlab()
# if not os.path.exists(buzcode_path):
#     raise ValueError(f'Could not find path to buzcode library!')
# eng.addpath(buzcode_path, nargout=0)
# eng.quit()