import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("/usr/local/lib/python2.7/dist-packages")
import pandas as pd
import numpy as np
import os
from scipy.signal import butter, filtfilt

sacr = False

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def normalize(X):
    """Normalize the columns of X (zero mean, unit variance).
    Args:
        X (np array): data array.
    Returns:
        X_norm (np array): data array with normalized column values.

    """
    EPSILON = 1e-12  # to avoid divide by zero
    X = np.nan_to_num(X)
    X_norm = ((X - np.nanmean(X, axis=0))
              / (np.nanstd(X, axis=0) + EPSILON))
    return X_norm


def get_start_indices(n_timesteps, window_len, window_overlap):
    """Obtain window start indices.
    Args:
        n_timesteps (int): number of total available timesteps.
        window_len (int): number of timesteps to include per window.
        window_overlap (float): 0 to 1, how much consecutive windows
            should overlap.
    Returns:
        start_indices (np array): indices at which windows will start.
    """
    n_timesteps_valid = n_timesteps - window_len + 1
    step_size = int(window_len * (1 - window_overlap))
    if step_size <= 0:
        step_size = 1

    start_indices = np.arange(0, n_timesteps_valid, step_size,
        dtype=int)
    return start_indices


def extract_kinematics(fname):
    filename = input_dir + fname
    output_filename = output_dir + fname[:-4]
    print("Trying %s" % (filename))

    markers = ["lumbar", "ankle_l", "foot_l", "ankle_r", "foot_r"]
    df = pd.read_excel(filename)

    # ------------ Cols
    # If marker:
    # 2 * 4 * 3 = 24  marker trajectories
    # 1 * 1 * 3 = 3 SACR trajectories
    #           = 27

    # Get the pelvis

    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    traj = [None] * (len(markers))
    for i, v in enumerate(markers):
        traj[i] = df.filter(like=v).to_numpy(dtype='float')

    all_traj = np.hstack(traj)
    all_traj = butter_lowpass_filter(all_traj, cutoff=10, fs=128)

    # Get labels
    label_out = df.filter(like='freeze_label').to_numpy().squeeze()

    print("Writig %s" % filename)

    # window_len = 2500
    # start_indices = get_start_indices(len(all_traj), window_len,
    #                                   window_overlap=0)
    #
    # for k in start_indices:
    #     this_window_data = all_traj[k:k + window_len, :]
    #     this_window_label = label_out[k:k + window_len]
    np.savetxt(output_filename + '_input.csv', all_traj, delimiter=',')
    np.savetxt(output_filename + '_output.csv', label_out, delimiter=',')


if __name__ == "__main__":

    input_dir = "D:\\Stanford\\imus6_subjects7\\"
    output_dir = "D:\\Stanford\\out\\"

    files = os.listdir(input_dir)
    for filename in files:
        extract_kinematics(filename)
