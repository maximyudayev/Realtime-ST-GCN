import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("/usr/local/lib/python2.7/dist-packages")
import pandas as pd
import numpy as np
import btk
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

def extract_kinematics(fname):
    filename = input_dir + fname
    output_filename = output_dir + fname[:-4]
    print("Trying %s" % (filename))

    # Open c3d and read data
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(str(filename))
    reader.Update()
    acq = reader.GetOutput()
    nframes = acq.GetPointFrameNumber()

    start = acq.GetFirstFrame()
    end = acq.GetLastFrame()

    markers = ["LASI", "LKNE", "LANK", "LTOE", "RASI", "RKNE", "RANK", "RTOE"]

    # ------------ Cols
    # If marker:
    # 2 * 4 * 3 = 24  marker trajectories
    # 1 * 1 * 3 = 3 SACR trajectories
    #           = 27

    # Get the pelvis
    if sacr:
        SACR = acq.GetPoint("SACR").GetValues()
    else:
        LPSI = acq.GetPoint("LPSI").GetValues()
        RPSI = acq.GetPoint("RPSI").GetValues()
        SACR = (LPSI + RPSI) / 2

    # incrementX = 1 if midASI[100][0] > midASI[0][0] else -1

    traj = [None] * (len(markers))
    for i, v in enumerate(markers):
        try:
            traj[i] = acq.GetPoint(v).GetValues()
        except:
            print("Marker error")
            return

    all_traj = np.concatenate((SACR, np.hstack(traj)), axis=1)
    all_traj = butter_lowpass_filter(all_traj, cutoff=10, fs=100)

    # Add events as output
    outputs = np.array([[0] * nframes]).T
    for event in btk.Iterate(acq.GetEvents()):
        if start <= event.GetFrame() <= end:
            if event.GetLabel() == "Foot Strike":
                if event.GetContext() == 'Left':
                    outputs[event.GetFrame() - start, 0] = 1
                elif event.GetContext() == 'Right':
                    outputs[event.GetFrame() - start, 0] = 2
            elif event.GetLabel() == "Foot Off":
                if event.GetContext() == 'Left':
                    outputs[event.GetFrame() - start, 0] = 3
                elif event.GetContext() == 'Right':
                    outputs[event.GetFrame() - start, 0] = 4

    if (np.sum(outputs) == 0):
        print("No events in %s!" % (filename,))
        return

    arr = np.concatenate((all_traj, outputs), axis=1)

    # Remove data before and after first and last event
    positives = np.where(arr[:, 27] > 0.5)
    if len(positives[0]) == 0:
        return None

    first_event = positives[0][0]
    last_event = positives[0][-1]
    all_traj = all_traj[first_event:last_event] * 0.001

    # Get labels
    labels_start = np.array([[0] * (last_event - first_event)]).T
    labels_end = np.array([[0] * (last_event - first_event)]).T
    for event in btk.Iterate(acq.GetEvents()):
        if first_event < event.GetFrame() < last_event:
            if event.GetLabel() == "FOGs":
                labels_start[event.GetFrame() - first_event, 0] = 1
            if event.GetLabel() == "FOGe":
                labels_end[event.GetFrame() - first_event, 0] = 1

    FOGs = np.where(labels_start == 1)[0]
    FOGe = np.where(labels_end == 1)[0]

    label_out = np.zeros((all_traj.shape[0]))

    for k in range(len(FOGs)):
        label_out[FOGs[k]:FOGe[k]] = 1

    if len(FOGs) != len(FOGe):
        print("Unequal start and end events")
        return

    print("Writig %s" % filename)
    np.savetxt(output_filename + '_input_fog.csv', all_traj, delimiter=',')
    np.savetxt(output_filename + '_output_fog.csv', label_out, delimiter=',')


if __name__ == "__main__":

    if sacr:
        input_dir = "D:\\GraphNN\\final_dataset\\Spildooren\\fog_dataset_spildooren_fog\\"
        output_dir = "D:\\GraphNN\\final_dataset\\Spildooren\\spildooren_out_fog_filtered10\\"
    else:
        input_dir = "D:\\GraphNN\\final_dataset\\Vervoort\\fog_dataset_vervoort_fog\\"
        output_dir = "D:\\GraphNN\\final_dataset\\Vervoort\\Vervoort_out_filtered10\\"

    files = os.listdir(input_dir)
    for filename in files:
        extract_kinematics(filename)
