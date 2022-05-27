import numpy as np

def get_relative_coordinates(sample,
                             references=(4, 8, 12, 16)):
    # input: C, T, V, M
    # references=(4, 8, 12, 16)
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    C, t, V, M = sample.shape
    rel_coords = []
    #for i in range(len(references)):
    ref_loc = sample[:, :, references, :]
    coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
    rel_coords.append(coords_diff)
    
    # Shape: 4*C, t, V, M 
    rel_coords = np.vstack(rel_coords)
    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_coords
    return final_sample


def get_relative_coordinates_tcn(sample,
                             references=(4, 8, 12, 16)):
    # input: C, T, V, M
    # references=(4, 8, 12, 16)
    T, C = sample.shape
    final_sample = np.zeros((T, C))

    validFrames = (sample != 0).sum(axis=1) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[start:end, :]

    t = sample.shape[0]

    #iteratorx = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    iteratorx = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54]
    rel_coordsx = []
    # for i in range(len(references)):
    ref_loc = sample[:, 0]
    for v in iteratorx:
        coords_diff = (sample[:,v] - ref_loc)
        rel_coordsx.append(coords_diff)

    iteratory = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55]
    rel_coordsy = []
    # for i in range(len(references)):
    ref_loc = sample[:, 1]
    for w in iteratory:
        coords_diff = (sample[:, w] - ref_loc)
        rel_coordsy.append(coords_diff)

    iteratorz = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56]
    rel_coordsz = []
    # for i in range(len(references)):
    ref_loc = sample[:, 2]
    for h in iteratorz:
        coords_diff = (sample[:, h] - ref_loc)
        rel_coordsz.append(coords_diff)

    # Shape: t, c
    rel_coords = np.vstack((rel_coordsx, rel_coordsy, rel_coordsz)).T
    # Shape: C, T, V, M
    final_sample[start:end, :] = rel_coords
    return final_sample