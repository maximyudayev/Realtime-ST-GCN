import argparse
import numpy as np
import torch
import os

def permute(dir):
    for f in os.listdir(dir):
        data = np.float32(np.load('{0}/{1}'.format(dir, f)))
        with open('{0}/{1}'.format(dir, f), 'wb') as file:
            np.save(file, torch.from_numpy(data).permute(1,2,0,3).contiguous().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='SwapDatasetAxes',
        description='Script for swapping tensor axes in the dataset')

    parser.add_argument('path', type=str)

    args = parser.parse_args()

    permute(args.path)
