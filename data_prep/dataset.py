import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

class SkeletonDataset(Dataset):
    """Custom Dataset for lazy loading large skeleton Numpy files.
    
    Reads-in captures of length ``L`` and spits out pairs of a PyTorch 
    tensor and an action label corresponding to it.

    Args:
        data_file : ``str``
            Path to the .npy data file.
        
        label_file: ``str``
            Path to the .pkl data labels file.

    Shape:
        - Output[0]: :math:`<class 'torch.Tensor'> (C, L, V, M)`.
        
        where
            :math:`C` is the number of data channels (features).
            
            :math:`L` is the capture sequence length.
            
            :math:`V` is the number of graph nodes (joints).
            
            :math:`M` is the number of bodies in the sequence (always 1st priority body in our implementation).
    """

    def __init__(self, data_file, label_file):
        # open pickled Numpy array while keeping it on-disk
        self.data = np.load(data_file, mmap_mode='r')[:,:,:,:,0]

        # unpickle the label file in-memory
        with open(label_file, 'rb') as f:
            # self.labels = list(zip(*pickle.load(f))) # list of tuples of file name and class numbers
            self.labels = pickle.load(f)[1] # list of class numbers

    def __len__(self):
        # first dimension is the number of captures
        return self.data.shape[0]

    def __getitem__(self, index):
        # copy the read-only on-disk ndarray (1 video sequence) into a PyTorch tensor
        data = np.copy(self.data[index])
        # increment the ground truth label to map a C class non-empty classification dataset to a 
        # C+1 class segmentation dataset with 'background/none' class
        labels = torch.tensor(self.labels[index]+1)
        # broadcast the labels across the capture length dimension for framewise comparison to predictions
        # expanding labels tensor does not allocate new memory, only creates a new view on existing tensor
        return torch.from_numpy(data), labels[None].expand(data.shape[1])


class SkeletonDatasetFromDirectory(Dataset):
    """Custom Dataset for lazy loading directory of differently-sized skeleton Numpy files.
    
    Reads-in captures of length ``L`` individually and spits out a pair of a PyTorch 
    tensor and a ground truth tensor corresponding to it.

    Note:
        Batch size must equal 1 (or equal to the number of GPUs), since trials have different durations. 

    Args:
        data_dir : ``str``
            Path to the .npy data directory.
        
        label_dir: ``str``
            Path to the .csv labels directory.

    Shape:
        - Output[0]: :math:`<class 'torch.Tensor'> (C, L, V, M)`.
        
        where
            :math:`C` is the number of data channels (features).
            
            :math:`L` is the capture sequence length.
            
            :math:`V` is the number of graph nodes (joints).

            :math:`M` is the number of bodies in the sequence (always 1 in this processed dataset).     
    """

    def __init__(self, data_dir, label_dir):
        self.data = data_dir
        self.labels = label_dir

        # create a list for indexing and mapping between features and labels
        self.dir_list = [file.split('.npy')[0] for file in os.listdir(self.data)]

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        data = np.float32(np.load('{0}/{1}.npy'.format(self.data, self.dir_list[index]))[:,:,:,0])
        labels = np.int64(pd.read_csv('{0}/{1}.csv'.format(self.labels, self.dir_list[index]), header=None).values[:,0])
        
        return torch.from_numpy(data), torch.from_numpy(labels)