import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

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
            
            :math:`M` is the number of bodies in the sequence (always 2).
            
            :math:`V` is the number of graph nodes (joints).
    """

    def __init__(self, data_file, label_file):
        # open pickled Numpy array while keeping it on-disk
        self.data = np.load(data_file, mmap_mode='r')

        # unpickle the label file in-memory
        with open(label_file, 'rb') as f:
            # self.labels = list(zip(*pickle.load(f))) # list of tuples of file name and class numbers
            self.labels = pickle.load(f)[1] # list of class numbers

    def __len__(self):
        # first dimension is the number of captures
        return self.data.shape[0]

    def __getitem__(self, index):
        # copy the read-only on-disk ndarray (1 video sequence) into a PyTorch tensor
        return torch.from_numpy(np.copy(self.data[index])), self.labels[index]
