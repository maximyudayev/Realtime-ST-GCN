import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class SkeletonDataset(Dataset):
    r"""Custom Dataset for lazy loading large skeleton Numpy files.
    
    Reads-in single captures of length L and spits out a pair of PyTorch 
    tensors and action label.

    Args:
        data_file           (int): Number of input sample channels/features
        
        label_file          (int): Number of output classification classes

    Shape:
        - Output[0]: :math:`<class 'torch.Tensor'> (C, L, V, M)`
        - Output[1]: 
        
        where
            :math:`C` is the number of data channels (features),
            :math:`L` is the capture sequence length,
            :math:`M` is the number of bodies in the sequence (always 2),
            :math:`V` is the number of graph nodes (joints).
    """
    def __init__(self, data_file, label_file):
        # Open pickled Numpy array while keeping it on-disk
        self.data = np.load(data_file, mmap_mode='r')

        # Unpickle the label file in-memory
        with open(label_file, 'rb') as f:
            # self.labels = list(zip(*pickle.load(f))) # list of tuples of file name and class numbers
            self.labels = pickle.load(f)[1] # list of class numbers

    def __len__(self):
        # First dimension is the number of captures
        return self.data.shape[0]

    def __getitem__(self, index):
        # Copy the read-only on-disk ndarray (1 video sequence) into a PyTorch tensor
        return torch.from_numpy(np.copy(self.data[index])), self.labels[index]
